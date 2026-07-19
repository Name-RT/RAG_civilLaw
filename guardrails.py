"""
guardrails.py — ระบบความปลอดภัยของ AuditGuard AI

ความรับผิดชอบ:
  1. check_prompt_injection()  — ตรวจจับคำสั่งหลอก AI (Jailbreak / Prompt Injection)
  2. check_query_relevance()   — ตรวจสอบว่าคำถามเกี่ยวกับกฎหมายหรือไม่
                                 (ป้องกันการใช้ระบบในทางที่ผิดวัตถุประสงค์)
  3. filter_by_threshold()     — ตัด candidate ที่ similarity score ต่ำกว่าเกณฑ์ทิ้ง

Pipeline:
  User Query → [1] Injection Check → [2] Relevance Check → RAG Pipeline → [3] Score Filter
"""

import os
import json
from typing import Tuple

from config import LLM_GEMINI_MODEL, LLM_DEEPSEEK_MODEL, LLM_DEEPSEEK_URL, LLM_OPENAI_MODEL


# ──────────────────────────────────────────────
# 1. Prompt Injection Detection (Rule-based)
# ──────────────────────────────────────────────

_INJECTION_KEYWORDS = [
    # English
    "ignore previous", "forget previous", "system prompt", "forget all",
    "act as", "pretend to be", "bypass", "jailbreak", "DAN", "do anything now",
    # Thai
    "ลืมคำสั่ง", "ข้ามคำสั่ง", "ลืมกฎ", "ข้ามกฎ", "จงทำตัวเป็น",
]

def check_prompt_injection(query: str) -> bool:
    """
    ตรวจสอบ Prompt Injection ด้วย keyword matching

    Args:
        query: คำถามจากผู้ใช้

    Returns:
        True  = พบความเสี่ยง (ควรปฏิเสธ)
        False = ปลอดภัย
    """
    query_lower = query.lower()
    return any(kw in query_lower for kw in _INJECTION_KEYWORDS)


# ──────────────────────────────────────────────
# 2. Legal Relevance Check (LLM-based)
# ──────────────────────────────────────────────

_RELEVANCE_PROMPT_TEMPLATE = """วิเคราะห์คำถามต่อไปนี้ว่าเกี่ยวข้องกับกฎหมาย ข้อพิพาท หรือปัญหาที่อาจต้องใช้กฎหมายแก้ไขหรือไม่
คำถาม: "{query}"

กฎเกณฑ์:
- หากเป็นคำถามสัพเพเหระ (เช่น สูตรอาหาร สภาพอากาศ เล่นเกม ขอหวย) ให้ตอบ false
- หากเป็นคำถามการเมืองทั่วไป ที่ไม่ใช่ข้อกฎหมาย ให้ตอบ false
- หากเป็นคำถามเชิงกฎหมาย คดีความ สัญญา ทรัพย์สิน หนี้สิน ครอบครัว อุบัติเหตุ ให้ตอบ true
- หากกำกวมแต่มีโอกาสเกี่ยวกับกฎหมาย ให้ตอบ true ไว้ก่อน

ตอบในรูปแบบ JSON:
{{"is_legal_query": true หรือ false, "reason": "เหตุผลสั้นๆ"}}"""

_REJECT_MESSAGE = (
    "ขออภัย ฉันคือผู้ช่วยด้านกฎหมาย "
    "ไม่สามารถให้ข้อมูลหรือคำปรึกษาในเรื่องที่ไม่เกี่ยวข้องกับกฎหมายได้ครับ"
)


def check_query_relevance(query: str) -> Tuple[bool, str]:
    """
    ตรวจสอบว่าคำถามเกี่ยวข้องกับกฎหมายหรือไม่ โดยใช้ LLM

    ลำดับ LLM: ตามการตั้งค่าใน Admin Panel (active_llm) → fallback อัตโนมัติ
    ถ้าทุกตัวล้มเหลว: ให้ผ่านเสมอเพื่อไม่ให้ระบบค้าง

    Args:
        query: คำถามจากผู้ใช้

    Returns:
        (True, "")              = เกี่ยวข้อง ผ่านได้
        (False, reason_message) = ไม่เกี่ยวข้อง พร้อมข้อความปฏิเสธ
    """
    prompt = _RELEVANCE_PROMPT_TEMPLATE.format(query=query)

    for llm in _get_priority_list():
        result = _call_llm_for_relevance(llm, prompt)
        if result is not None:
            if not result.get("is_legal_query", True):
                return False, _REJECT_MESSAGE
            return True, ""

    # Fallback: allow through ถ้าทุก LLM ล้มเหลว
    return True, ""


# ──────────────────────────────────────────────
# 3. Score Threshold Filter
# ──────────────────────────────────────────────

def filter_by_threshold(retrieved_results: list, min_score: float = 0.4) -> list:
    """
    คัดกรอง candidate ที่ similarity score ต่ำกว่าเกณฑ์ออก

    Args:
        retrieved_results: list ของ dict ที่มี key "score"
        min_score        : คะแนนต่ำสุดที่ยอมรับ (default 0.4 สำหรับ multilingual-e5-small)

    Returns:
        list ของ candidate ที่ผ่านเกณฑ์
    """
    return [item for item in retrieved_results if item.get("score", 1.0) >= min_score]


# ──────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────

def _get_priority_list() -> list[str]:
    """อ่าน active_llm จาก Admin Panel และคืนลำดับ fallback"""
    from admin import load_settings
    settings   = load_settings()
    active_llm = settings.get("active_llm", "deepseek")
    priority   = [active_llm]
    for llm in ["deepseek", "openai", "gemini"]:
        if llm != active_llm:
            priority.append(llm)
    return priority


def _call_llm_for_relevance(llm: str, prompt: str) -> dict | None:
    """
    เรียก LLM หนึ่งตัวเพื่อประเมิน relevance และคืน dict

    Returns:
        dict {"is_legal_query": bool, "reason": str} หรือ None ถ้าล้มเหลว
    """
    gemini_key   = os.getenv("GEMINI_API_KEY")
    openai_key   = os.getenv("OPENAI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")

    try:
        if llm == "gemini" and gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                model_name=LLM_GEMINI_MODEL,
                generation_config={"response_mime_type": "application/json", "temperature": 0.1},
            )
            return json.loads(model.generate_content(prompt).text)

        elif llm in ("deepseek", "openai"):
            api_key  = deepseek_key if llm == "deepseek" else openai_key
            base_url = LLM_DEEPSEEK_URL if llm == "deepseek" else None
            model_name = LLM_DEEPSEEK_MODEL if llm == "deepseek" else LLM_OPENAI_MODEL

            if not api_key:
                return None

            from openai import OpenAI
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            client = OpenAI(**kwargs)

            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "คุณต้องตอบในรูปแบบ JSON ตามคำสั่งเสมอ"},
                    {"role": "user",   "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            return json.loads(resp.choices[0].message.content)

    except Exception as e:
        print(f"⚠️ Guardrail {llm} failed: {e}")

    return None
