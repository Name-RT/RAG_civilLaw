"""
llm_service.py — บริการเรียกใช้ LLM (DeepSeek / OpenAI / Gemini)

ความรับผิดชอบ:
  1. expand_query_to_legal_terms()  — แปลงภาษาชีวิตประจำวันเป็นศัพท์กฎหมาย
                                     ก่อนนำไปค้นใน Vector DB (Query Expansion)
  2. generate_legal_analysis()      — วิเคราะห์คำถามร่วมกับมาตรากฎหมายที่ดึงมา
                                     และให้คำแนะนำทางกฎหมายแบบเฉพาะบุคคล

ลำดับการเรียก LLM:
  Admin Panel → active_llm (primary) → fallback ตัวอื่นโดยอัตโนมัติ
  DeepSeek (flash) → OpenAI (luna) → Gemini (3.5-flash)
"""

import os
import json
from typing import Optional

from config import (
    LLM_GEMINI_MODEL,
    LLM_OPENAI_MODEL,
    LLM_DEEPSEEK_MODEL,
    LLM_DEEPSEEK_URL,
)

# ──────────────────────────────────────────────
# Disclaimer ต่อท้ายทุกคำตอบจาก AI
# ──────────────────────────────────────────────
_DISCLAIMER = (
    "\n\n⚠️ **คำเตือน:** นี่คือการวิเคราะห์เบื้องต้นโดย AI "
    "ไม่ใช่คำปรึกษาทางกฎหมายที่มีผลผูกพัน โปรดปรึกษาทนายความ"
)


# ══════════════════════════════════════════════
# Helper: LLM Clients
# ══════════════════════════════════════════════

def _get_priority_list() -> list[str]:
    """
    อ่านค่า active_llm จาก Admin Panel แล้วคืนลำดับ fallback

    Returns:
        list ของชื่อ provider เรียงตามลำดับที่จะลอง
        เช่น ["deepseek", "openai", "gemini"]
    """
    from admin import load_settings
    settings   = load_settings()
    active_llm = settings.get("active_llm", "deepseek")
    priority   = [active_llm]
    for llm in ["deepseek", "openai", "gemini"]:
        if llm != active_llm:
            priority.append(llm)
    return priority


def _get_openai_compat_client(llm: str):
    """
    สร้าง OpenAI-compatible client สำหรับ DeepSeek หรือ OpenAI

    Args:
        llm: "deepseek" หรือ "openai"

    Returns:
        tuple (client, model_name) หรือ (None, None) ถ้าไม่มี API key
    """
    from openai import OpenAI

    if llm == "deepseek":
        key = os.getenv("DEEPSEEK_API_KEY")
        if key:
            return OpenAI(api_key=key, base_url=LLM_DEEPSEEK_URL), LLM_DEEPSEEK_MODEL

    elif llm == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return OpenAI(api_key=key), LLM_OPENAI_MODEL

    return None, None


def _call_gemini(prompt: str) -> Optional[dict]:
    """
    เรียก Gemini API และ parse JSON response

    Args:
        prompt: prompt ที่ต้องการส่งให้ Gemini

    Returns:
        dict จาก JSON หรือ None ถ้าเกิด error
    """
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        model = genai.GenerativeModel(
            model_name=LLM_GEMINI_MODEL,
            generation_config={"response_mime_type": "application/json"},
        )
        return json.loads(model.generate_content(prompt).text)
    except Exception as e:
        print(f"⚠️ Gemini API failed: {e}")
        return None


def _call_openai_compat(client, model_name: str, system: str, user: str) -> Optional[dict]:
    """
    เรียก OpenAI-compatible API และ parse JSON response

    Args:
        client    : OpenAI client (DeepSeek หรือ OpenAI)
        model_name: ชื่อโมเดล
        system    : system prompt
        user      : user prompt

    Returns:
        dict จาก JSON หรือ None ถ้าเกิด error
    """
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"⚠️ LLM call failed ({model_name}): {e}")
        return None


def _try_all_llms(prompt: str, system: str = "คุณคือผู้ช่วยกฎหมายมืออาชีพ ตอบในรูปแบบ JSON เสมอ") -> Optional[dict]:
    """
    ลองเรียก LLM ทุกตัวตามลำดับ priority จนกว่าจะสำเร็จ

    Args:
        prompt: user prompt
        system: system prompt

    Returns:
        dict ผลลัพธ์ หรือ None ถ้าทุกตัวล้มเหลว
    """
    for llm in _get_priority_list():
        result = None
        if llm == "gemini":
            result = _call_gemini(prompt)
        else:
            client, model_name = _get_openai_compat_client(llm)
            if client:
                print(f"🤖 Calling {llm} ({model_name})...")
                result = _call_openai_compat(client, model_name, system, prompt)
        if result:
            return result
    return None


# ══════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════

def expand_query_to_legal_terms(query: str) -> str:
    """
    [Step 1 ของ RAG Pipeline] แปลงคำถามภาษาชีวิตประจำวันเป็นศัพท์กฎหมาย

    ทำไมต้องทำ:
        คนถาม "ถูกรถชน" แต่ฐานข้อมูลมีข้อความ "ผู้ใดกระทำโดยประมาท"
        การแปลงช่วยให้ vector similarity score สูงขึ้น ค้นเจอมากขึ้น

    Args:
        query: คำถามต้นฉบับจากผู้ใช้

    Returns:
        คำค้นหาในภาษากฎหมาย (หรือคำถามเดิมถ้า LLM ล้มเหลว)

    Example:
        "ถูกรถชน" → "ละเมิด ประมาท ค่าสินไหมทดแทน ความรับผิดทางแพ่ง"
    """
    prompt = (
        f'แปลงคำถามต่อไปนี้ให้เป็นคำค้นหาทางกฎหมาย สั้นๆ ไม่เกิน 30 คำ\n'
        f'โดยใช้ศัพท์กฎหมายที่ตรงกับประมวลกฎหมายแพ่งและพาณิชย์ หรือประมวลกฎหมายอาญาของไทย\n\n'
        f'คำถาม: "{query}"\n\n'
        f'ตอบในรูปแบบ JSON:\n'
        f'{{"legal_query": "ตัวอย่าง: ละเมิด ประมาท ค่าสินไหมทดแทน ความรับผิดในทางแพ่ง"}}'
    )

    result = _try_all_llms(prompt, system="คุณคือผู้เชี่ยวชาญกฎหมายไทย ตอบเป็น JSON เสมอ")
    if result and result.get("legal_query"):
        expanded = result["legal_query"]
        print(f"🔍 Query Expansion: '{query}' → '{expanded}'")
        return expanded

    # Fallback: ใช้คำถามเดิม
    return query


def generate_legal_analysis(query: str, retrieved_results: list) -> Optional[dict]:
    """
    [Step ท้ายของ RAG Pipeline] วิเคราะห์และให้คำแนะนำทางกฎหมาย

    มี 2 โหมด:
      1. Normal mode  — retrieved_results มีข้อมูล → วิเคราะห์จากมาตราที่ดึงมา
      2. Fallback mode — retrieved_results ว่าง   → ให้ LLM ตอบจากความรู้โดยตรง
                         พร้อมระบุมาตราที่คาดว่าเกี่ยวข้องในรูปแบบ "มาตรา xxx"
                         เพื่อให้ผู้ใช้กดดูรายละเอียดได้

    Args:
        query            : คำถามต้นฉบับจากผู้ใช้
        retrieved_results: list ของ dict {section, category, original_text, simplified_text, score}

    Returns:
        dict {has_analysis, relevant_sections, general_meaning, recommendation}
        หรือ None ถ้า LLM ทุกตัวล้มเหลว
    """
    if retrieved_results:
        prompt = _build_analysis_prompt(query, retrieved_results)
    else:
        prompt = _build_fallback_prompt(query)

    result = _try_all_llms(prompt)
    if not result:
        return None

    return {
        "has_analysis":      True,
        "relevant_sections": result.get("relevant_sections", "").strip(),
        "general_meaning":   result.get("general_meaning",   "").strip(),
        "recommendation":    result.get("recommendation",    "").strip() + _DISCLAIMER,
    }


# ──────────────────────────────────────────────
# Prompt Builders (แยกออกมาเพื่อความอ่านง่าย)
# ──────────────────────────────────────────────

def _build_analysis_prompt(query: str, results: list) -> str:
    """สร้าง prompt สำหรับวิเคราะห์จากมาตราที่ดึงมาได้"""
    context = "\n---\n".join(
        f"มาตรา: {r['section']}\n"
        f"หมวดหมู่: {r['category']}\n"
        f"ตัวบทกฎหมาย: {r['original_text']}\n"
        f"คำอธิบาย: {r['simplified_text']}"
        for r in results
    )
    return (
        f'คุณคือผู้เชี่ยวชาญกฎหมายไทย วิเคราะห์คำถามร่วมกับมาตรากฎหมายในฐานข้อมูล\n\n'
        f'คำถาม: "{query}"\n\n'
        f'มาตรากฎหมายที่ดึงมาได้:\n{context}\n\n'
        f'คำแนะนำ:\n'
        f'1. ตัดมาตราที่ไม่เกี่ยวข้องออก\n'
        f'2. ระบุมาตราในรูปแบบ "มาตรา xxx" คั่นด้วย ", " เพื่อให้ผู้ใช้กดดูได้\n'
        f'3. สรุปความหมายในภาษาพูด\n'
        f'4. ให้คำแนะนำปฏิบัติพร้อมหลักฐานที่ต้องเตรียม\n\n'
        f'ตอบ JSON: {{"relevant_sections":"...", "general_meaning":"...", "recommendation":"..."}}'
    )


def _build_fallback_prompt(query: str) -> str:
    """สร้าง prompt สำหรับกรณีที่ Vector DB ไม่มีผลลัพธ์"""
    return (
        f'คุณคือผู้เชี่ยวชาญกฎหมายไทย ระบบไม่พบมาตราตรงในฐานข้อมูล\n'
        f'จงตอบจากความรู้ทางกฎหมายของคุณเอง\n\n'
        f'คำถาม: "{query}"\n\n'
        f'คำแนะนำ:\n'
        f'1. ระบุหมวดกฎหมายที่เกี่ยวข้อง\n'
        f'2. ระบุมาตราในรูปแบบ "มาตรา xxx" คั่นด้วย ", " เพื่อให้ผู้ใช้กดดูได้\n'
        f'3. ให้คำแนะนำปฏิบัติพร้อมหลักฐานที่ต้องเตรียม\n'
        f'4. ย้ำให้ผู้ใช้ปรึกษาทนายความ\n\n'
        f'ตอบ JSON: {{"relevant_sections":"...", "general_meaning":"...", '
        f'"recommendation":"... [หมายเหตุ: ไม่พบมาตราตรงในฐานข้อมูล คำแนะนำนี้มาจากความรู้ AI]"}}'
    )
