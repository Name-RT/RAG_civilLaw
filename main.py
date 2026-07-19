"""
main.py — Entry Point ของ AuditGuard AI API

เซิร์ฟเวอร์ FastAPI ที่รวมทุก module เข้าด้วยกัน

Endpoints:
  GET  /               → serve index.html (หน้าค้นหากฎหมาย)
  GET  /admin/*        → serve หน้า Admin Panel (static/admin.html)
  POST /ask            → RAG pipeline หลัก: รับคำถาม → ค้นกฎหมาย → วิเคราะห์ด้วย AI
  GET  /section/{num}  → ดึงตัวบทกฎหมายตามเลขมาตรา
  GET  /health         → ตรวจสอบสถานะ backend
  GET/POST /api/admin/* → Admin API (จัดการ settings, ข้อมูลกฎหมาย)

RAG Pipeline ใน /ask:
  1. Security: Injection check → Relevance check (guardrails.py)
  2. Query Expansion: ภาษาชีวิตประจำวัน → ศัพท์กฎหมาย (llm_service.py)
  3. Retrieval: Keyword classify → Vector search in Qdrant (database.py)
  4. Filtering: Score threshold → Deduplication → Reranking (optional)
  5. Analysis: LLM วิเคราะห์และให้คำแนะนำ (llm_service.py)
"""

import os
import sys
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

from config import (
    INDEX_HTML, STATIC_DIR,
    RATE_LIMIT_ASK, RATE_LIMIT_SECTION,
    LAWS_FILE,
)
from utils import classify_query, has_simplified_explanation, convert_arabic_to_thai
from database import db_manager
from llm_service import generate_legal_analysis, expand_query_to_legal_terms
from guardrails import check_query_relevance, filter_by_threshold, check_prompt_injection
from admin import admin_router, load_settings

# ──────────────────────────────────────────────
# ตั้งค่า stdout เป็น UTF-8 (แก้ปัญหาภาษาไทย crash บน Windows)
# ──────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

load_dotenv()

# ──────────────────────────────────────────────
# Rate Limiter
# ──────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ──────────────────────────────────────────────
# Lifespan: รันเมื่อ startup และ shutdown
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """เชื่อมต่อ Qdrant และเตรียม Vector Index เมื่อ server เริ่มทำงาน"""
    db_manager.connect_and_initialize()
    yield
    # (Optional) cleanup on shutdown


# ──────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="ThaiLaw AI — Thai Law Search & Analysis API",
    description=(
        "ระบบค้นหาและวิเคราะห์ข้อกฎหมายแพ่ง-อาญาของไทยด้วย RAG + LLM\n\n"
        "รองรับ LLM: DeepSeek / OpenAI / Gemini (สลับได้จาก Admin Panel)"
    ),
    version="2.0.0",
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ในโปรดักชันควรกำหนด origin ที่อนุญาต
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Routes
app.include_router(admin_router)
app.mount("/admin", StaticFiles(directory=STATIC_DIR, html=True), name="static")


# ══════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve หน้าเว็บหลัก (index.html)"""
    return FileResponse(INDEX_HTML)


class QueryRequest(BaseModel):
    """Request body สำหรับ POST /ask"""
    query: str
    top_k: int = 5


@app.post("/ask", summary="ค้นหาและวิเคราะห์กฎหมาย")
@limiter.limit(RATE_LIMIT_ASK)
async def ask_law(request: Request, body: QueryRequest):
    """
    RAG Pipeline หลักของระบบ

    Flow:
        1. [Security]   ตรวจ Prompt Injection → ตรวจความเกี่ยวข้องกับกฎหมาย
        2. [Expansion]  แปลงคำถามเป็นศัพท์กฎหมาย (Query Expansion ด้วย LLM)
        3. [Retrieval]  Classify หมวดหมู่ → ค้นใน Qdrant ด้วย Vector Similarity
        4. [Filter]     ตัด score ต่ำ → deduplicate → (optional) rerank
        5. [Analysis]   LLM วิเคราะห์ผลและให้คำแนะนำ
                        → ถ้าไม่พบใน DB: LLM ตอบจากความรู้โดยตรง

    Args:
        body.query: คำถามจากผู้ใช้
        body.top_k: จำนวนมาตราที่ต้องการแสดง (default 5)

    Returns:
        {query, detected_category, found, results[], legal_analysis{}}
    """
    # ── Guard: ตรวจสอบ database ──
    if not db_manager.index:
        raise HTTPException(
            status_code=503,
            detail="ฐานข้อมูลเวกเตอร์ยังไม่พร้อม กรุณาตรวจสอบ Qdrant Container",
        )
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="กรุณากรอกคำถามที่ต้องการค้นหา")

    # ── Step 1: Security ──
    if check_prompt_injection(body.query):
        return _build_error_response(body.query, "ระบบตรวจพบความเสี่ยง Prompt Injection")

    is_relevant, reject_reason = check_query_relevance(body.query)
    if not is_relevant:
        return _build_error_response(body.query, reject_reason)

    try:
        settings  = load_settings()
        top_k     = body.top_k or settings.get("top_k", 5)
        min_score = settings.get("similarity_threshold", 0.4)
        # ดึง candidate มากกว่า top_k เผื่อตัดและ rerank ภายหลัง
        candidate_k = max(35, top_k * 6)

        # ── Step 2: Query Expansion ──
        expanded_query = expand_query_to_legal_terms(body.query)

        # ── Step 3: Retrieval ──
        detected_category = classify_query(body.query)
        retrieved_nodes   = _retrieve_nodes(expanded_query, detected_category, candidate_k)

        # ── Step 4: Filter + Deduplicate ──
        raw_results     = _process_nodes(retrieved_nodes)
        top_results     = _apply_reranker(body.query, raw_results, settings, top_k)
        filtered_results = filter_by_threshold(top_results, min_score=min_score)

        # ── Step 5: LLM Analysis (or Fallback) ──
        if not filtered_results:
            print("⚠️ ไม่พบผลลัพธ์ใน Vector DB → LLM ตอบจากความรู้โดยตรง")
            return _build_llm_fallback_response(body.query, detected_category)

        legal_analysis = generate_legal_analysis(body.query, filtered_results) or {
            "has_analysis": False, "relevant_sections": "", "general_meaning": "", "recommendation": "",
        }

        return {
            "query":             body.query,
            "detected_category": detected_category,
            "found":             len(filtered_results),
            "results":           filtered_results,
            "legal_analysis":    legal_analysis,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการประมวลผล: {e}")


@app.get("/section/{num}", summary="ดึงตัวบทกฎหมายตามเลขมาตรา")
@limiter.limit(RATE_LIMIT_SECTION)
async def get_section(request: Request, num: str):
    """
    ค้นหาตัวบทกฎหมายจากเลขมาตรา รองรับทั้งเลขอารบิกและเลขไทย

    Args:
        num: เลขมาตรา เช่น "420" หรือ "๔๒๐"

    Returns:
        {section, category, original_text, simplified_text}
    """
    if not os.path.exists(LAWS_FILE):
        raise HTTPException(status_code=500, detail="ไม่พบไฟล์ข้อมูลกฎหมาย")

    thai_num = convert_arabic_to_thai(num)
    targets  = {f"มาตรา {num}", f"มาตรา {thai_num}"}

    try:
        with open(LAWS_FILE, "r", encoding="utf-8") as f:
            laws_data = json.load(f)

        for item in laws_data:
            if item["section"].strip() in targets:
                return {
                    "section":        item["section"],
                    "category":       item["category"],
                    "original_text":  item["original_text"],
                    "simplified_text": item.get("simplified_text", ""),
                }

        raise HTTPException(status_code=404, detail=f"ไม่พบมาตรา {num} ในระบบ")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", summary="ตรวจสอบสถานะ backend")
async def health_check():
    """
    Liveness/Readiness probe

    Returns:
        {status: "OK"|"DB_DISCONNECTED", database_connected: bool}
    """
    db_connected = False
    status       = "OK"
    try:
        if db_manager.client:
            db_manager.client.get_collections()
            db_connected = True
    except Exception:
        status = "DB_DISCONNECTED"

    return {"status": status, "database_connected": db_connected}


# ══════════════════════════════════════════════
# Private Helpers
# ══════════════════════════════════════════════

def _build_error_response(query: str, reason: str) -> dict:
    """สร้าง response มาตรฐานสำหรับกรณีปฏิเสธคำถาม"""
    return {
        "query":             query,
        "detected_category": None,
        "found":             0,
        "results":           [],
        "legal_analysis": {
            "has_analysis":      True,
            "relevant_sections": "ปฏิเสธการตอบคำถาม",
            "general_meaning":   "-",
            "recommendation":    reason,
        },
    }


def _build_llm_fallback_response(query: str, detected_category) -> dict:
    """สร้าง response สำหรับกรณีไม่พบใน Vector DB — ใช้ LLM ตอบโดยตรง"""
    llm_analysis = generate_legal_analysis(query, []) or {
        "has_analysis":      True,
        "relevant_sections": "ไม่พบมาตราในฐานข้อมูล",
        "general_meaning":   "-",
        "recommendation":    "ระบบไม่สามารถให้คำแนะนำได้ในขณะนี้ กรุณาปรึกษาทนายความ",
    }
    return {
        "query":             query,
        "detected_category": detected_category,
        "found":             0,
        "results":           [],
        "legal_analysis":    llm_analysis,
    }


def _retrieve_nodes(expanded_query: str, detected_category: str | None, candidate_k: int) -> list:
    """
    ค้นหา candidate จาก Qdrant

    กลยุทธ์:
    - ถ้าระบุหมวดได้: ค้นเฉพาะหมวดก่อน → fallback ทั้ง collection ถ้าได้น้อยกว่า 5
    - ถ้าระบุหมวดไม่ได้: ค้นทั้ง collection ทันที
    """
    if detected_category:
        try:
            filters = MetadataFilters(filters=[
                MetadataFilter(key="category", value=detected_category)
            ])
            cat_retriever = db_manager.index.as_retriever(
                similarity_top_k=candidate_k, filters=filters
            )
            nodes = cat_retriever.retrieve(expanded_query)
            if len(nodes) >= 5:
                return nodes
        except Exception:
            pass

    # Fallback: ค้นทั้ง collection
    fallback = db_manager.index.as_retriever(similarity_top_k=candidate_k)
    return fallback.retrieve(expanded_query)


def _process_nodes(retrieved_nodes: list) -> list:
    """
    แปลง LlamaIndex nodes เป็น dict และ deduplicate ตามชื่อมาตรา

    Returns:
        list ของ {rank, score, section, category, original_text, simplified_text}
    """
    results      = []
    seen_sections = set()

    for node_with_score in retrieved_nodes:
        metadata     = node_with_score.node.metadata
        section_name = metadata.get("section", "ไม่ระบุ").strip()

        # ── Deduplication ──
        if section_name in seen_sections:
            continue
        seen_sections.add(section_name)

        # ── แยก original_text ออกจาก combined_text ──
        node_text = node_with_score.node.get_content()
        if "ตัวบทกฎหมาย:" in node_text:
            original_text = node_text.split("ตัวบทกฎหมาย:")[1].strip()
        else:
            original_text = node_text

        simplified_text = metadata.get("simplified_text", "").strip()
        if not has_simplified_explanation(original_text, simplified_text):
            simplified_text = ""

        results.append({
            "rank":          len(results) + 1,
            "score":         float(node_with_score.score),
            "section":       section_name,
            "category":      metadata.get("category", "ไม่ระบุ"),
            "original_text": original_text,
            "simplified_text": simplified_text,
        })

    return results


def _apply_reranker(query: str, results: list, settings: dict, top_k: int) -> list:
    """
    ใช้ Reranker จัดอันดับใหม่ (ถ้าเปิดใช้) หรือตัดให้เหลือ top_k

    Args:
        query   : คำถามต้นฉบับ (ไม่ใช่ expanded)
        results : list ผลลัพธ์ดิบ
        settings: dict การตั้งค่าจาก Admin
        top_k   : จำนวนผลลัพธ์สุดท้ายที่ต้องการ

    Returns:
        list ที่จัดอันดับแล้วและตัดให้เหลือ top_k
    """
    from reranker import reranker_manager
    if settings.get("use_reranker", False) and reranker_manager.enabled:
        return reranker_manager.rerank(query, results, top_k=top_k)
    return results[:top_k]


# ──────────────────────────────────────────────
# Dev Server Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
