"""
admin.py — Admin Panel API สำหรับจัดการระบบ AuditGuard AI

Endpoints:
  GET  /api/admin/settings        — ดูการตั้งค่าปัจจุบัน
  POST /api/admin/settings        — แก้ไขการตั้งค่า (LLM, threshold, top_k, reranker)
  POST /api/admin/sync            — ดึงข้อมูลจาก HuggingFace และ ingest ใหม่
  GET  /api/admin/laws            — ดูรายการกฎหมายในฐานข้อมูล (รองรับ filter + search)
  PUT  /api/admin/laws            — แก้ไขข้อมูลกฎหมายและ reingest อัตโนมัติ

การตั้งค่าจะถูกบันทึกใน data/settings.json และโหลดทุกครั้งที่มีการเรียก API
"""

import os
import json
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from config import SETTINGS_FILE, LAWS_FILE, DEFAULT_SETTINGS

admin_router = APIRouter()


# ══════════════════════════════════════════════
# Settings Management
# ══════════════════════════════════════════════

def load_settings() -> Dict[str, Any]:
    """
    โหลดการตั้งค่าจากไฟล์ settings.json

    Returns:
        dict ของ settings หรือ DEFAULT_SETTINGS ถ้ายังไม่มีไฟล์
    """
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> None:
    """
    บันทึกการตั้งค่าลงไฟล์ settings.json

    Args:
        settings: dict ของ settings ที่ต้องการบันทึก
    """
    os.makedirs(os.path.dirname(SETTINGS_FILE), exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


class SettingsUpdate(BaseModel):
    """Schema สำหรับ request body ของ POST /api/admin/settings"""
    chunk_size:           int
    chunk_overlap:        int
    top_k:                int
    similarity_threshold: float
    active_llm:           str    # "gemini" | "openai" | "deepseek"
    use_reranker:         bool


@admin_router.get("/api/admin/settings", summary="ดูการตั้งค่าปัจจุบัน")
async def get_settings():
    return load_settings()


@admin_router.post("/api/admin/settings", summary="แก้ไขการตั้งค่า")
async def update_settings(settings: SettingsUpdate):
    current = load_settings()
    current.update(settings.dict())
    save_settings(current)

    # อัปเดต Reranker ทันทีโดยไม่ต้อง restart
    from reranker import reranker_manager
    if current["use_reranker"] and not reranker_manager.enabled:
        reranker_manager.load_model()
    elif not current["use_reranker"]:
        reranker_manager.enabled = False

    return {"status": "success", "message": "บันทึกการตั้งค่าเรียบร้อยแล้ว"}


# ══════════════════════════════════════════════
# Data Sync
# ══════════════════════════════════════════════

@admin_router.post("/api/admin/sync", summary="ดึงข้อมูลจาก HuggingFace และ ingest ใหม่")
async def sync_data():
    """ดึงข้อมูลกฎหมายล่าสุดจาก HuggingFace แล้ว reingest ลง Qdrant"""
    try:
        from ingest_huggingface import fetch_and_process_laws
        from database import db_manager

        fetch_and_process_laws()
        db_manager.connect_and_initialize(force_reingest=True)
        return {"status": "success", "message": "ซิงค์และอัปเดตฐานข้อมูลสำเร็จ"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════
# Law Data Management
# ══════════════════════════════════════════════

class LawUpdate(BaseModel):
    """Schema สำหรับ request body ของ PUT /api/admin/laws"""
    law_type:       str
    law_name:       str
    section:        str
    category:       str
    original_text:  str
    simplified_text: str


@admin_router.get("/api/admin/laws", summary="ดูรายการกฎหมาย")
async def get_laws(search: str = "", law_type: str = ""):
    """
    ดึงรายชื่อกฎหมายทั้งหมด รองรับการกรองตาม law_type และ search keyword

    Query Params:
        search  : ค้นหาจากชื่อมาตรา เนื้อหา หรือชื่อกฎหมาย
        law_type: กรองตามประเภทกฎหมาย เช่น "criminal" หรือ "civil"

    Returns:
        list ของกฎหมาย (จำกัดที่ 100 รายการ)
    """
    if not os.path.exists(LAWS_FILE):
        return []

    with open(LAWS_FILE, "r", encoding="utf-8") as f:
        laws = json.load(f)

    if law_type:
        laws = [law for law in laws if law.get("law_type") == law_type]

    if search:
        search_lower = search.lower()
        laws = [
            law for law in laws
            if search_lower in law.get("section",       "").lower()
            or search_lower in law.get("original_text", "").lower()
            or search_lower in law.get("law_name",      "").lower()
        ]

    return laws[:100]  # จำกัด 100 รายการ ป้องกัน browser ค้าง


@admin_router.put("/api/admin/laws", summary="แก้ไขข้อมูลกฎหมาย")
async def update_law(law_update: LawUpdate):
    """
    แก้ไขข้อมูลมาตรากฎหมาย และสั่ง reingest ลง Qdrant โดยอัตโนมัติ

    ใช้ (section + law_name) เป็น composite key ในการค้นหา
    """
    if not os.path.exists(LAWS_FILE):
        raise HTTPException(status_code=404, detail="ไม่พบไฟล์ข้อมูลกฎหมาย")

    with open(LAWS_FILE, "r", encoding="utf-8") as f:
        laws = json.load(f)

    updated = False
    for i, law in enumerate(laws):
        if law.get("section") == law_update.section and law.get("law_name") == law_update.law_name:
            laws[i].update({
                "law_type":       law_update.law_type,
                "category":       law_update.category,
                "original_text":  law_update.original_text,
                "simplified_text": law_update.simplified_text,
            })
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail="ไม่พบมาตรานี้ในระบบ")

    with open(LAWS_FILE, "w", encoding="utf-8") as f:
        json.dump(laws, f, ensure_ascii=False, indent=4)

    # Reingest อัตโนมัติหลังแก้ไข
    from database import db_manager
    try:
        db_manager.connect_and_initialize(force_reingest=True)
    except Exception as e:
        print(f"⚠️ Reingest after update failed: {e}")

    return {"status": "success", "message": f"อัปเดตมาตรา {law_update.section} สำเร็จแล้ว"}
