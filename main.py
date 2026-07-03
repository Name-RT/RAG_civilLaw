import os
import json
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# นำเข้าจากโมดูลที่แยกออกมา
from utils import classify_query, has_simplified_explanation, convert_arabic_to_thai
from database import db_manager
from llm_service import generate_legal_analysis
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter

# บังคับการส่งออกแบบ UTF-8 ป้องกันคีย์เวิร์ดภาษาไทยแครชบน Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

# โหลดตัวแปรสภาพแวดล้อม
load_dotenv()

app = FastAPI(
    title="LegalShield AI API",
    description="ระบบค้นหาและวิเคราะห์ข้อกฎหมายแพ่ง-อาญาอัตโนมัติ",
    version="1.0.0"
)

# ตั้งค่า CORS เพื่อให้หน้าเว็บ (HTML) สามารถยิงเรียกใช้ API ข้ามโดเมนได้โดยไม่ติดบล็อก
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # อนุญาตให้เรียกจากหน้าเว็บใดก็ได้ (เหมาะสำหรับการพัฒนาในเครื่องโลคอล)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# เริ่มต้นการเชื่อมต่อ Database เมื่อสตาร์ทแอป
@app.on_event("startup")
async def startup_event():
    db_manager.connect_and_initialize()

# โมเดลข้อมูลสำหรับการรับคำขอค้นหา (Request body)
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/ask")
async def ask_law(request: QueryRequest):
    if not db_manager.retriever or not db_manager.index:
        raise HTTPException(
            status_code=503,
            detail="ฐานข้อมูลเวกเตอร์ยังไม่ได้เริ่มทำงาน (กรุณาเปิดบริการ Qdrant Container)"
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="กรุณากรอกคำถามที่ต้องการค้นหา")

    try:
        top_k = request.top_k or 5
        # ดึงจำนวน candidate มากขึ้นเพื่อเอามาคัดเลือกข้อที่มีคำอธิบายและตัดข้อมูลซ้ำออก
        candidate_k = max(35, top_k * 6)

        # 1. วิเคราะห์หมวดหมู่ของคำถาม
        detected_category = classify_query(request.query)

        # 2. สร้าง retriever ตามหมวดหมู่ที่ตรวจพบ
        if detected_category:
            try:
                filters = MetadataFilters(filters=[
                    MetadataFilter(key="category", value=detected_category)
                ])
                cat_retriever = db_manager.index.as_retriever(similarity_top_k=candidate_k, filters=filters)
                retrieved_nodes = cat_retriever.retrieve(request.query)
                # Fallback: ถ้าดึงมาได้น้อยกว่า 5 ให้ค้นทั้ง collection
                if len(retrieved_nodes) < 5:
                    fallback_retriever = db_manager.index.as_retriever(similarity_top_k=candidate_k)
                    retrieved_nodes = fallback_retriever.retrieve(request.query)
                    detected_category = None  # แจ้งว่าใช้ fallback
            except Exception:
                fallback_retriever = db_manager.index.as_retriever(similarity_top_k=candidate_k)
                retrieved_nodes = fallback_retriever.retrieve(request.query)
                detected_category = None
        else:
            fallback_retriever = db_manager.index.as_retriever(similarity_top_k=candidate_k)
            retrieved_nodes = fallback_retriever.retrieve(request.query)
            
        results = []
        seen_sections = set()
        
        for node_with_score in retrieved_nodes:
            metadata = node_with_score.node.metadata
            score = node_with_score.score
            
            # ดึงข้อมูลมาตราและหมวดหมู่
            section_name = metadata.get("section", "ไม่ระบุ").strip()
            category = metadata.get("category", "ไม่ระบุ")
            simplified_text = metadata.get("simplified_text", "").strip()
            
            # คลีนและดึงข้อความกฎหมายดั้งเดิมจากเนื้อหา
            node_text = node_with_score.node.get_content()
            original_text = ""
            for line in node_text.split('\n'):
                if line.startswith("ตัวบทกฎหมาย:"):
                    original_text = line.replace("ตัวบทกฎหมาย:", "").strip()
                    break
            if not original_text:
                original_text = node_text
                
            # 1. ตรวจสอบมาตราซ้ำ (De-duplication)
            if section_name in seen_sections:
                continue
                
            # 2. ตรวจสอบว่ามีคำอธิบายย่อยหรือไม่
            if not has_simplified_explanation(original_text, simplified_text):
                simplified_text = ""
                
            seen_sections.add(section_name)
            
            results.append({
                "rank": len(results) + 1,
                "score": float(score),
                "section": section_name,
                "category": category,
                "original_text": original_text,
                "simplified_text": simplified_text
            })
            
            # บรรลุเป้าหมายการดึงข้อมูลครบจำนวน
            if len(results) >= top_k:
                break
                
        # 3. ส่งข้อมูลให้ LLM ร่วมประมวลผลและให้คำแนะนำ
        legal_analysis = generate_legal_analysis(request.query, results)
        if not legal_analysis:
            legal_analysis = {
                "has_analysis": False,
                "relevant_sections": "",
                "general_meaning": "",
                "recommendation": ""
            }
            
        return {
            "query": request.query,
            "detected_category": detected_category,
            "found": len(results),
            "results": results,
            "legal_analysis": legal_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการประมวลผล: {str(e)}")

@app.get("/section/{num}")
async def get_section(num: str):
    """ดึงเนื้อหามาตรากฎหมายแบบระบุเลขมาตราโดยตรง"""
    json_path = "data/processed_laws.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=500, detail="ไม่พบไฟล์ข้อมูลกฎหมาย")
        
    try:
        thai_num = convert_arabic_to_thai(num)
        
        target_1 = f"มาตรา {num}"
        target_2 = f"มาตรา {thai_num}"
        
        with open(json_path, "r", encoding="utf-8") as f:
            laws_data = json.load(f)
            
        for item in laws_data:
            sec_clean = item["section"].strip()
            if sec_clean == target_1 or sec_clean == target_2:
                return {
                    "section": item["section"],
                    "category": item["category"],
                    "original_text": item["original_text"],
                    "simplified_text": item.get("simplified_text", "")
                }
                
        raise HTTPException(status_code=404, detail=f"ไม่พบมาตรา {num} ในระบบ")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """เช็คสถานะการเชื่อมต่อบริการหลังบ้าน"""
    status = "OK"
    db_connected = False
    try:
        if db_manager.client:
            db_manager.client.get_collections()
            db_connected = True
    except Exception:
        status = "DB_DISCONNECTED"
        
    return {
        "status": status,
        "database_connected": db_connected
    }

if __name__ == "__main__":
    import uvicorn
    # สตาร์ทรันเซิร์ฟเวอร์บนพอร์ต 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
