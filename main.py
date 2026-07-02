import os
import json
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter
from dotenv import load_dotenv

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

# โมเดลข้อมูลสำหรับการรับคำขอค้นหา (Request body)
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


# =====================================================
# ระบบวิเคราะห์หมวดหมู่จากคำถามผู้ใช้ (Query Classifier)
# =====================================================
CATEGORY_KEYWORDS = [
    ("มรดก",                          ["มรดก", "พินัยกรรม", "ทายาท", "รับมรดก", "ผู้จัดการมรดก", "ยกมรดก", "สืบมรดก", "ตัดทายาท"]),
    ("ครอบครัว",                       ["สมรส", "แต่งงาน", "หย่า", "หมั้น", "คู่สมรส", "สามี", "ภรรยา", "บุตร", "ลูก",
                                         "ค่าเลี้ยงดู", "อำนาจปกครอง", "ฟ้องหย่า", "จดทะเบียนสมรส", "สินสมรส",
                                         "บิดา", "มารดา", "พ่อ", "แม่", "รับบุตรบุญธรรม"]),
    ("ละเมิด",                         ["ละเมิด", "ชนรถ", "ทำให้เสียหาย", "ประมาท", "ค่าเสียหาย", "สินไหมทดแทน",
                                         "อุบัติเหตุ", "บาดเจ็บ", "ทำร้าย", "จงใจ", "ขับรถชน"]),
    ("จำนองและจำนำ",                   ["จำนอง", "จำนำ", "บังคับจำนอง", "ไถ่ถอน", "ยึดทรัพย์จำนอง"]),
    ("กู้ยืมและค้ำประกัน",              ["กู้", "ยืมเงิน", "เงินกู้", "ค้ำประกัน", "ผู้ค้ำ", "ดอกเบี้ย",
                                         "หนี้เงินกู้", "กู้เงิน", "คืนเงิน", "สัญญากู้"]),
    ("เช่าทรัพย์และเช่าซื้อ",           ["เช่า", "เช่าบ้าน", "เช่าซื้อ", "ผู้เช่า", "เจ้าของบ้าน",
                                         "ค่าเช่า", "ไล่ออก", "ไม่ยอมออก", "บอกเลิกเช่า"]),
    ("ซื้อขาย",                        ["ซื้อขาย", "ซื้อของ", "ขายของ", "ชำรุด", "บกพร่อง",
                                         "สินค้า", "ราคา", "ส่งมอบ", "ผิดสัญญาซื้อขาย", "ขายฝาก"]),
    ("จ้างงาน",                        ["จ้างงาน", "จ้างแรงงาน", "ลูกจ้าง", "นายจ้าง", "ค่าจ้าง",
                                         "ค่าแรง", "เลิกจ้าง", "ออกจากงาน", "จ้างทำของ", "ไล่ออก"]),
    ("ฝากทรัพย์และยืมใช้",             ["ฝากของ", "ฝากทรัพย์", "ยืมของ", "ยืมใช้", "คืนของ", "ของหาย"]),
    ("ตัวแทนและนายหน้า",               ["ตัวแทน", "นายหน้า", "มอบอำนาจ", "หนังสือมอบอำนาจ", "ค่านายหน้า"]),
    ("นิติบุคคล",                      ["บริษัท", "ห้างหุ้นส่วน", "จดทะเบียน", "หุ้น", "ผู้ถือหุ้น",
                                         "กรรมการ", "สมาคม", "มูลนิธิ", "หุ้นส่วน"]),
    ("ทรัพย์และทรัพย์สิน",             ["กรรมสิทธิ์", "ครอบครอง", "ที่ดิน", "โฉนด", "อสังหาริมทรัพย์",
                                         "บุกรุก", "ทางผ่าน", "ภาระจำยอม", "ทางจำเป็น"]),
    ("บุคคลและความสามารถทางกฎหมาย",    ["ผู้เยาว์", "เด็ก", "อายุต่ำกว่า", "บรรลุนิติภาวะ", "คนบ้า",
                                         "คนไร้ความสามารถ", "ผู้อนุบาล", "ผู้พิทักษ์"]),
    ("หนี้และการชำระหนี้",             ["หนี้", "เจ้าหนี้", "ลูกหนี้", "ชำระหนี้", "ผิดนัด",
                                         "ไม่ยอมจ่าย", "ทวงหนี้", "บังคับชำระ"]),
    ("นิติกรรมและสัญญาทั่วไป",         ["สัญญา", "ข้อตกลง", "ผิดสัญญา", "บอกเลิกสัญญา", "โมฆะ",
                                         "ยกเลิก", "คู่สัญญา", "ทำสัญญา", "นิติกรรม"]),
]

def classify_query(query: str) -> Optional[str]:
    """
    วิเคราะห์คำถามผู้ใช้และคืนค่าหมวดหมู่ที่น่าจะตรงที่สุด
    คืนค่า None ถ้าระบุหมวดหมู่ไม่ได้ (จะ fallback ไปค้นทั้ง collection)
    """
    scores = {}
    for cat_name, keywords in CATEGORY_KEYWORDS:
        count = sum(1 for kw in keywords if kw in query)
        if count > 0:
            scores[cat_name] = count
    if not scores:
        return None
    return max(scores, key=lambda k: scores[k])

def has_simplified_explanation(original: str, simplified: str) -> bool:
    orig_clean = original.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    simp_clean = simplified.strip().replace(" ", "").replace("\n", "").replace("\r", "")
    
    if not simp_clean:
        return False
    if simp_clean == orig_clean:
        return False
    if len(simp_clean) < 15:
        return False
    return True

# ข้อมูลสาธิตกรณีไม่มีฐานข้อมูลจริงในเครื่อง (ใช้เพื่อเริ่มระบบได้ทันทีเมื่อโคลนโปรเจกต์)
DEMO_LAWS = [
    {
        "section": "มาตรา ๑๕",
        "category": "บุคคลและความสามารถทางกฎหมาย",
        "original_text": "สภาพบุคคลย่อมเริ่มแต่เมื่อคลอดแล้วอยู่รอดเป็นทารก และสิ้นสุดลงเมื่อตาย",
        "simplified_text": "สภาพความเป็นบุคคลเริ่มต้นตั้งแต่เวลาที่เราเกิดมาและรอดชีวิต และจะสิ้นสุดลงเมื่อเราเสียชีวิต"
    },
    {
        "section": "มาตรา ๒๑",
        "category": "บุคคลและความสามารถทางกฎหมาย",
        "original_text": "ผู้เยาว์จะทำนิติกรรมใดๆ ต้องได้รับความยินยอมของผู้แทนโดยชอบธรรมก่อน การใดๆ ที่ผู้เยาว์ได้ทำลงปราศจากความยินยอมเช่นว่านั้นท่านว่าเป็นโมฆียะ เว้นแต่จะบัญญัติไว้เป็นอย่างอื่น",
        "simplified_text": "เด็กที่ยังไม่บรรลุนิติภาวะ (ผู้เยาว์) หากจะเซ็นสัญญาหรือทำข้อตกลงใดๆ จะต้องได้รับความยินยอมจากพ่อแม่หรือผู้ปกครองก่อน มิฉะนั้นสัญญาดังกล่าวอาจถูกยกเลิก (เป็นโมฆียะ) ในภายหลังได้"
    },
    {
        "section": "มาตรา ๖๕๓",
        "category": "กู้ยืมและค้ำประกัน",
        "original_text": "การกู้ยืมเงินกว่าสองพันบาทขึ้นไปนั้น หากมิได้มีหลักฐานเป็นหนังสืออย่างใดอย่างหนึ่งลงลายมือชื่อผู้ยืมเป็นสำคัญ จะฟ้องร้องบังคับคดีหาได้ไม่",
        "simplified_text": "หากกู้ยืมเงินกันเกินกว่า 2,000 บาทขึ้นไป จะต้องมีหลักฐานการกู้เงินเป็นหนังสือหรือลายลักษณ์อักษรที่ลงลายมือชื่อผู้ยืม จึงจะสามารถใช้ฟ้องร้องดำเนินคดีตามกฎหมายได้"
    },
    {
        "section": "มาตรา ๑๓๐๔",
        "category": "ทรัพย์และทรัพย์สิน",
        "original_text": "สาธารณสมบัติของแผ่นดินนั้น รวมทรัพย์สินทุกชนิดของแผ่นดินซึ่งมีไว้เพื่อสาธารณประโยชน์หรือสงวนไว้เพื่อประโยชน์ร่วมกัน",
        "simplified_text": "ทรัพย์สินส่วนรวมของแผ่นดินที่มีไว้เพื่อประโยชน์ของส่วนรวมหรือทุกคนในประเทศร่วมกัน"
    },
    {
        "section": "มาตรา ๔๒๐",
        "category": "ละเมิด",
        "original_text": "ผู้ใดจงใจหรือประมาทเลินเล่อ ทำต่อบุคคลอื่นโดยผิดกฎหมายให้เขาเสียหายถึงแก่ชีวิตก็ดี แก่ร่างกายก็ดี อนามัยก็ดี เสรีภาพก็ดี ทรัพย์สินหรือสิทธิอย่างหนึ่งอย่างใดก็ดี ท่านว่าผู้นั้นทำละเมิดจำต้องชดใช้ค่าสินไหมทดแทนเพื่อการนั้น",
        "simplified_text": "ใครก็ตามที่ตั้งใจหรือประมาทเลินเล่อแล้วทำให้ผู้อื่นได้รับความเสียหายต่อชีวิต ร่างกาย สุขภาพ เสรีภาพ ทรัพย์สิน หรือสิทธิ์ต่าง ๆ ถือเป็นการทำละเมิด และต้องจ่ายค่าชดเชยค่าเสียหายให้กับผู้เสียหาย"
    }
]

# ตั้งค่าโมเดลเวกเตอร์ Embedding และเชื่อมต่อ Qdrant ตั้งแต่สตาร์ทแอป
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")
Settings.embed_model = embed_model
Settings.llm = None  # เน้นระบบสืบค้นตรงตัวบทและการอธิบายเปรียบเทียบก่อน

try:
    client = QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(client=client, collection_name="thai_laws")
    
    # ตรวจสอบสถานะชุดข้อมูลภายใน Qdrant
    collections = client.get_collections().collections
    exists = any(c.name == "thai_laws" for c in collections)
    is_empty = True
    if exists:
        try:
            count = client.count(collection_name="thai_laws").count
            if count > 0:
                is_empty = False
        except Exception:
            pass
            
    if is_empty:
        print("💾 ไม่พบข้อมูลกฎหมายในระบบ ทำการสร้างชุดข้อมูลตัวอย่างเริ่มต้น (Demo Laws Ingestion)...")
        os.makedirs("data", exist_ok=True)
        json_path = "data/processed_laws.json"
        if not os.path.exists(json_path):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(DEMO_LAWS, f, ensure_ascii=False, indent=4)
                
        # เตรียมเอกสารนำเข้า
        from llama_index.core import Document
        documents = []
        for law in DEMO_LAWS:
            content = f"ตัวบทกฎหมาย: {law['original_text']}"
            doc = Document(
                text=content,
                metadata={
                    "section": law["section"],
                    "category": law["category"],
                    "simplified_text": law["simplified_text"]
                }
            )
            documents.append(doc)
            
        # สร้าง Collection ใหม่
        if exists:
            client.delete_collection(collection_name="thai_laws")
            
        from qdrant_client.models import Distance, VectorParams
        client.create_collection(
            collection_name="thai_laws",
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        
        from llama_index.core import StorageContext
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print("[OK] Ingested demo laws into Qdrant successfully.")
    else:
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
    # fallback retriever (ไม่มี filter) ใช้เมื่อจำแนกหมวดหมู่ไม่ได้
    retriever = VectorIndexRetriever(index=index, similarity_top_k=5)
    print("[OK] Connected to Qdrant Vector DB successfully.")
except Exception as e:
    print(f"[ERROR] Connection to Qdrant failed: {e}")
    retriever = None
    index = None

def generate_legal_analysis(query: str, retrieved_results: list) -> Optional[dict]:
    """
    ส่งข้อความคำถามและมาตรากฎหมายที่ดึงได้ไปประมวลผลต่อด้วย LLM (Gemini / ChatGPT / DeepSeek)
    เพื่อคัดกรองมาตราที่ไม่เกี่ยวข้องออก สรุปตัวบททั่วไป และให้คำแนะนำทางกฎหมายแบบส่วนตัว
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    
    # ประกอบบริบทเนื้อหาของมาตรากฎหมายที่ดึงได้
    context_parts = []
    for item in retrieved_results:
        context_parts.append(
            f"มาตรา: {item['section']}\n"
            f"หมวดหมู่: {item['category']}\n"
            f"ตัวบทกฎหมายดิบ: {item['original_text']}\n"
            f"คำอธิบายทั่วไปเดิม: {item['simplified_text']}\n"
        )
    context_str = "\n---\n".join(context_parts)
    
    prompt = f"""คุณคือผู้เชี่ยวชาญด้านกฎหมายไทย หน้าที่ของคุณคือวิเคราะห์คำถามของผู้ใช้ร่วมกับบทบัญญัติกฎหมายที่ดึงมาจากฐานข้อมูล เพื่อให้คำแนะนำที่ถูกต้อง น่าเชื่อถือ และเข้าใจง่ายที่สุด

คำถามของผู้ใช้: "{query}"

มาตรากฎหมายที่ดึงขึ้นมาได้จากฐานข้อมูล:
{context_str}

คำแนะนำการตอบกลับ:
1. ตรวจสอบว่ามาตราที่ดึงขึ้นมามีความเกี่ยวข้องกับคำถามและข้อเท็จจริงของผู้ใช้จริงๆ หรือไม่ หากมีมาตราใดที่ไม่เกี่ยวข้องเลย ให้ตัดออกและไม่ต้องนำมากล่าวถึงในข้อความ
2. ระบุชื่อมาตราที่เกี่ยวข้องทั้งหมดที่ผ่านการคัดกรองแล้วในส่วน 'relevant_sections'
3. สรุปความหมายโดยทั่วไปของมาตราที่เกี่ยวข้องเหล่านั้น ให้เข้าใจง่ายในภาษาพูดของคนทั่วไปในส่วน 'general_meaning'
4. ให้คำแนะนำทางกฎหมาย (Legal Recommendation) สำหรับกรณีของผู้ใช้ในส่วน 'recommendation' โดยระบุตัวมาตราที่เกี่ยวข้อง และแนะนำช่องทางปฏิบัติจริง (เช่น หลักฐานหนังสือสัญญา, แชทไลน์, หรือสลิปโอนเงินตามกฎหมาย)

กรุณาตอบกลับในรูปแบบ JSON เสมอ โดยใช้โครงสร้างดังนี้:
{{
  "relevant_sections": "ระบุมาตราที่เกี่ยวข้องทั้งหมด เช่น มาตรา ๖๕๓, มาตรา ... (หากไม่มีเลยให้ระบุ 'ไม่มีมาตราที่เกี่ยวข้องโดยตรง')",
  "general_meaning": "สรุปความหมายโดยทั่วไปของมาตราเหล่านี้ในภาษาพูด...",
  "recommendation": "คำแนะนำทางกฎหมายและการปฏิบัติสำหรับกรณีของผู้ใช้..."
}}"""

    # --- 1. ลองใช้ Gemini API ---
    if gemini_key:
        try:
            print("🤖 Calling Gemini 1.5 Flash for analysis...")
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content(prompt)
            analysis = json.loads(response.text)
            return {
                "has_analysis": True,
                "relevant_sections": analysis.get("relevant_sections", "").strip(),
                "general_meaning": analysis.get("general_meaning", "").strip(),
                "recommendation": analysis.get("recommendation", "").strip()
            }
        except Exception as e:
            print(f"⚠️ Gemini API failed: {e}. Trying fallback...")
            
    # --- 2. ลองใช้ OpenAI / DeepSeek (ระบบสำรอง fallback) ---
    api_key = openai_key
    base_url = None
    model_name = "gpt-4o-mini"
    
    if not api_key and deepseek_key:
        api_key = deepseek_key
        base_url = "https://api.deepseek.com"
        model_name = "deepseek-chat"
        
    if not api_key:
        return None
        
    try:
        print(f"🤖 Calling OpenAI/DeepSeek ({model_name}) fallback...")
        from openai import OpenAI
        client_opts = {"api_key": api_key}
        if base_url:
            client_opts["base_url"] = base_url
            
        llm_client = OpenAI(**client_opts)
        response = llm_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "คุณคือผู้ช่วยกฎหมายมืออาชีพ ตอบในรูปแบบ JSON เสมอ"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        analysis = json.loads(response.choices[0].message.content)
        return {
            "has_analysis": True,
            "relevant_sections": analysis.get("relevant_sections", "").strip(),
            "general_meaning": analysis.get("general_meaning", "").strip(),
            "recommendation": analysis.get("recommendation", "").strip()
        }
    except Exception as e:
        print(f"⚠️ Error running LLM Fallback: {e}")
        return None

@app.post("/ask")
async def ask_law(request: QueryRequest):
    if not retriever or not index:
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
                cat_retriever = index.as_retriever(similarity_top_k=candidate_k, filters=filters)
                retrieved_nodes = cat_retriever.retrieve(request.query)
                # Fallback: ถ้าดึงมาได้น้อยกว่า 5 ให้ค้นทั้ง collection
                if len(retrieved_nodes) < 5:
                    fallback_retriever = index.as_retriever(similarity_top_k=candidate_k)
                    retrieved_nodes = fallback_retriever.retrieve(request.query)
                    detected_category = None  # แจ้งว่าใช้ fallback
            except Exception:
                fallback_retriever = index.as_retriever(similarity_top_k=candidate_k)
                retrieved_nodes = fallback_retriever.retrieve(request.query)
                detected_category = None
        else:
            fallback_retriever = index.as_retriever(similarity_top_k=candidate_k)
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
                
            # 2. ตรวจสอบว่ามีคำอธิบายย่อยหรือไม่ (ถ้าไม่มีให้ตั้งค่าเป็นค่าว่างเพื่อให้หน้าเว็บไม่แสดงกล่องคำอธิบาย)
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
                
        # 3. ส่งข้อมูลให้ LLM (Gemini / ChatGPT / DeepSeek) ร่วมประมวลผลและให้คำแนะนำ
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
    """ดึงเนื้อหามาตรากฎหมายแบบระบุเลขมาตราโดยตรง (ใช้ในส่วนคลิกอ้างอิงมาตราอื่น)"""
    json_path = "data/processed_laws.json"
    if not os.path.exists(json_path):
        raise HTTPException(status_code=500, detail="ไม่พบไฟล์ข้อมูลกฎหมาย")
        
    try:
        # แปลงตัวเลขอารบิกเป็นตัวเลขไทยเพื่อจับคู่การค้นหาได้ทั้งคู่
        arabic_to_thai = {"0":"๐", "1":"๑", "2":"๒", "3":"๓", "4":"๔", "5":"๕", "6":"๖", "7":"๗", "8":"๘", "9":"๙"}
        thai_num = "".join([arabic_to_thai.get(c, c) for c in num])
        
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
        if client:
            client.get_collections()
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
