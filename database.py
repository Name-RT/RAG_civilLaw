import os
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever

class DatabaseManager:
    def __init__(self, qdrant_url: str = "http://localhost:6333", collection_name: str = "thai_laws"):
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name
        self.client = None
        self.vector_store = None
        self.index = None
        self.retriever = None

        # ตั้งค่าโมเดลเวกเตอร์ Embedding ตั้งแต่สตาร์ท
        self.embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-small")
        Settings.embed_model = self.embed_model
        Settings.llm = None  # เน้นระบบสืบค้นตรงตัวบทและการอธิบายเปรียบเทียบก่อน

    def connect_and_initialize(self):
        try:
            self.client = QdrantClient(url=self.qdrant_url)
            self.vector_store = QdrantVectorStore(client=self.client, collection_name=self.collection_name)
            
            # ตรวจสอบสถานะชุดข้อมูลภายใน Qdrant
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            is_empty = True
            if exists:
                try:
                    count = self.client.count(collection_name=self.collection_name).count
                    if count > 0:
                        is_empty = False
                except Exception:
                    pass
                    
            if is_empty:
                print("💾 ไม่พบข้อมูลกฎหมายในระบบ ทำการสร้างชุดข้อมูลตัวอย่างเริ่มต้น (Demo Laws Ingestion)...")
                os.makedirs("data", exist_ok=True)
                json_path = "data/processed_laws.json"
                template_path = "data/processed_laws_template.json"
                
                laws_data = []
                if not os.path.exists(json_path):
                    # ถ้าไม่มีไฟล์ data ให้โหลดจาก template ไปสร้าง
                    if os.path.exists(template_path):
                        with open(template_path, "r", encoding="utf-8") as f:
                            laws_data = json.load(f)
                        with open(json_path, "w", encoding="utf-8") as f:
                            json.dump(laws_data, f, ensure_ascii=False, indent=4)
                    else:
                        print(f"[ERROR] ไม่พบไฟล์ {template_path} สำหรับจำลองข้อมูล")
                        laws_data = []
                else:
                    with open(json_path, "r", encoding="utf-8") as f:
                        laws_data = json.load(f)
                        
                if laws_data:
                    self._ingest_data(laws_data, exists)
                else:
                    print("[WARNING] ไม่มีข้อมูลสำหรับ Ingest")
            else:
                self.index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
                
            # fallback retriever (ไม่มี filter) ใช้เมื่อจำแนกหมวดหมู่ไม่ได้
            if self.index:
                self.retriever = VectorIndexRetriever(index=self.index, similarity_top_k=5)
            print("[OK] Connected to Qdrant Vector DB successfully.")
            
        except Exception as e:
            print(f"[ERROR] Connection to Qdrant failed: {e}")
            self.retriever = None
            self.index = None

    def _ingest_data(self, laws_data, collection_exists):
        documents = []
        for law in laws_data:
            content = f"ตัวบทกฎหมาย: {law['original_text']}"
            doc = Document(
                text=content,
                metadata={
                    "section": law["section"],
                    "category": law["category"],
                    "simplified_text": law.get("simplified_text", "")
                }
            )
            documents.append(doc)
            
        # สร้าง Collection ใหม่
        if collection_exists:
            self.client.delete_collection(collection_name=self.collection_name)
            
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print("[OK] Ingested laws into Qdrant successfully.")

# สร้าง Singleton instance ของระบบจัดการฐานข้อมูล
db_manager = DatabaseManager()
