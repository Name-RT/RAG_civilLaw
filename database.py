"""
database.py — จัดการการเชื่อมต่อและการสืบค้นข้อมูลจาก Qdrant Vector Database

ความรับผิดชอบ:
  - โหลด Embedding Model (multilingual-e5-small, dim=384)
  - เชื่อมต่อ Qdrant และ mount เป็น VectorStoreIndex
  - Ingest ข้อมูลกฎหมายลง Qdrant (กรณีที่ยังไม่มีข้อมูล หรือ force_reingest=True)
  - expose `db_manager` singleton เพื่อใช้ร่วมกันทั้งระบบ
"""

import os
import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.core.settings import Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever

from config import (
    QDRANT_URL, COLLECTION_NAME,
    EMBED_MODEL_NAME, EMBED_DIM,
    LAWS_FILE,
)


class DatabaseManager:
    """
    Singleton ที่จัดการ lifecycle ของ Vector DB ทั้งหมด

    Attributes:
        client      : QdrantClient — เชื่อมต่อ Qdrant
        vector_store: QdrantVectorStore — wrapper ของ LlamaIndex
        index       : VectorStoreIndex — ใช้ query และ insert
        retriever   : VectorIndexRetriever — ใช้ดึง top-k candidates
        embed_model : HuggingFaceEmbedding — โมเดลสร้าง vector (dim=384)
    """

    def __init__(self):
        self.client       = None
        self.vector_store = None
        self.index        = None
        self.retriever    = None
        # โหลด Embedding Model ตั้งแต่ init เพื่อให้พร้อมใช้ตลอด
        self.embed_model  = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
        Settings.embed_model = self.embed_model
        Settings.llm = None  # ปิด LLM ของ LlamaIndex — ระบบใช้ llm_service.py แทน

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect_and_initialize(self, force_reingest: bool = False) -> None:
        """
        เชื่อมต่อ Qdrant และเตรียม index ให้พร้อมรับ query

        Args:
            force_reingest: ถ้า True จะลบ collection เดิมและ ingest ใหม่ทั้งหมด
        """
        try:
            self.client       = QdrantClient(url=QDRANT_URL)
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=COLLECTION_NAME,
            )

            needs_ingest = self._check_needs_ingest(force_reingest)

            if needs_ingest:
                self._ingest_data()
            else:
                print(f"✅ ใช้ข้อมูลที่มีอยู่ใน Qdrant (collection: {COLLECTION_NAME})")

            storage_ctx = StorageContext.from_defaults(vector_store=self.vector_store)
            self.index  = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=storage_ctx,
            )
            self.retriever = VectorIndexRetriever(index=self.index, similarity_top_k=5)
            print("✅ DatabaseManager พร้อมรับคำขอแล้ว")

        except Exception as e:
            print(f"❌ DatabaseManager เชื่อมต่อไม่สำเร็จ: {e}")
            self.index     = None
            self.retriever = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_needs_ingest(self, force_reingest: bool) -> bool:
        """ตรวจสอบว่าต้อง ingest ข้อมูลหรือไม่"""
        if force_reingest:
            self._delete_collection()
            return True

        collections = self.client.get_collections().collections
        exists      = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            return True

        try:
            count = self.client.count(collection_name=COLLECTION_NAME).count
            return count == 0
        except Exception:
            return True

    def _delete_collection(self) -> None:
        """ลบ collection เดิมออกก่อน ingest ใหม่"""
        try:
            self.client.delete_collection(COLLECTION_NAME)
            print(f"🗑️  ลบ collection '{COLLECTION_NAME}' เดิมแล้ว")
        except Exception:
            pass

    def _create_collection(self) -> None:
        """สร้าง collection ใหม่ด้วย dimension ที่ตรงกับ embed model"""
        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )

    def _ingest_data(self) -> None:
        """โหลดข้อมูลกฎหมายจาก JSON และ embed ลง Qdrant"""
        print("📥 กำลัง Ingest ข้อมูลกฎหมายลง Qdrant...")

        if not os.path.exists(LAWS_FILE):
            print(f"❌ ไม่พบไฟล์ข้อมูล: {LAWS_FILE}")
            return

        with open(LAWS_FILE, "r", encoding="utf-8") as f:
            laws_data = json.load(f)

        self._create_collection()

        documents = []
        for item in laws_data:
            # ข้อความที่ใช้สร้าง vector ควรมีบริบทครบ เพื่อให้ค้นเจอง่าย
            combined_text = (
                f"มาตรา: {item['section']}\n"
                f"หมวดหมู่: {item['category']}\n\n"
                f"ตัวบทกฎหมาย: {item['original_text']}"
            )
            documents.append(
                Document(
                    text=combined_text,
                    metadata={
                        "section":         item["section"],
                        "category":        item["category"],
                        "law_type":        item.get("law_type", ""),
                        "law_name":        item.get("law_name", ""),
                        "simplified_text": item.get("simplified_text", ""),
                    },
                )
            )

        storage_ctx = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex.from_documents(documents, storage_context=storage_ctx)
        print(f"✅ Ingest เสร็จสิ้น — {len(documents)} มาตรา")


# Singleton — import ตัวนี้ไปใช้ในทุกโมดูล
db_manager = DatabaseManager()
