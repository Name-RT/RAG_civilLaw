"""
config.py — ค่าคงที่และการตั้งค่าส่วนกลางของระบบ AuditGuard AI
"""

# ──────────────────────────────────────────────
# Qdrant Vector DB
# ──────────────────────────────────────────────
QDRANT_URL         = "http://localhost:6333"
COLLECTION_NAME    = "thai_laws"

# ──────────────────────────────────────────────
# Embedding Model (ต้องตรงกับ dimension ใน Qdrant)
# Dimension: 384  →  intfloat/multilingual-e5-small
# ──────────────────────────────────────────────
EMBED_MODEL_NAME   = "intfloat/multilingual-e5-small"
EMBED_DIM          = 384

# ──────────────────────────────────────────────
# Reranker Model (เปิดใช้ได้จาก Admin Panel)
# ──────────────────────────────────────────────
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANKER_MAX_LEN    = 512

# ──────────────────────────────────────────────
# LLM Models (per provider)
# ──────────────────────────────────────────────
LLM_GEMINI_MODEL    = "gemini-3.5-flash"
LLM_OPENAI_MODEL    = "gpt-5.6-luna"
LLM_DEEPSEEK_MODEL  = "deepseek-v4-flash"
LLM_DEEPSEEK_URL    = "https://api.deepseek.com"

# ──────────────────────────────────────────────
# Default RAG Settings (ใช้เมื่อ data/settings.json ยังไม่มี)
# ──────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "chunk_size":           512,
    "chunk_overlap":        50,
    "top_k":                5,
    "similarity_threshold": 0.4,   # เหมาะกับ multilingual-e5-small
    "active_llm":           "deepseek",
    "use_reranker":         False,
}

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
SETTINGS_FILE  = "data/settings.json"
LAWS_FILE      = "data/processed_laws.json"
STATIC_DIR     = "static"
INDEX_HTML     = "index.html"

# ──────────────────────────────────────────────
# Rate Limiting
# ──────────────────────────────────────────────
RATE_LIMIT_ASK     = "10/minute"
RATE_LIMIT_SECTION = "30/minute"
