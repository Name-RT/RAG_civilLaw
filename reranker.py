import os
from sentence_transformers import CrossEncoder

class RerankerManager:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self.model = None
        self.enabled = False

    def load_model(self):
        try:
            print(f"🔄 กำลังโหลด Reranker Model ({self.model_name})...")
            # โหลด CrossEncoder สำหรับจัดอันดับ
            self.model = CrossEncoder(self.model_name, max_length=512)
            self.enabled = True
            print("✅ โหลด Reranker สำเร็จ")
        except Exception as e:
            print(f"⚠️ ไม่สามารถโหลด Reranker ได้: {e}")
            self.enabled = False

    def rerank(self, query: str, results: list, top_k: int = 5) -> list:
        if not self.enabled or not self.model or not results:
            return results[:top_k]
            
        # สร้างคู่คำถาม-คำตอบเพื่อส่งให้ CrossEncoder ประเมิน
        pairs = []
        for res in results:
            text = f"มาตรา: {res['section']} {res['original_text']}"
            pairs.append([query, text])
            
        try:
            # คำนวณคะแนนความเกี่ยวข้อง (ยิ่งมากยิ่งเกี่ยว)
            scores = self.model.predict(pairs)
            
            # อัปเดตคะแนนใหม่ลงไป
            for i, score in enumerate(scores):
                results[i]['rerank_score'] = float(score)
                
            # เรียงลำดับจากคะแนนสูงไปต่ำ
            results.sort(key=lambda x: x.get('rerank_score', -999), reverse=True)
            
            # อัปเดต rank 
            for i, res in enumerate(results):
                res['rank'] = i + 1
                
            return results[:top_k]
        except Exception as e:
            print(f"⚠️ Reranker error: {e}")
            return results[:top_k]

# สร้าง Singleton
reranker_manager = RerankerManager()
