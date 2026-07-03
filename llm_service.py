import os
import json
from typing import Optional

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
