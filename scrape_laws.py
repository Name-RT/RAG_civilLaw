import requests
import re
import os
import json
import sys
import time

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Import keywords from utils.py to classify categories
try:
    from utils import classify_query, CATEGORY_KEYWORDS
except ImportError:
    CATEGORY_KEYWORDS = [
        ("มรดก", ["มรดก", "พินัยกรรม", "ทายาท", "รับมรดก", "ผู้จัดการมรดก"]),
        ("ครอบครัว", ["สมรส", "แต่งงาน", "หย่า", "หมั้น", "คู่สมรส", "บุตร"]),
        ("ละเมิด", ["ละเมิด", "ชนรถ", "ทำให้เสียหาย", "ประมาท", "ค่าเสียหาย"]),
        ("จำนองและจำนำ", ["จำนอง", "จำนำ", "บังคับจำนอง", "ไถ่ถอน"]),
        ("กู้ยืมและค้ำประกัน", ["กู้", "ยืมเงิน", "เงินกู้", "ค้ำประกัน", "ผู้ค้ำ"]),
        ("เช่าทรัพย์และเช่าซื้อ", ["เช่า", "เช่าบ้าน", "เช่าซื้อ", "ผู้เช่า"]),
        ("ซื้อขาย", ["ซื้อขาย", "ซื้อของ", "ขายของ", "ชำรุด", "บกพร่อง", "ขายฝาก"]),
        ("จ้างงาน", ["จ้างงาน", "จ้างแรงงาน", "ลูกจ้าง", "นายจ้าง", "ค่าจ้าง"]),
        ("ฝากทรัพย์และยืมใช้", ["ฝากของ", "ฝากทรัพย์", "ยืมของ", "ยืมใช้"]),
        ("ตัวแทนและนายหน้า", ["ตัวแทน", "นายหน้า", "มอบอำนาจ", "ค่านายหน้า"]),
        ("นิติบุคคล", ["บริษัท", "ห้างหุ้นส่วน", "จดทะเบียน", "หุ้น", "กรรมการ"]),
        ("ทรัพย์และทรัพย์สิน", ["กรรมสิทธิ์", "ครอบครอง", "ที่ดิน", "โฉนด"]),
        ("บุคคลและความสามารถทางกฎหมาย", ["ผู้เยาว์", "เด็ก", "บรรลุนิติภาวะ", "คนบ้า"]),
        ("หนี้และการชำระหนี้", ["หนี้", "เจ้าหนี้", "ลูกหนี้", "ชำระหนี้", "ผิดนัด"]),
        ("นิติกรรมและสัญญาทั่วไป", ["สัญญา", "ข้อตกลง", "ผิดสัญญา", "บอกเลิก", "โมฆะ"]),
    ]
    def classify_query(query: str):
        scores = {}
        for cat_name, keywords in CATEGORY_KEYWORDS:
            count = sum(1 for kw in keywords if kw in query)
            if count > 0:
                scores[cat_name] = count
        if not scores:
            return "หลักทั่วไป"
        return max(scores, key=lambda k: scores[k])

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def translate_thai_to_arabic(text):
    thai_digits = "๐๑๒๓๔๕๖๗๘๙"
    arabic_digits = "0123456789"
    trans = str.maketrans(thai_digits, arabic_digits)
    return text.translate(trans)

def get_dropdown_urls(landing_url, pattern_regex):
    print(f"📥 กำลังดึงรายชื่อลิงก์จาก: {landing_url} ...")
    try:
        response = requests.get(landing_url, headers=HEADERS)
        response.encoding = 'utf-8'
        if response.status_code != 200:
            print(f"❌ ดึงข้อมูลสารบัญล้มเหลว: {response.status_code}")
            return []
            
        options = re.findall(pattern_regex, response.text)
        urls = []
        for url, name in options:
            if not url.startswith("http"):
                url = "https://www.drthawip.com" + url
            urls.append((url, name.strip()))
        
        # กรองเอา URL ที่ซ้ำออก
        unique_urls = []
        seen = set()
        for url, name in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append((url, name))
                
        print(f"✅ พบลิงก์กลุ่มมาตราทั้งหมด {len(unique_urls)} กลุ่ม")
        return unique_urls
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return []

def scrape_sections_from_url(url, law_type, law_name):
    max_retries = 3
    retry_delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.encoding = 'utf-8'
            
            if response.status_code != 200:
                print(f"  ⚠️ [Attempt {attempt}/{max_retries}] Status {response.status_code} for {url}. Retrying...")
                time.sleep(retry_delay)
                continue
                
            html_content = response.text
            
            # ตรวจสอบ Soft-block จาก WAF
            if "Not Acceptable" in html_content or "mod_security" in html_content:
                print(f"  ⚠️ [Attempt {attempt}/{max_retries}] Soft-blocked by WAF. Retrying in {retry_delay * 2}s...")
                time.sleep(retry_delay * 2)
                continue

            # หา start position ของ field-item even
            start_marker = 'class="field-item even"'
            start_pos = html_content.find(start_marker)
            if start_pos == -1:
                start_marker = 'property="content:encoded"'
                start_pos = html_content.find(start_marker)

            if start_pos == -1:
                print(f"  ⚠️ [Attempt {attempt}/{max_retries}] Content div not found. Retrying...")
                time.sleep(retry_delay)
                continue

            # หา > ที่ปิด opening tag แล้วเริ่มนับเนื้อหา
            tag_end = html_content.find('>', start_pos)
            if tag_end == -1:
                time.sleep(retry_delay)
                continue
            content_start = tag_end + 1
            
            # นับ depth ของ div เพื่อหา closing tag ที่ถูกต้อง
            depth = 1
            pos = content_start
            while pos < len(html_content) and depth > 0:
                open_tag = html_content.find('<div', pos)
                close_tag = html_content.find('</div>', pos)
                if close_tag == -1:
                    break
                if open_tag != -1 and open_tag < close_tag:
                    depth += 1
                    pos = open_tag + 4
                else:
                    depth -= 1
                    pos = close_tag + 6

            html_chunk = html_content[content_start:pos - 6]
            
            # ตัดเชิงอรรถ/Footnotes ออก (<hr /> อยู่ภายใน field-item div)
            if "<hr />" in html_chunk:
                html_chunk = html_chunk.split("<hr />")[0]

            # แปลง tag บล็อกให้เป็น newline ก่อน
            html_chunk = html_chunk.replace("<p>", "\n").replace("</p>", "\n")
            html_chunk = html_chunk.replace("<br />", "\n").replace("<br>", "\n")
            html_chunk = html_chunk.replace("</div>", "\n")
            
            # ล้าง tag อื่นๆ ทั้งหมด
            text = re.sub(r'<[^>]+>', '', html_chunk)
            text = text.replace("&nbsp;", " ").replace("\xa0", " ")
            text = "\n" + text
            
            # แยกมาตรา (มองหามาตราที่ขึ้นต้นบรรทัดใหม่)
            pattern = r'(?=\n[ \t]*มาตรา\s+[๐-๙0-9]+)'
            parts = re.split(pattern, text)
            
            sections_found = []
            for part in parts:
                part_clean = part.strip()
                if not part_clean:
                    continue
                    
                lines = [re.sub(r'\s+', ' ', line).strip() for line in part_clean.split("\n")]
                cleaned_text = "\n".join([l for l in lines if l])
                
                if not cleaned_text.startswith("มาตรา"):
                    continue
                    
                # ดึงชื่อมาตราออก เช่น มาตรา ๓๓๖ ทวิ หรือ มาตรา ๒๗๖/๑
                section_match = re.match(
                    r'^(มาตรา\s+[๐-๙0-9]+(?:/[๐-๙0-9]+)?(?:\s*(?:ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัปต|อัฐ|นพ|ทศ))?)', 
                    cleaned_text
                )
                if section_match:
                    section_name = section_match.group(1)
                else:
                    section_name = "มาตราไม่ทราบ"
                    
                # แยกหมวดหมู่ย่อย
                category = classify_query(cleaned_text[:300]) or "ทั่วไป"
                
                sections_found.append({
                    "law_type": law_type,
                    "law_name": law_name,
                    "section": section_name,
                    "category": category,
                    "original_text": cleaned_text,
                    "simplified_text": ""
                })
                
            if sections_found:
                return sections_found
            else:
                print(f"  ⚠️ [Attempt {attempt}/{max_retries}] 0 sections parsed for {url}. Retrying...")
                time.sleep(retry_delay)
                
        except Exception as e:
            print(f"  ⚠️ [Attempt {attempt}/{max_retries}] Error fetching {url}: {e}. Retrying...")
            time.sleep(retry_delay)
            
    print(f"  ❌ ล้มเหลวโดยสิ้นเชิงในการดึงลิงก์: {url}")
    return []

def main():
    all_laws = []
    
    # 1. ดึงประมวลกฎหมายอาญา
    print("=============================================")
    print("⚖️ กำลังเริ่มดึงประมวลกฎหมายอาญา (ฉบับหลัก)")
    print("=============================================")
    crim_landing = "https://www.drthawip.com/criminalcode/1-3"
    crim_regex = r'<option value="([^"]*criminalcode/[^"]*)">มาตรา ([^<\n]*)'
    crim_urls = get_dropdown_urls(crim_landing, crim_regex)
    
    for idx, (url, name) in enumerate(crim_urls, 1):
        print(f"[{idx}/{len(crim_urls)}] กำลังดึง {name} ({url}) ...")
        secs = scrape_sections_from_url(url, "อาญา", "ประมวลกฎหมายอาญา")
        all_laws.extend(secs)
        print(f"  -> พบ {len(secs)} มาตรา")
        time.sleep(1.0)  # หน่วงเวลา 1.0 วินาที

    # 2. ดึงประมวลกฎหมายแพ่งและพาณิชย์
    print("\n=============================================")
    print("⚖️ กำลังเริ่มดึงประมวลกฎหมายแพ่งและพาณิชย์ (ฉบับหลัก)")
    print("=============================================")
    civil_landing = "https://www.drthawip.com/civilandcommercialcode/004"
    civil_regex = r'<option value="([^"]*civilandcommercialcode/[^"]*)">มาตรา ([^<\n]*)'
    civil_urls = get_dropdown_urls(civil_landing, civil_regex)
    
    for idx, (url, name) in enumerate(civil_urls, 1):
        print(f"[{idx}/{len(civil_urls)}] กำลังดึง {name} ({url}) ...")
        secs = scrape_sections_from_url(url, "แพ่ง", "ประมวลกฎหมายแพ่งและพาณิชย์")
        all_laws.extend(secs)
        print(f"  -> พบ {len(secs)} มาตรา")
        time.sleep(1.0)  # หน่วงเวลา 1.0 วินาที

    print(f"\n🎉 ดึงข้อมูลกฎหมายเสร็จสมบูรณ์!")
    print(f"- ดึงข้อมูลมาตรากฎหมายได้ทั้งหมด: {len(all_laws)} มาตรา")
    
    # บันทึกไฟล์ JSON
    output_path = "data/processed_laws.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_laws, f, ensure_ascii=False, indent=4)
        
    print(f"💾 บันทึกไฟล์ข้อมูลสำเร็จที่: {output_path}")

if __name__ == "__main__":
    main()
