# -*- coding: utf-8 -*-
"""
数据清洗脚本：将爬取的原始专利文档清洗为结构化 JSON
"""
#需要加入First Action for RCE
import os
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from pathlib import Path
pytesseract.pytesseract.tesseract_cmd = r"D:\software\tesseract\tesseract.exe"
import gc
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 新增：全局 Session 配置 ---
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

MY_POPPLER_PATH = r"D:\software\poppler-25.12.0\Library\bin"


BASE_SAVE_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_C_Category"
CLEANED_DATA_DIR = os.path.join(BASE_SAVE_DIR, "cleaned_json_output1") # 建议在原目录下建个子文件夹存 JSON

# 如需对已生成的 JSON 进行 oa_type/rejection_types 回填修正，可将该路径指向目标 cleaned_json_output1 目录
EXISTING_CLEANED_JSON_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_C_Category\cleaned_json_output1"


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def clean_xml_content(xml_path):
    """
    清洗 XML 文件，提取纯文本内容，移除标签和无关信息
    """
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 使用 BeautifulSoup 解析 HTML/XML
        soup = BeautifulSoup(content, 'html.parser')
        
        # 移除 <br> 与 </body> 之间的无关内容（文件名等）
        body = soup.find('body')
        if body:
            # 删除所有 br 标签后面的纯数字内容
            for br in body.find_all('br'):
                # 检查 br 后面的文本节点
                next_sibling = br.next_sibling
                while next_sibling:
                    if isinstance(next_sibling, str):
                        # 如果是纯数字或 -数字- 格式，删除
                        text = next_sibling.strip()
                        if text and (text.isdigit() or re.match(r'^-\d+(-\d+)?-?$', text)):
                            to_remove = next_sibling
                            next_sibling = next_sibling.next_sibling
                            to_remove.extract() if hasattr(to_remove, 'extract') else None
                            continue
                    break
        
        # 提取纯文本
        text = soup.get_text(separator='\n', strip=True)
        
        # 清理多余的空行和无意义的内容
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            # 跳过空行
            if not line:
                continue
            # Remove line-start coordinate residue like "0.090.08 "
            line = re.sub(r'^\d+\.\d+(?:\.\d+)*\s*', '', line)
            if not line:
                continue
            # Skip SVG placeholders / embed markers and OCR coordinate noise
            if 'SVG' in line or '{EMBED HERE}' in line:
                continue
            # 跳过纯数字行
            if line.isdigit():
                continue
            # 跳过 -数字- 格式的行
            if re.match(r'^-\d+(-\d+)?-?$', line):
                continue
            # Skip numeric-only lines with dots (e.g., OCR coordinates)
            if not re.search(r'[A-Za-z]', line) and re.search(r'\d', line):
                continue
            # 跳过 Date: 开头的行
            if line.startswith('Date:'):
                continue
            lines.append(line)
        
        cleaned_text = '\n'.join(lines)
        
        return cleaned_text
    except Exception as e:
        print(f"❌ 清洗 XML 失败 {xml_path}: {str(e)}")
        return ""


def clean_abstract_content(xml_path):
    """
    尝试提取更干净的 Abstract，过滤 PCT/WIPO 封面页噪声与表头信息
    """
    try:
        text = clean_xml_content(xml_path)
        if not text:
            return ""

        # If this looks like a PCT/WIPO cover page and no explicit ABSTRACT header, return empty
        up_full = text.upper()
        if ("INTERNATIONAL APPLICATION PUBLISHED UNDER THE PATENT COOPERATION TREATY" in up_full
            or "PATENT COOPERATION TREATY" in up_full
            or "INTERNATIONAL BUREAU" in up_full
            or "WIPO" in up_full):
            if "ABSTRACT" not in up_full:
                return ""

        lines = text.split('\n')
        # If an explicit ABSTRACT header exists, start after it
        start_idx = 0
        for i, line in enumerate(lines):
            up = line.strip().upper()
            if up == "ABSTRACT" or up.startswith("ABSTRACT "):
                start_idx = i + 1
                break
        lines = lines[start_idx:]

        stop_markers = {
            "CLAIMS", "BACKGROUND", "SUMMARY", "TECHNICAL FIELD", "FIELD",
            "BRIEF DESCRIPTION", "DETAILED DESCRIPTION"
        }
        bad_kw = [
            "INTERNATIONAL APPLICATION PUBLISHED UNDER THE PATENT COOPERATION TREATY",
            "PATENT COOPERATION TREATY",
            "INTERNATIONAL PUBLICATION NUMBER",
            "INTERNATIONAL PUBLICATION DATE",
            "INTERNATIONAL APPLICATION NUMBER",
            "INTERNATIONAL FILING DATE",
            "INTERNATIONAL BUREAU",
            "DESIGNATED STATES",
            "WIPO"
        ]

        cleaned = []
        for line in lines:
            up = line.strip().upper()
            if not up:
                continue
            if up in stop_markers:
                break
            if any(k in up for k in bad_kw):
                continue
            if re.match(r'^(W\s*O|WO)\s*\d{4}/\d+', up):
                continue
            if re.match(r'^PCT/\w+', up):
                continue
            cleaned.append(line)

        abstract = '\n'.join(cleaned).strip()
        if any(k in abstract.upper() for k in ["INTERNATIONAL BUREAU", "DESIGNATED STATES", "PCT/"]):
            return ""
        return abstract
    except Exception as e:
        print(f"鉂?娓呮礂 Abstract 澶辫触 {xml_path}: {str(e)}")
        return ""


def clean_office_action(txt_path):
    """
    清洗 Office Action 文本，移除 Contact 行及之后的内容
    注意：Office Action 中的 "Conclusion" 是章节标题，不应该截断
    """
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找 "Contact" 相关关键词并截断
        contact_patterns = [
            'Any inquiry concerning this communication',
            'Examiner interviews are available',
            'If attempts to reach the examiner'
        ]
        
        for pattern in contact_patterns:
            contact_idx = content.find(pattern)
            if contact_idx != -1:
                content = content[:contact_idx]
                break
        
        # 清理空行
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        cleaned_text = '\n'.join(lines)
        
        return cleaned_text
    except Exception as e:
        print(f"❌ 清洗 OA 失败 {txt_path}: {str(e)}")
        return ""


def is_oa_text_filename(filename):
    if not filename or not filename.endswith('.txt'):
        return False
    name = filename.lower()
    if 'summary' in name:
        return False
    keywords = [
        'office_action',
        'office action',
        'final_rejection',
        'final rejection',
        'non_final',
        'non-final',
        'nonfinal',
        'rejection',
        'notice_of_allowance',
        'notice of allowance',
        'allowance',
        'allowable subject matter',
        'granted',
        'advisory',
        'quayle',
    ]
    return any(k in name for k in keywords)


def clean_remarks(xml_path):
    """
    清洗 Remarks，保留到 Conclusion 之前的内容，并保留完整的段落
    """
    try:
        content = clean_xml_content(xml_path)
        
        # 查找 Conclusion 并保留完整段落
        conclusion_patterns = [
            r'Conclusion',
            r'CONCLUSION',
            r'Respectfully submitted',
            r'RESPECTFULLY SUBMITTED'
        ]
        
        # 按行分割
        lines = content.split('\n')
        result_lines = []
        found_conclusion = False
        
        for i, line in enumerate(lines):
            # 检查是否到达 Conclusion
            for pattern in conclusion_patterns:
                if pattern in line:
                    # 保留包含 Conclusion 的这一行
                    result_lines.append(line)
                    found_conclusion = True
                    break
            
            if found_conclusion:
                break
            
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    except Exception as e:
        print(f"❌ 清洗 Remarks 失败 {xml_path}: {str(e)}")
        return ""


def clean_claims_content(xml_path):
    """
    清洗 Claims，并修复常见断行问题（如 'claim 8wherein' -> 'claim 8 wherein'）
    """
    try:
        text = clean_xml_content(xml_path)
        if not text:
            return ""
        # Insert missing space after claim number
        text = re.sub(r'(\bclaim\s+\d+)([A-Za-z])', r'\1 \2', text, flags=re.IGNORECASE)
        # Fix common OCR join: "claim 8wherein" even without space after number
        text = re.sub(r'(\bclaim\s+\d+)\s*wherein', r'\1 wherein', text, flags=re.IGNORECASE)
        return text
    except Exception as e:
        print(f"鉂?娓呮礂 Claims 澶辫触 {xml_path}: {str(e)}")
        return ""


def extract_oa_metadata(oa_text):
    text_lower = oa_text.lower()
    metadata = {"oa_type": "Office Action", "rejection_types": []}

    # 0. 先提取 rejection_types（后续推断 oa_type 可能会用到）
    if '35 u.s.c. 102' in text_lower or '§102' in text_lower or '§ 102' in text_lower:
        metadata["rejection_types"].append("102 - Lack of Novelty")
    if '35 u.s.c. 103' in text_lower or '§103' in text_lower or '§ 103' in text_lower:
        metadata["rejection_types"].append("103 - Obviousness")
    if '35 u.s.c. 101' in text_lower or '§101' in text_lower or '§ 101' in text_lower:
        metadata["rejection_types"].append("101 - Patent-Eligible Subject Matter")
    if '35 u.s.c. 112' in text_lower or '§112' in text_lower or '§ 112' in text_lower:
        metadata["rejection_types"].append("112 - Written Description/Enablement")
    if 'double patenting' in text_lower:
        metadata["rejection_types"].append("Double Patenting")

    # 1. 优先识别程序性通知 (往往不带 Rejection 类型)
    if any(k in text_lower for k in ['restriction requirement', 'requirement for restriction', 'election requirement']):
        metadata["oa_type"] = "Restriction Requirement"
        return metadata
    if 'advisory action' in text_lower:
        metadata["oa_type"] = "Advisory Action"
        return metadata
    if 'ex parte quayle' in text_lower:
        metadata["oa_type"] = "Ex Parte Quayle Action"
        return metadata

    # 2. 识别授权 (需确保没有 active rejection)
    if re.search(r'\bnotice of allowance\b|\breasons for allowance\b', text_lower):
        if not re.search(r'\bare rejected\b|\bis rejected\b', text_lower):
            metadata["oa_type"] = "Notice of Allowance"
            return metadata

    # 3. 识别明确的 final / non-final
    if re.search(r'\bthis action is made final\b|\bfinal office action\b|\bfinal rejection\b', text_lower):
        metadata["oa_type"] = "Final Rejection"
        return metadata
    if re.search(r'\bnon[- ]?final (rejection|office action)\b', text_lower):
        metadata["oa_type"] = "Non-Final Rejection"
        return metadata

    # 4. 弱推断：如果出现了驳回法条/驳回措辞，但没有显式 final/non-final，默认按 Non-Final Rejection
    # 这是为了避免大量落到笼统的 "Office Action"。
    has_rejection_signal = False
    if metadata["rejection_types"]:
        has_rejection_signal = True
    if re.search(r'\b(are|is)\s+rejected\b', text_lower):
        has_rejection_signal = True
    if re.search(r'\brejected\s+under\b|\brejection\s+under\b', text_lower):
        has_rejection_signal = True
    if '35 u.s.c' in text_lower or 'u.s.c.' in text_lower or '§' in text_lower:
        # 出现大量法条符号通常也是 rejection 类 OA
        if metadata["rejection_types"]:
            has_rejection_signal = True

    if has_rejection_signal:
        metadata["oa_type"] = "Non-Final Rejection"
        return metadata

    # 5. 兜底
    return metadata


def parse_references_pdf(pdf_path):
    """
    解析 References Cited PDF 文件，提取引用专利信息
    先尝试直接提取表格，如果失败则使用 OCR
    返回格式: [
        {
            "document_number": "US-20010025289-A1",
            "date": "09-2001",
            "name": "Jenkins, Michael D.",
            "cpc_classification": "G06F1/3246",
            "us_classification": "708/141"
        },
        ...
    ]
    """
    references = []
    
    try:
        # 方法1：尝试直接提取表格（适用于文本 PDF）
        with pdfplumber.open(pdf_path) as pdf:
            has_tables = False
            for page in pdf.pages:
                tables = page.extract_tables()
                
                if tables:  # 发现表格
                    has_tables = True
                    for table in tables:
                        if not table:
                            continue
                        
                        # 查找表头行
                        header_row_idx = -1
                        for i, row in enumerate(table):
                            if row and any(cell and 'Document Number' in str(cell) for cell in row):
                                header_row_idx = i
                                break
                        
                        if header_row_idx == -1:
                            continue
                        
                        # 解析表头，找到各列的索引
                        header = table[header_row_idx]
                        col_indices = {
                            'document_number': -1,
                            'date': -1,
                            'name': -1,
                            'cpc': -1,
                            'us': -1
                        }
                        
                        for idx, cell in enumerate(header):
                            if not cell:
                                continue
                            cell_lower = str(cell).lower()
                            if 'document number' in cell_lower or 'country code' in cell_lower:
                                col_indices['document_number'] = idx
                            elif 'date' in cell_lower:
                                col_indices['date'] = idx
                            elif 'name' in cell_lower:
                                col_indices['name'] = idx
                            elif 'cpc' in cell_lower:
                                col_indices['cpc'] = idx
                            elif 'us class' in cell_lower:
                                col_indices['us'] = idx
                        
                        # 解析数据行
                        for i in range(header_row_idx + 1, len(table)):
                            row = table[i]
                            if not row or not any(row):
                                continue
                            
                            if col_indices['document_number'] >= 0 and col_indices['document_number'] < len(row):
                                doc_num = str(row[col_indices['document_number']] or '').strip()
                                if not doc_num or doc_num == '*' or len(doc_num) < 5:
                                    continue
                                
                                ref = {
                                    "document_number": normalize_doc_number(doc_num),
                                    "date": normalize_date(str(row[col_indices['date']] or '').strip()) if col_indices['date'] >= 0 and col_indices['date'] < len(row) else '',
                                    "name": str(row[col_indices['name']] or '').strip() if col_indices['name'] >= 0 and col_indices['name'] < len(row) else '',
                                    "cpc_classification": str(row[col_indices['cpc']] or '').strip() if col_indices['cpc'] >= 0 and col_indices['cpc'] < len(row) else '',
                                    "us_classification": str(row[col_indices['us']] or '').strip() if col_indices['us'] >= 0 and col_indices['us'] < len(row) else ''
                                }
                                references.append(ref)
            
            # 如果直接提取成功，返回结果
            if has_tables and references:
                if is_reference_quality_ok(references, source="table"):
                    return references
                print("  [WARN] Table parse low quality, fallback to OCR...")
        
        # 方法2：使用 OCR（适用于图片 PDF）
        print(f"  🔍 未检测到表格，尝试 OCR 解析...")
        references = parse_references_with_ocr(pdf_path)
        
    except Exception as e:
        print(f"⚠️ 解析 PDF 失败 {pdf_path}: {str(e)}")
    
    return references

# 允许中间出现空格或连字符
# Handles US1234567, 12,345,678, 2020/0123456, US-2020/0123456 etc.
DOC_NUM_RE = re.compile(r"(?:US[- ]?)?(?:\d{1,4}[/ ,]\d{3,11}(?:[- ]?[A-Z]\d?)?|\d{7,11}(?:[- ]?[A-Z]\d?)?)", re.IGNORECASE)
DATE_YMD_RE = re.compile(r"\b(20\d{2})[-/](\d{2})[-/](\d{2})\b")
DATE_MDY_RE = re.compile(r"\b(\d{2})[-/](\d{2})[-/](20\d{2})\b")
# Standard patent date formats: MM-DD-YYYY, YYYY-MM-DD, or MM-YYYY
DATE_STANDARD_RE = re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b|\b\d{1,2}[-/]\d{4}\b")
CPC_RE = re.compile(r"\b[A-H]\d{2}[A-Z]\d*/\d+\b")
CPC_FUZZ_RE = re.compile(r"\b[A-H][0-9OIS]{2}[A-Z][0-9OIS]{1,4}/[0-9OIS]{1,6}\b", re.IGNORECASE)
US_CLASS_RE = re.compile(r"\b\d{1,3}/\d{1,3}\b")
IGNORE_KEYWORDS = ["U.S. PATENT", "FOREIGN PATENT", "NON-PATENT", "OFFICE", "REVISION"]
def is_reference_quality_ok(references, source="table"):
    if not references:
        return False
    total = len(references)
    date_ok = sum(1 for r in references if r.get("date"))
    cpc_ok = sum(1 for r in references if r.get("cpc_classification"))
    us_ok = sum(1 for r in references if r.get("us_classification"))
    doc_ok = sum(1 for r in references if r.get("document_number"))
    
    # OCR is often missing metadata, so we are more lenient
    if source == "ocr":
        return doc_ok > 0
        
    if doc_ok < max(1, total // 2):
        return False
    if date_ok == 0 and cpc_ok == 0 and us_ok == 0:
        return False
    return True



def normalize_doc_number(raw):
    if not raw:
        return ''
    s = str(raw).upper().strip()
    s = s.replace(' ', '').replace('_', '').replace('/', '').replace(',', '')
    if s.startswith('US') and not s.startswith('US-'):
        s = 'US-' + s[2:]
    s = s.replace('US--', 'US-')
    return s


def normalize_date(raw):
    if not raw:
        return ''
    raw = str(raw).strip()
    m = DATE_YMD_RE.search(raw)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = DATE_MDY_RE.search(raw)
    if m:
        return f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
    return ''


def normalize_cpc(raw):
    if not raw:
        return ''
    s = str(raw).upper()
    s = s.replace('O', '0').replace('I', '1').replace('L', '1').replace('S', '5')
    m = CPC_RE.search(s)
    if m:
        return m.group(0)
    m = CPC_FUZZ_RE.search(s)
    if m:
        val = m.group(0)
        val = val.replace('O', '0').replace('I', '1').replace('L', '1').replace('S', '5')
        return val
    return ''


def clean_ocr_name(name):
    if not name:
        return ''
    n = name
    n = re.sub(r'\bU\.S\. PATENT DOCUMENTS\b', ' ', n, flags=re.IGNORECASE)
    n = re.sub(r'\bFOREIGN PATENT DOCUMENTS\b', ' ', n, flags=re.IGNORECASE)
    n = re.sub(r'\bPATENT DOCUMENTS\b', ' ', n, flags=re.IGNORECASE)
    n = re.sub(r'^[\s\*]+', ' ', n)
    n = re.sub(r'\b[A-D]\b', ' ', n)
    n = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', ' ', n)
    n = re.sub(r'\b\d{2}-\d{2}\b', ' ', n)
    n = re.sub(r'\b\d{8,}-[A-Z]\d?\b', ' ', n)
    n = re.sub(r'\b\d{8,}\b', ' ', n)
    n = re.sub(r'[^A-Za-z;,.\-\s]', ' ', n)
    n = re.sub(r'\s+', ' ', n).strip(' -;:,')
    return n


def parse_ocr_reference_line(line):
    # 1. 预清洗：洗掉常见的页脚和说明文字噪声
    temp_line = line.upper()
    if any(k in temp_line for k in IGNORE_KEYWORDS) or "PAGE" in temp_line:
        return None

    # 2. 匹配专利号
    m_doc = DOC_NUM_RE.search(line)
    if not m_doc:
        return None
    
    raw_doc_num = m_doc.group(0)
    
    # 防御性拦截：如果提取到的“专利号”里数字少于 5 个（如 "US-O"），判定为噪声
    if len(re.findall(r'\d', raw_doc_num)) < 5:
        return None

    # 3. 提取日期
    date = ''
    m_date = DATE_STANDARD_RE.search(line)
    if m_date:
        date = m_date.group(0)

    # 4. 提取分类
    cpc = normalize_cpc(line)
    m_us = US_CLASS_RE.search(line)
    us_class = m_us.group(0) if m_us else ""

    # 5. 提取姓名 (Name) - 核心逻辑：从 Block 中扣除已识别的字段
    # 先把识别出的 doc_num, date, cpc, us_class 从原始字符串中移除
    name_part = line
    for field in [raw_doc_num, date, cpc, us_class]:
        if field:
            name_part = name_part.replace(field, " ")
    
    # 进一步清洗掉表格线字符 | 和多余空格
    name_part = re.sub(r'[|,\*\_]', ' ', name_part)
    name = clean_ocr_name(name_part)

    return {
        "document_number": normalize_doc_number(raw_doc_num),
        "date": date,
        "name": name,
        "cpc_classification": cpc,
        "us_classification": us_class
    }



def parse_ocr_references_from_lines(lines, max_lookahead=5):
    references = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # 跳过表头行
        if "Document Number" in line or "U.S. PATENT DOCUMENTS" in line:
            i += 1
            continue
            
        # 只有发现类似专利号的特征才开始构建 Block
        if not DOC_NUM_RE.search(line):
            i += 1
            continue
            
        # 向下合并多行，直到覆盖到日期和姓名
        block = line
        j = i + 1
        while j < len(lines) and j <= i + max_lookahead:
            # 如果下一行又是新的专利号开头，则提前停止合并
            if DOC_NUM_RE.search(lines[j]) and len(re.findall(r'\d', lines[j])) > 5:
                break
            block += " " + lines[j]
            j += 1
            
        ref = parse_ocr_reference_line(block)
        if ref:
            references.append(ref)
        i = j
    return references



def parse_references_with_ocr(pdf_path):
    references = []
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(
            pdf_path,
            dpi=200,
            poppler_path=MY_POPPLER_PATH,
        )
        seen = set()
        for i, image in enumerate(images):
            print(f"    [OCR] Processing page {i+1}/{len(images)}...")
            text = pytesseract.image_to_string(image.convert('L'), lang='eng')
            if not text.strip():
                print(f"    [OCR] Page {i+1} returned no text.")
                continue
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            page_refs = parse_ocr_references_from_lines(lines)
            print(f"    [OCR] Page {i+1} found {len(page_refs)} candidate rows")
            for ref in page_refs:
                doc_num = ref.get('document_number', '')
                if not doc_num or doc_num in seen:
                    continue
                seen.add(doc_num)
                references.append(ref)
            image.close() # Close each image after processing
        
        if references:
            if is_reference_quality_ok(references, source="ocr"):
                print(f"  OK OCR parsed {len(references)} references")
            else:
                print(f"  [WARN] OCR parsed {len(references)} references but failed quality check.")
                references = []
        else:
            print(f"  [WARN] OCR finished but found no references. Check if images are empty.")
        gc.collect()
    except Exception as e:
        print(f"  ❌ OCR parse failed: {str(e)}")
    return references


def is_valid_office_action(oa_text):
    """
    检查是否是有效的 Office Action
    排除：DECISION ON PETITION、Fee Reminder 等非 OA 文档
    """
    if not oa_text or len(oa_text.strip()) < 50:
        return False
    
    text_lower = oa_text.lower()
    
    # 排除非 OA 类型的文档
    non_oa_keywords = [
        'decision on petition',
        'petition under',
        'petition is granted',
        'petition is denied',
        'fee reminder',
        'notice of missing parts',
        'filing receipt',
        'notice of abandonment'
    ]
    
    for keyword in non_oa_keywords:
        if keyword in text_lower:
            return False
    
    # 必须包含 OA 相关的关键词（放宽以提高容错）
    oa_keywords = [
        'office action',
        'non-final rejection',
        'final rejection',
        'notice of allowance',
        'detailed action',
        'examiner',
        'claim',
        'rejection',
        'u.s.c',
        '35 u.s.c'
    ]
    
    # 至少匹配一个 OA 关键词
    has_oa_keyword = any(keyword in text_lower for keyword in oa_keywords)
    if has_oa_keyword:
        return True
    # 兜底：包含法条符号也视为可能 OA
    if re.search(r'35\\s*u\\.s\\.c|u\\.s\\.c|§', text_lower):
        return True
    
    return False


def infer_oa_type_from_folder_name(folder_name):
    name = folder_name.lower()
    if 'notice_of_allowance' in name or 'notice of allowance' in name or 'allowance' in name:
        return 'Notice of Allowance'
    if 'final_rejection' in name or 'final rejection' in name or 'final office action' in name:
        return 'Final Rejection'
    if 'non-final' in name or 'non_final' in name or 'nonfinal' in name:
        return 'Non-Final Rejection'
    if 'restriction' in name:
        return 'Restriction Requirement'
    if 'advisory' in name:
        return 'Advisory Action'
    if 'quayle' in name:
        return 'Ex Parte Quayle Action'
    if 'office_action' in name or 'office action' in name:
        return 'Office Action'
    return ""


def process_oa_folder(oa_folder_path, oa_number):
    """
    处理单个 OA 文件夹，提取所有相关文档
    """
    oa_data = {
        "oa_number": oa_number,
        "oa_date": "",
        "oa_type": "",
        "rejection_types": [],
        "office_action_text": "",
        "current_abstract": "",
        "current_claims": "",
        "current_specification": "",
        "remarks": "",
        "next_claims": "",
        "drawings_path": "",  # Drawings PDF 的相对路径
        "references_cited": []  # 解析后的引用信息
    }
    
    # 从文件夹名提取日期
    folder_name = os.path.basename(oa_folder_path)
    folder_oa_type = infer_oa_type_from_folder_name(folder_name)
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', folder_name)
    if date_match:
        oa_data["oa_date"] = date_match.group(1)
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(oa_folder_path):
        filepath = os.path.join(oa_folder_path, filename)
        
        if not os.path.isfile(filepath):
            continue
        
        # Office Action 文本
        if is_oa_text_filename(filename):
            oa_text = clean_office_action(filepath)
            
            # 检查是否是有效的 Office Action
            if not is_valid_office_action(oa_text):
                if folder_oa_type:
                    print(f"  [WARN] OA text low-signal, keep by folder name: {folder_name}")
                else:
                    print(f"  ⚠️ 跳过无效 OA: {folder_name} (非真实 Office Action)")
                    return None  # 返回 None 表示跳过这个 OA
            
            oa_data["office_action_text"] = oa_text
            # 提取元数据
            metadata = extract_oa_metadata(oa_text)
            oa_data["oa_type"] = metadata["oa_type"]
            oa_data["rejection_types"] = metadata["rejection_types"]
        
        # Abstract
        elif 'Abstract' in filename and filename.endswith('.xml'):
            oa_data["current_abstract"] = clean_abstract_content(filepath)
        
        # Claims (当前版本)
        elif 'Claims' in filename and filename.endswith('.xml'):
            if not oa_data["current_claims"]:  # 第一个 Claims 作为当前版本
                oa_data["current_claims"] = clean_claims_content(filepath)
            else:  # 第二个 Claims 作为 next version
                oa_data["next_claims"] = clean_claims_content(filepath)
        
        # Specification
        elif 'Specification' in filename and filename.endswith('.xml'):
            oa_data["current_specification"] = clean_xml_content(filepath)
        
        # Remarks
        elif 'Remarks' in filename and filename.endswith('.xml'):
            oa_data["remarks"] = clean_remarks(filepath)
        
        # Drawings (PDF - 保存相对路径)
        elif 'Drawings' in filename and filename.endswith('.pdf'):
            # 计算相对于 BASE_SAVE_DIR 的相对路径
            rel_path = os.path.relpath(filepath, BASE_SAVE_DIR)
            oa_data["drawings_path"] = rel_path
        
        # References Cited (PDF - 解析并提取信息)
        elif 'References_Cited' in filename and filename.endswith('.pdf'):
            references = parse_references_pdf(filepath)
            oa_data["references_cited"] = references
            if references:
                print(f"  ✅ 解析到 {len(references)} 条引用")
    
    # If oa_type is empty, fallback to folder name
    if not oa_data["oa_type"] and folder_oa_type:
        oa_data["oa_type"] = folder_oa_type

    return oa_data


def process_patent(patent_folder_path):
    """
    处理单个专利文件夹，生成完整的 JSON 数据
    """
    patent_name = os.path.basename(patent_folder_path)
    
    # 提取申请号和专利号
    app_match = re.search(r'App_(\d+)', patent_name)
    patent_match = re.search(r'Patent_(\w+)_', patent_name)
    
    patent_data = {
        "patent_number": patent_match.group(1) if patent_match else "Unknown",
        "application_number": app_match.group(1) if app_match else "Unknown",
        "initial_submission": {
            "abstract": "",
            "claims": "",
            "specification": "",
            "drawings_path": ""  # Drawings PDF 的相对路径
        },
        "office_actions": []
    }
    
    # 处理初始提交
    initial_dir = os.path.join(patent_folder_path, "00_Initial_Submission")
    if os.path.exists(initial_dir):
        for filename in os.listdir(initial_dir):
            filepath = os.path.join(initial_dir, filename)
            if 'Abstract' in filename and filename.endswith('.xml'):
                patent_data["initial_submission"]["abstract"] = clean_abstract_content(filepath)
            elif 'Claims' in filename and filename.endswith('.xml'):
                patent_data["initial_submission"]["claims"] = clean_claims_content(filepath)
            elif 'Specification' in filename and filename.endswith('.xml'):
                patent_data["initial_submission"]["specification"] = clean_xml_content(filepath)
            elif 'Drawings' in filename and filename.endswith('.pdf'):
                # 保存 Drawings PDF 的相对路径
                rel_path = os.path.relpath(filepath, BASE_SAVE_DIR)
                patent_data["initial_submission"]["drawings_path"] = rel_path
    
    # 处理所有 OA 文件夹
    oa_folders = sorted([
        f for f in os.listdir(patent_folder_path)
        if f.startswith('OA') and os.path.isdir(os.path.join(patent_folder_path, f))
    ])
    
    for idx, oa_folder in enumerate(oa_folders, 1):
        oa_folder_path = os.path.join(patent_folder_path, oa_folder)
        oa_data = process_oa_folder(oa_folder_path, idx)
        
        # 如果返回 None，说明是无效 OA，跳过
        if oa_data is not None:
            patent_data["office_actions"].append(oa_data)
    
    return patent_data


def backfill_existing_cleaned_json(input_dir, output_dir=None):
    """对已生成的 cleaned json 进行 oa_type/rejection_types 回填修正。

    - input_dir: 现有 cleaned_json_output1 目录
    - output_dir: 若为 None 则原地覆盖；否则写入到新目录
    """
    if not input_dir or not os.path.isdir(input_dir):
        print(f"❌ backfill 输入目录不存在: {input_dir}")
        return

    if output_dir:
        ensure_dir(output_dir)

    json_files = [
        f for f in os.listdir(input_dir)
        if f.endswith('.json') and os.path.isfile(os.path.join(input_dir, f))
    ]
    print(f"📦 backfill 发现 {len(json_files)} 个 json")

    updated_files = 0
    updated_oa = 0

    for fn in json_files:
        in_path = os.path.join(input_dir, fn)
        out_path = os.path.join(output_dir, fn) if output_dir else in_path

        try:
            with open(in_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            oas = data.get('office_actions') or []
            file_changed = False

            for oa in oas:
                oa_text = oa.get('office_action_text') or ''
                if not oa_text:
                    continue

                meta = extract_oa_metadata(oa_text)
                new_type = meta.get('oa_type')
                new_rej = meta.get('rejection_types') or []

                if new_type and oa.get('oa_type') != new_type:
                    oa['oa_type'] = new_type
                    file_changed = True

                # rejection_types 用重算结果覆盖（更一致）
                if oa.get('rejection_types') != new_rej:
                    oa['rejection_types'] = new_rej
                    file_changed = True

                if file_changed:
                    updated_oa += 1

            if file_changed:
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                updated_files += 1

        except Exception as e:
            print(f"❌ backfill 失败 {fn}: {str(e)}")

        finally:
            gc.collect()

    print(f"✅ backfill 完成：更新文件 {updated_files}/{len(json_files)}，涉及 OA(粗略计数) {updated_oa}")


def main():
    ensure_dir(CLEANED_DATA_DIR)

    # 1. 获取已经生成的 JSON 文件名（不含后缀）
    processed_patents = {
        Path(f).stem for f in os.listdir(CLEANED_DATA_DIR)
        if f.endswith('.json')
    }

    patent_folders = [
        f for f in os.listdir(BASE_SAVE_DIR)
        if f.startswith('Patent_') and os.path.isdir(os.path.join(BASE_SAVE_DIR, f))
    ]

    # 2. 过滤掉已处理的
    # to_process = [f for f in patent_folders if f not in processed_patents]
    # print(f"📊 找到 {len(patent_folders)} 个专利，已处理 {len(processed_patents)} 个，待处理 {len(to_process)} 个")

    to_process = patent_folders 
    print(f"📊 找到 {len(patent_folders)} 个专利，将全部执行重新清洗（覆盖模式）")

    for patent_folder in to_process:
        print(f"\n🔄 正在处理: {patent_folder}")
        patent_folder_path = os.path.join(BASE_SAVE_DIR, patent_folder)

        try:
            # 处理专利数据
            patent_data = process_patent(patent_folder_path)

            # 保存为 JSON
            output_filename = f"{patent_folder}.json"
            output_path = os.path.join(CLEANED_DATA_DIR, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(patent_data, f, ensure_ascii=False, indent=2)

            print(f"✅ 已保存: {output_filename}")
            print(f"   - 初始提交: {'✓' if patent_data['initial_submission']['claims'] else '✗'}")
            print(f"   - Office Actions: {len(patent_data['office_actions'])} 个")

        except Exception as e:
            print(f"❌ 处理失败: {str(e)}")

        finally:
            # 每一轮循环结束强行清理一次内存
            gc.collect()

    print(f"\n🎉 清洗完成！数据保存在: {CLEANED_DATA_DIR}")


if __name__ == "__main__":
    # 如需回填修正，取消注释并按需设置输出目录
    backfill_existing_cleaned_json(EXISTING_CLEANED_JSON_DIR, output_dir=None)
    main()
