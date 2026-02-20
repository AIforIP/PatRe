# -*- coding: utf-8 -*-
"""
数据清洗脚本：将爬取的原始专利文档清洗为结构化 JSON
"""
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

MY_POPPLER_PATH = r"D:\software\poppler-25.12.0\Library\bin"


BASE_SAVE_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_G_Category"
CLEANED_DATA_DIR = os.path.join(BASE_SAVE_DIR, "cleaned_json_output") # 建议在原目录下建个子文件夹存 JSON


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
            # 跳过纯数字行
            if line.isdigit():
                continue
            # 跳过 -数字- 格式的行
            if re.match(r'^-\d+(-\d+)?-?$', line):
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


def extract_oa_metadata(oa_text):
    """
    从 Office Action 中提取元数据（类型、rejection 类型等）
    """
    metadata = {
        "oa_type": "Office Action",  # 默认为 Office Action 而不是 Unknown
        "rejection_types": []
    }
    
    text_lower = oa_text.lower()
    
    # 识别 OA 类型（按优先级顺序）
    # Allowance indicators should win only if there is no active rejection in this OA.
    def _has_active_rejection(txt: str) -> bool:
        for seg in re.split(r'[.\n]', txt):
            if not seg:
                continue
            if 'withdrawn' in seg:
                continue
            if re.search(r'\\b(are|is|be|been)\\s+rejected\\b', seg):
                return True
            if re.search(r'\\brejected\\s+under\\b|\\brejected\\s+on\\s+the\\s+ground\\b', seg):
                return True
            if 'provisionally rejected' in seg:
                return True
        return False

    active_rejection = _has_active_rejection(text_lower)

    if re.search(r'\\bnotice of allowance\\b|\\breasons for allowance\\b|\\ballowable\\b|\\ballowed\\b', text_lower) and not active_rejection:
        metadata["oa_type"] = "Notice of Allowance"
    elif re.search(r'\\bnon[-\\s]?final (rejection|office action)\\b', text_lower) and active_rejection:
        metadata["oa_type"] = "Non-Final Rejection"
    elif re.search(r'\\bthis action is made final\\b|\\bfinal office action\\b|\\bfinal rejection\\b', text_lower) and active_rejection:
        metadata["oa_type"] = "Final Rejection"
    elif active_rejection:
        metadata["oa_type"] = "Office Action"
    elif 'restriction requirement' in text_lower or 'requirement for restriction' in text_lower or 'election requirement' in text_lower:
        metadata["oa_type"] = "Restriction Requirement"
    elif 'advisory action' in text_lower:
        metadata["oa_type"] = "Advisory Action"
    elif 'ex parte quayle' in text_lower:
        metadata["oa_type"] = "Ex Parte Quayle Action"
    elif 'office action' in text_lower:
        metadata["oa_type"] = "Office Action"
    else:
        # 如果通过了 is_valid_office_action 但没匹配到具体类型，仍标记为 Office Action
        metadata["oa_type"] = "Office Action"
    
    # 识别 rejection 类型
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
                            elif 'date' in cell_lower and 'mm-y' in cell_lower:
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
                                    "document_number": doc_num,
                                    "date": str(row[col_indices['date']] or '').strip() if col_indices['date'] >= 0 and col_indices['date'] < len(row) else '',
                                    "name": str(row[col_indices['name']] or '').strip() if col_indices['name'] >= 0 and col_indices['name'] < len(row) else '',
                                    "cpc_classification": str(row[col_indices['cpc']] or '').strip() if col_indices['cpc'] >= 0 and col_indices['cpc'] < len(row) else '',
                                    "us_classification": str(row[col_indices['us']] or '').strip() if col_indices['us'] >= 0 and col_indices['us'] < len(row) else ''
                                }
                                references.append(ref)
            
            # 如果直接提取成功，返回结果
            if has_tables and references:
                return references
        
        # 方法2：使用 OCR（适用于图片 PDF）
        print(f"  🔍 未检测到表格，尝试 OCR 解析...")
        references = parse_references_with_ocr(pdf_path)
        
    except Exception as e:
        print(f"⚠️ 解析 PDF 失败 {pdf_path}: {str(e)}")
    
    return references


def parse_references_with_ocr(pdf_path):
    references = []
    try:
        # 优化点：使用 generator 逐页处理，不一次性加载所有图片
        # 限制只扫描前 5 页（通常专利引用表不会太长，防止无限扫描）
        from pdf2image import convert_from_path
        
        # 增加内存清理设置
        images = convert_from_path(
            pdf_path, 
            dpi=200, # 适当降低 DPI（200 足够 OCR），大幅减少内存占用
            poppler_path=MY_POPPLER_PATH,
            #last_page=10 # 强制限制最大页数，防止遇到异常巨大的 PDF
        )
        
        for image in images:
            text = pytesseract.image_to_string(image.convert('L'), lang='eng')
            
            # 解析文本提取引用信息
            lines = text.split('\n')
            
            # 查找表头（包含 Document Number）
            header_idx = -1
            for i, line in enumerate(lines):
                if 'Document Number' in line or 'Country Code' in line:
                    header_idx = i
                    break
            
            if header_idx == -1:
                continue
            
            # 解析数据行（简单的正则匹配）
            for i in range(header_idx + 1, len(lines)):
                line = lines[i].strip()
                if not line or len(line) < 10:
                    continue
                
                # 匹配专利号格式：US-XXXXXXXX-XX 或 USXXXXXXXX
                doc_match = re.search(r'(US[- ]?\d{4,}[- ]?[A-Z]\d?)', line, re.IGNORECASE)
                if doc_match:
                    doc_num = doc_match.group(1)
                    
                    # 尝试提取日期（MM-YYYY 或 MM/YYYY）
                    date_match = re.search(r'(\d{2}[-/]\d{4})', line)
                    date_str = date_match.group(1) if date_match else ''
                    
                    # 尝试提取姓名（假设在专利号后面）
                    name_match = re.search(r'([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*)', line)
                    name_str = name_match.group(1) if name_match else ''
                    
                    ref = {
                        "document_number": doc_num,
                        "date": date_str,
                        "name": name_str,
                        "cpc_classification": "",
                        "us_classification": ""
                    }
                    references.append(ref)
        
        if references:
            print(f"  ✅ OCR 解析到 {len(references)} 条引用")
    
        image.close() 
        gc.collect() # 显式清理内存
            
    except Exception as e:
        print(f"  ⚠️ OCR 解析失败: {str(e)}")
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
    
    # 必须包含 OA 相关的关键词
    oa_keywords = [
        'office action',
        'non-final rejection',
        'final rejection',
        'notice of allowance',
        'examiner',
        'claim',
        'rejection'
    ]
    
    # 至少匹配一个 OA 关键词
    has_oa_keyword = any(keyword in text_lower for keyword in oa_keywords)
    
    return has_oa_keyword


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
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', folder_name)
    if date_match:
        oa_data["oa_date"] = date_match.group(1)
    
    # 遍历文件夹中的所有文件
    for filename in os.listdir(oa_folder_path):
        filepath = os.path.join(oa_folder_path, filename)
        
        if not os.path.isfile(filepath):
            continue
        
        # Office Action 文本
        if 'Office_Action' in filename and filename.endswith('.txt'):
            oa_text = clean_office_action(filepath)
            
            # 检查是否是有效的 Office Action
            if not is_valid_office_action(oa_text):
                print(f"  ⚠️ 跳过无效 OA: {folder_name} (非真实 Office Action)")
                return None  # 返回 None 表示跳过这个 OA
            
            oa_data["office_action_text"] = oa_text
            # 提取元数据
            metadata = extract_oa_metadata(oa_text)
            oa_data["oa_type"] = metadata["oa_type"]
            oa_data["rejection_types"] = metadata["rejection_types"]
        
        # Abstract
        elif 'Abstract' in filename and filename.endswith('.xml'):
            oa_data["current_abstract"] = clean_xml_content(filepath)
        
        # Claims (当前版本)
        elif 'Claims' in filename and filename.endswith('.xml'):
            if not oa_data["current_claims"]:  # 第一个 Claims 作为当前版本
                oa_data["current_claims"] = clean_xml_content(filepath)
            else:  # 第二个 Claims 作为 next version
                oa_data["next_claims"] = clean_xml_content(filepath)
        
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
                patent_data["initial_submission"]["abstract"] = clean_xml_content(filepath)
            elif 'Claims' in filename and filename.endswith('.xml'):
                patent_data["initial_submission"]["claims"] = clean_xml_content(filepath)
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
    to_process = [f for f in patent_folders if f not in processed_patents]
    print(f"📊 找到 {len(patent_folders)} 个专利，已处理 {len(processed_patents)} 个，待处理 {len(to_process)} 个")
    
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
    main()
