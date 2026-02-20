# -*- coding: utf-8 -*-
import os
import json
import time
import random
import requests
import re
import threading
from bs4 import BeautifulSoup
from bs4.element import Tag
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 1. 路径与全局配置 =====================
BASE_DATA_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset"
TARGET_CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
REF_POOL_DIR = os.path.join(BASE_DATA_DIR, "ReferencePool_Full")
STATUS_LOG_FILE = os.path.join(REF_POOL_DIR, "crawl_status.json")

# 并发数建议：3-5。设为 4 理论速度提升 3-4 倍。
MAX_WORKERS = 4 
STATUS_LOCK = threading.Lock() # 线程锁，保护状态写操作

PROXIES = {"http": "http://127.0.0.1:8081", "https": "http://127.0.0.1:8081"}
HEADERS_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0"
]

# ===================== 2. 解析工具函数 =====================

def standardize_pid(pid):
    if not pid: return ""
    p = re.sub(r'[^a-zA-Z0-9]', '', str(pid)).upper()
    if not p: return ""
    if p[0].isdigit(): p = "US" + p
    return p

def extract_patent_metadata(soup):
    title = ""
    t_meta = soup.find("meta", {"name": "DC.title"})
    if t_meta: title = t_meta.get("content")
    
    date = ""
    d_meta = soup.find("meta", {"name": "DC.date", "scheme": "dateSubmitted"}) or \
             soup.find("meta", {"itemprop": "publicationDate"})
    if d_meta: date = d_meta.get("content")
    return title, date

def extract_complete_summary(soup):
    def _is_stop_heading(text):
        t = text.lower()
        return any(stop in t for stop in ['background', 'brief', 'detailed', 'claims'])
    
    def _is_noise(text):
        t = text.lower()
        return any(noise in t[:50] for noise in ['claim of priority', 'cross-reference'])

    content_parts = []
    # 使用 string 参数查找标题
    start_node = soup.find(['heading', 'h2', 'h3'], string=re.compile(r'^SUMMARY$', re.I))
    if not start_node:
        start_node = soup.find(['heading', 'h2', 'h3'], string=re.compile(r'Summary of (the )?Invention', re.I))

    if start_node:
        for sibling in start_node.find_next_siblings():
            sib_tag = sibling.name
            sib_text = sibling.get_text().strip()
            if sib_tag in ['heading', 'h2', 'h3'] and _is_stop_heading(sib_text): break
            if 'description-paragraph' in sibling.get('class', []) or sib_tag in ['p', 'div']:
                if sib_text and not _is_noise(sib_text): content_parts.append(sib_text)
        if content_parts: return "\n\n".join(content_parts)

    return "Summary Content Not Found"

# ===================== 3. 核心单步爬取函数 =====================

def process_one_patent(pid, provenance_db, session):
    url = f"https://patents.google.com/patent/{pid}/en"
    try:
        headers = {"User-Agent": random.choice(HEADERS_LIST), "Accept-Language": "en-US,en;q=0.9"}
        resp = session.get(url, headers=headers, proxies=PROXIES, timeout=30)
        
        if resp.status_code != 200:
            return pid, {"status": resp.status_code, "url": url}
        
        soup = BeautifulSoup(resp.text, "lxml")
        title, pub_date = extract_patent_metadata(soup)
        
        abs_tag = soup.find("section", {"itemprop": "abstract"}) or soup.find("div", class_="abstract")
        abstract_text = abs_tag.get_text(" ", strip=True) if abs_tag else ""
        
        summary_text = extract_complete_summary(soup)
        
        raw_claims = [div.get_text(" ", strip=True) for div in soup.find_all("div", class_="claim-text") 
                      if not div.find_parent("div", class_="claim-text")]
        claims_list = [c for c in raw_claims if len(c) > 10]
        
        desc_sec = soup.find("section", {"itemprop": "description"})
        full_desc = desc_sec.get_text(" ", strip=True) if desc_sec else ""

        quality_flag = "OK" if (abstract_text and summary_text != "Summary Content Not Found" and claims_list) else "Partial"

        # 整理结果数据
        result_data = {
            "patent_number": pid,
            "std_pid": pid,
            "title": title,
            "publication_date": pub_date,
            "abstract": abstract_text,
            "summary_of_invention": summary_text,
            "full_description": full_desc,
            "claims": claims_list,
            "provenance": provenance_db.get(pid, []),
            "data_quality": quality_flag,
            "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 填充默认字段补全
        default_fields = {
            "application_number": "", "category": "", "source_file": "",
            "oa_index": None, "oa_number": None, "oa_date": "", "oa_type": "",
            "rejection_types": [], "references_cited": [], "oa_text_len": None,
            "match_kind": "", "match_raw": "", "match_norm": "", "match_count": 0, "context": ""
        }
        for k, v in default_fields.items():
            result_data.setdefault(k, v)

        # 保存 JSON 文件
        save_path = os.path.join(REF_POOL_DIR, f"{pid}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        return pid, {
            "status": 200,
            "quality": quality_flag,
            "crawl_time": result_data["crawl_time"]
        }
    except Exception as e:
        return pid, {"status": "Error", "msg": str(e)}

# ===================== 4. 主任务控制 =====================

def is_reference_complete(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        summary = data.get("summary_of_invention", "")
        claims = data.get("claims", [])
        return summary and summary != "Summary Content Not Found" and len(claims) > 0
    except: return False

def analyze_provenance():
    provenance_map = {}
    print("🔍 正在扫描主数据集建立溯源关系...")
    for cat in TARGET_CATEGORIES:
        cat_dir = os.path.join(BASE_DATA_DIR, f"PatTimeline_{cat}_Category", "cleaned_json_output")
        if not os.path.exists(cat_dir): continue
        for filename in os.listdir(cat_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(cat_dir, filename), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        p_num = data.get("patent_number", "Unknown")
                        for oa in data.get("office_actions", []):
                            for ref in oa.get("references_cited", []):
                                std_p = standardize_pid(ref.get("document_number", ""))
                                if not std_p: continue
                                if std_p not in provenance_map: provenance_map[std_p] = []
                                provenance_map[std_p].append({"cat": cat, "main": p_num, "file": filename})
                except: continue
    return sorted(list(provenance_map.keys())), provenance_map

def main():
    if not os.path.exists(REF_POOL_DIR): os.makedirs(REF_POOL_DIR)
    
    status_tracker = {}
    if os.path.exists(STATUS_LOG_FILE):
        try:
            with open(STATUS_LOG_FILE, 'r', encoding='utf-8') as f:
                status_tracker = json.load(f)
        except: pass

    all_target_pids, provenance_db = analyze_provenance()
    to_do_pids = []
    for pid in all_target_pids:
        file_path = os.path.join(REF_POOL_DIR, f"{pid}.json")
        # 检查是否已完成
        if os.path.exists(file_path) and status_tracker.get(pid, {}).get("status") == 200 and is_reference_complete(file_path):
            continue
        to_do_pids.append(pid)

    print(f"📊 待抓取: {len(to_do_pids)} | 并发线程: {MAX_WORKERS}")

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_pid = {executor.submit(process_one_patent, pid, provenance_db, session): pid for pid in to_do_pids}
            pbar = tqdm(as_completed(future_to_pid), total=len(to_do_pids), desc="PatRe Crawler")
            
            for i, future in enumerate(pbar):
                pid, status_info = future.result()
                
                with STATUS_LOCK:
                    status_tracker[pid] = status_info
                    # 每 10 个保存一次状态文件
                    if (i + 1) % 10 == 0:
                        with open(STATUS_LOG_FILE, 'w', encoding='utf-8') as f:
                            json.dump(status_tracker, f, indent=2)

                pbar.set_postfix({"PID": pid, "Status": status_info['status']})
                # 随机休眠，减少被反爬几率
                time.sleep(random.uniform(0.5, 1.5))

    with open(STATUS_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(status_tracker, f, indent=2)
    print("\n✅ 任务处理完成！")

if __name__ == "__main__":
    main()