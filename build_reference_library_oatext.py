# -*- coding: utf-8 -*-
import os
import json
import time
import random
import requests
import re
from bs4 import BeautifulSoup
from bs4.element import Tag
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ===================== 1. 路径与全局配置 =====================
BASE_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset"
INPUT_JSONL = os.path.join(BASE_DIR, "oa_text_references.jsonl")
REF_POOL_DIR = os.path.join(BASE_DIR, "ReferencePool_Full")
STATUS_LOG_FILE = os.path.join(REF_POOL_DIR, "crawl_status.json")

MAX_WORKERS = 4 
# 确保你的代理地址正确，如果不需要代理请设为 None
PROXIES = {"http": "http://127.0.0.1:8081", "https": "http://127.0.0.1:8081"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
}

# ===================== 2. 核心逻辑 =====================

def standardize_pid(pid):
    if not pid: return ""
    p = re.sub(r'[^a-zA-Z0-9]', '', str(pid)).upper()
    if not p: return ""
    if p[0].isdigit(): p = "US" + p
    return p

def extract_complete_summary(soup):
    """从 BeautifulSoup 对象中提取摘要"""
    def _is_stop_heading(text):
        t = text.lower()
        return any(stop in t for stop in ['background', 'brief', 'detailed', 'claims'])
    
    def _is_noise(text):
        t = text.lower()
        return any(noise in t[:50] for noise in ['claim of priority', 'cross-reference'])

    content_parts = []
    # 尝试定位 Summary 标题
    start_node = soup.find(['heading', 'h2', 'h3'], string=re.compile(r'^SUMMARY$', re.I))
    if not start_node:
        start_node = soup.find(['heading', 'h2', 'h3'], string=re.compile(r'Summary of (the )?Invention', re.I))

    if start_node:
        for sibling in start_node.find_next_siblings():
            sib_tag = sibling.name
            sib_text = sibling.get_text().strip()
            if sib_tag in ['heading', 'h2', 'h3'] and _is_stop_heading(sib_text):
                break
            if 'description-paragraph' in sibling.get('class', []) or sib_tag in ['p', 'div']:
                if sib_text and not _is_noise(sib_text):
                    content_parts.append(sib_text)
        if content_parts:
            return "\n\n".join(content_parts)

    # 备选方案：直接在 description section 查找
    desc_sec = soup.find("section", {"itemprop": "description"})
    if desc_sec:
        headings = desc_sec.find_all(['heading', 'h2', 'h3'])
        for h in headings:
            h_text = h.get_text(" ", strip=True)
            if not re.match(r'^(summary|summary of (the )?invention)\b', h_text, re.I):
                continue
            collected = []
            for el in h.next_elements:
                if el is h: continue
                if isinstance(el, Tag):
                    if el.name in ['heading', 'h2', 'h3']: break
                    if el.name in ['p', 'div'] or 'description-paragraph' in el.get('class', []):
                        txt = el.get_text(" ", strip=True)
                        if txt and not _is_noise(txt):
                            collected.append(txt)
            if collected:
                return "\n\n".join(collected)
    return "Summary Content Not Found"

def extract_patent_content(html_text):
    """解析 Google Patents 页面内容"""
    soup = BeautifulSoup(html_text, "lxml")
    
    # 1. 标题和日期
    title = ""
    t_meta = soup.find("meta", {"name": "DC.title"})
    if t_meta: title = t_meta.get("content")
    
    pub_date = ""
    d_meta = soup.find("meta", {"name": "DC.date", "scheme": "dateSubmitted"}) or \
             soup.find("meta", {"itemprop": "publicationDate"})
    if d_meta: pub_date = d_meta.get("content")

    # 2. 摘要
    abs_tag = soup.find("section", {"itemprop": "abstract"}) or soup.find("div", class_="abstract")
    abstract = abs_tag.get_text(" ", strip=True) if abs_tag else ""

    summary_text = extract_complete_summary(soup)

    # 3. 权利要求 (Claims)
    claims = [div.get_text(" ", strip=True) for div in soup.find_all("div", class_="claim-text") 
              if not div.find_parent("div", class_="claim-text")]

    # 4. 说明书全文 (Description)
    desc_sec = soup.find("section", {"itemprop": "description"})
    full_desc = desc_sec.get_text(" ", strip=True) if desc_sec else ""

    return {
        "title": title,
        "publication_date": pub_date,
        "abstract": abstract,
        "summary_of_invention": summary_text,
        "claims": claims,
        "full_description": full_desc
    }

def process_task(task_info, session):
    """处理单个任务"""
    pid = task_info["std_pid"]
    url = f"https://patents.google.com/patent/{pid}/en"
    
    try:
        resp = session.get(url, headers=HEADERS, proxies=PROXIES, timeout=20)
        if resp.status_code != 200:
            return pid, {"status": resp.status_code, "url": url}

        # 1. 提取网页内容
        content = extract_patent_content(resp.text)
        
        # 2. 合并原始信息与爬取内容
        result_data = {**task_info, **content}

        # 3. 计算数据质量
        abstract = result_data.get("abstract", "")
        summary = result_data.get("summary_of_invention", "")
        claims = result_data.get("claims") or []
        
        if abstract and (summary and summary != "Summary Content Not Found") and claims:
            quality = "OK"
        elif abstract or (summary and summary != "Summary Content Not Found") or claims:
            quality = "Partial"
        else:
            quality = "Missing"

        result_data["data_quality"] = quality
        
        # 4. 填充默认值 (确保字段对齐)
        fields = [
            "application_number", "category", "source_file", "oa_index", 
            "oa_number", "oa_date", "oa_type", "rejection_types", 
            "references_cited", "oa_text_len", "match_kind", "match_raw", 
            "match_norm", "match_count", "context"
        ]
        for field in fields:
            result_data.setdefault(field, None if "index" in field or "count" in field else "")
        
        result_data.setdefault("provenance", [])
        result_data["crawl_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        # 5. 保存单个 JSON
        save_path = os.path.join(REF_POOL_DIR, f"{pid}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        return pid, {"status": 200, "time": result_data["crawl_time"]}

    except Exception as e:
        return pid, {"status": "Error", "msg": str(e)}

# ===================== 3. 主程序 =====================

def main():
    if not os.path.exists(REF_POOL_DIR): os.makedirs(REF_POOL_DIR)

    status_tracker = {}
    if os.path.exists(STATUS_LOG_FILE):
        try:
            with open(STATUS_LOG_FILE, 'r', encoding='utf-8') as f:
                status_tracker = json.load(f)
        except: pass

    tasks = {}
    print(f"🔍 扫描输入文件: {INPUT_JSONL}")
    if not os.path.exists(INPUT_JSONL):
        print(f"❌ 找不到输入文件: {INPUT_JSONL}")
        return

    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                raw_pid = item.get("match_norm") or item.get("patent_number")
                if not raw_pid: continue
                
                std_pid = standardize_pid(raw_pid)
                if std_pid not in tasks:
                    item["std_pid"] = std_pid
                    tasks[std_pid] = item
            except: continue

    to_do_pids = []
    for pid in tasks:
        file_path = os.path.join(REF_POOL_DIR, f"{pid}.json")
        if os.path.exists(file_path) and status_tracker.get(pid, {}).get("status") == 200:
            continue
        to_do_pids.append(pid)

    print(f"📊 发现总专利数: {len(tasks)} | 还需要爬取: {len(to_do_pids)}")

    

    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_pid = {executor.submit(process_task, tasks[pid], session): pid for pid in to_do_pids}
            
            pbar = tqdm(as_completed(future_to_pid), total=len(to_do_pids), desc="PatRe Crawler")
            for i, future in enumerate(pbar):
                pid, res = future.result()
                status_tracker[pid] = res
                pbar.set_postfix({"PID": pid, "Status": res['status']})
                
                if i % 10 == 0:
                    with open(STATUS_LOG_FILE, 'w', encoding='utf-8') as f:
                        json.dump(status_tracker, f, indent=2)
                
                # 适度休眠防止被封
                time.sleep(random.uniform(0.5, 1.5))

    with open(STATUS_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(status_tracker, f, indent=2)
    print("\n✅ 爬取任务结束！")

if __name__ == "__main__":
    main()