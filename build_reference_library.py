# -*- coding: utf-8 -*-
import os
import json
import time
import random
import requests
import re
from bs4 import BeautifulSoup
from tqdm import tqdm  # 用于显示进度条

# ===================== 1. 路径与全局配置 =====================
BASE_DATA_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset"
TARGET_CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
REF_POOL_DIR = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\ReferencePool_Full"
STATUS_LOG_FILE = os.path.join(REF_POOL_DIR, "crawl_status.json")

# 代理配置 (请确保 Clash 开启 8081 端口)
PROXIES = {"http": "http://127.0.0.1:8081", "https": "http://127.0.0.1:8081"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

# ===================== 2. 增强型解析工具函数 =====================

def standardize_pid(pid):
    """标准化专利号格式"""
    p = re.sub(r'[^a-zA-Z0-9]', '', str(pid)).upper()
    if not p.startswith("US"): p = "US" + p
    return p

def extract_patent_metadata(soup):
    """提取 Title 和 Publication Date (多重回退机制)"""
    title = ""
    # 尝试 Meta 标签
    t_meta = soup.find("meta", {"name": "DC.title"})
    if t_meta: title = t_meta.get("content")
    if not title:
        t_tag = soup.find("h1", {"itemprop": "name"})
        title = t_tag.get_text(strip=True) if t_tag else ""
    
    date = ""
    # 尝试多种日期 Meta 和 Span
    d_meta = soup.find("meta", {"name": "DC.date", "scheme": "dateSubmitted"}) or \
             soup.find("meta", {"itemprop": "publicationDate"})
    if d_meta: 
        date = d_meta.get("content")
    if not date:
        d_span = soup.find("span", {"itemprop": "publicationDate"})
        date = d_span.get_text(strip=True) if d_span else ""
    
    return title, date

def extract_complete_summary(soup):
    content_parts = []
    # 定位 Summary 起始点
    start_node = soup.find(['heading', 'h2', 'h3'], text=re.compile(r'^SUMMARY$', re.I))
    if not start_node:
        start_node = soup.find(['heading', 'h2', 'h3'], text=re.compile(r'Summary of (the )?Invention', re.I))

    if start_node:
        for sibling in start_node.find_next_siblings():
            sib_tag = sibling.name
            sib_text = sibling.get_text().strip()
            sib_text_lower = sib_text.lower()

            # 1. 强终止条件：遇到标准标题标签
            if sib_tag in ['heading', 'h2', 'h3'] and \
               any(stop in sib_text_lower for stop in ['background', 'brief', 'detailed', 'claims']):
                break
            
            # 2. 强终止条件：遇到段落内包含的伪标题（针对你刚才出现的问题）
            # 检查段落是否以 "BRIEF DESCRIPTION" 或 "DETAILED DESCRIPTION" 开头
            if any(sib_text_lower.startswith(stop) for stop in ['brief description', 'detailed description', 'claims']):
                break

            # 3. 过滤噪音 div
            if sibling.get('class') and any(c in sibling.get('class') for c in ['matrix', 'table', 'knowledge-card']):
                continue

            # 4. 收集内容
            if 'description-paragraph' in sibling.get('class', []) or sib_tag in ['p', 'div']:
                if sib_text and not any(noise in sib_text_lower[:50] for noise in ['claim of priority', 'cross-reference']):
                    content_parts.append(sib_text)
    
    return "\n\n".join(content_parts) if content_parts else "Summary Content Not Found"
# ===================== 3. 数据溯源与状态管理 =====================

def analyze_provenance():
    """扫描 A-H 全类数据，建立引文 -> 来源文件的溯源关系"""
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
                                if std_p not in provenance_map: provenance_map[std_p] = []
                                provenance_map[std_p].append({"cat": cat, "main": p_num, "file": filename})
                except: continue
    return sorted(list(provenance_map.keys())), provenance_map

# ===================== 4. 主爬取任务 =====================

def main():
    if not os.path.exists(REF_POOL_DIR): os.makedirs(REF_POOL_DIR)
    
    # 加载断点状态
    status_tracker = {}
    if os.path.exists(STATUS_LOG_FILE):
        try:
            with open(STATUS_LOG_FILE, 'r', encoding='utf-8') as f:
                status_tracker = json.load(f)
        except: status_tracker = {}

    # 1. 扫描需要爬取的 PID
    all_target_pids, provenance_db = analyze_provenance()
    
    # 2. 过滤已完成的（断点续传逻辑）
    to_do_pids = []
    for pid in all_target_pids:
        file_path = os.path.join(REF_POOL_DIR, f"{pid}.json")
        # 如果文件存在 且 账本记录状态为 200，则视为已完成
        if os.path.exists(file_path) and status_tracker.get(pid, {}).get("status") == 200:
            continue
        to_do_pids.append(pid)

    print(f"📊 总需引文: {len(all_target_pids)} | 已存在: {len(all_target_pids)-len(to_do_pids)} | 待抓取: {len(to_do_pids)}")

    # 3. 启动进度条爬取
    pbar = tqdm(to_do_pids, desc="PatRe Crawler", unit="patent")
    
    for pid in pbar:
        pbar.set_postfix({"Current": pid})
        url = f"https://patents.google.com/patent/{pid}/en"
        
        try:
            resp = requests.get(url, headers=HEADERS, proxies=PROXIES, timeout=30)
            if resp.status_code != 200:
                status_tracker[pid] = {"status": resp.status_code, "url": url}
                continue
            
            soup = BeautifulSoup(resp.text, "lxml")
            title, pub_date = extract_patent_metadata(soup)
            summary = extract_complete_summary(soup)
            
            # 抓取全文和权利要求
            desc_sec = soup.find("section", {"itemprop": "description"})
            full_desc = desc_sec.get_text(" ", strip=True) if desc_sec else ""
            claims = [div.get_text(" ", strip=True) for div in soup.find_all("div", class_="claim-text") 
                      if not div.find_parent("div", class_="claim-text")]

            # 存储结构化数据
            result_data = {
                "patent_number": pid,
                "title": title,
                "publication_date": pub_date,
                "abstract": soup.find("section", {"itemprop": "abstract"}).get_text(" ", strip=True) if soup.find("section", {"itemprop": "abstract"}) else "",
                "summary_of_invention": summary,
                "full_description": full_desc,
                "claims": claims,
                "provenance": provenance_db[pid]
            }

            with open(os.path.join(REF_POOL_DIR, f"{pid}.json"), 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)

            # 更新账本
            status_tracker[pid] = {
                "status": 200,
                "crawl_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stats": {"desc_len": len(full_desc), "summary_len": len(summary), "claims_count": len(claims)}
            }

        except Exception as e:
            status_tracker[pid] = {"status": "Error", "msg": str(e)}
        
        # 频率限制与实时存档
        time.sleep(random.uniform(5, 8))
        if len(status_tracker) % 5 == 0:
            with open(STATUS_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(status_tracker, f, indent=2)

    # 最终保存
    with open(STATUS_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(status_tracker, f, indent=2)
    print("\n✅ 所有任务已完成！引文库已就绪。")

if __name__ == "__main__":
    main()