import pdfplumber
import os

pdf_path = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\Temp_Process\Patent_12337242_App_17976389\OA01_2024-12-09_Office_Action\OA1_References_Cited_2024-12-09.pdf"

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
else:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"Number of pages: {len(pdf.pages)}")
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                print(f"--- Page {i+1} Text Snippet ---")
                print(text[:500] if text else "No text extracted")
                
                tables = page.extract_tables()
                print(f"Tables found: {len(tables)}")
    except Exception as e:
        print(f"Error: {e}")
