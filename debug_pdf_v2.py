import pdfplumber
import os

pdf_path = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\Temp_Process\Patent_12337242_App_17976389\OA01_2024-12-09_Office_Action\OA1_References_Cited_2024-12-09.pdf"
output_file = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\pdf_debug_output.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    if not os.path.exists(pdf_path):
        f.write(f"File not found: {pdf_path}\n")
    else:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                f.write(f"Number of pages: {len(pdf.pages)}\n")
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    f.write(f"--- Page {i+1} Text Snippet ---\n")
                    f.write((text[:500] if text else "No text extracted") + "\n")
                    
                    tables = page.extract_tables()
                    f.write(f"Tables found: {len(tables)}\n")
                    if tables:
                        for j, table in enumerate(tables):
                            f.write(f"Table {j+1} rows: {len(table)}\n")
        except Exception as e:
            f.write(f"Error: {e}\n")
