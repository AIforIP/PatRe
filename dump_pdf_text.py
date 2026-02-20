import pdfplumber
import os

pdf_path = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\Temp_Process\Patent_12337242_App_17976389\OA01_2024-12-09_Office_Action\OA1_References_Cited_2024-12-09.pdf"
output_file = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\pdf_raw_text.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    if not os.path.exists(pdf_path):
        f.write(f"File not found: {pdf_path}\n")
    else:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    f.write(f"--- Page {i+1} ---\n")
                    f.write(page.extract_text() or "[No text]")
                    f.write("\n")
        except Exception as e:
            f.write(f"Error: {e}\n")
