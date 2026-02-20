import os
print("Script started")
pdf_path = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\Temp_Process\Patent_12337242_App_17976389\OA01_2024-12-09_Office_Action\OA1_References_Cited_2024-12-09.pdf"
print(f"Checking: {pdf_path}")
print(f"Exists: {os.path.exists(pdf_path)}")
print(f"Size: {os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 'N/A'}")
