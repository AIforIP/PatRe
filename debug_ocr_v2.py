import pytesseract
from pdf2image import convert_from_path
import os
import gc

pytesseract.pytesseract.tesseract_cmd = r"D:\software\tesseract\tesseract.exe"
poppler_path = r"D:\software\poppler-25.12.0\Library\bin"
pdf_path = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\Temp_Process\Patent_12337242_App_17976389\OA01_2024-12-09_Office_Action\OA1_References_Cited_2024-12-09.pdf"
output_file = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\ocr_raw_output.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    if not os.path.exists(pdf_path):
        f.write(f"File not found: {pdf_path}\n")
    else:
        try:
            f.write(f"PDF exists, size: {os.path.getsize(pdf_path)}\n")
            images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
            f.write(f"Converted {len(images)} pages\n")
            for i, img in enumerate(images):
                f.write(f"--- Page {i+1} OCR Start ---\n")
                text = pytesseract.image_to_string(img.convert('L'), lang='eng')
                f.write(text)
                f.write(f"\n--- Page {i+1} OCR End ---\n")
                img.close()
            gc.collect()
            f.write("\nOCR process completed.\n")
        except Exception as e:
            f.write(f"Error: {e}\n")
