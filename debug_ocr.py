import pytesseract
from pdf2image import convert_from_path
import os

pytesseract.pytesseract.tesseract_cmd = r"D:\software\tesseract\tesseract.exe"
poppler_path = r"D:\software\poppler-25.12.0\Library\bin"
pdf_path = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\Temp_Process\Patent_12337242_App_17976389\OA01_2024-12-09_Office_Action\OA1_References_Cited_2024-12-09.pdf"

if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
else:
    try:
        images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
        print(f"Converted {len(images)} pages")
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img.convert('L'), lang='eng')
            print(f"--- Page {i+1} OCR ---")
            print(text[:500])
            break # Just first page
    except Exception as e:
        print(f"Error: {e}")
