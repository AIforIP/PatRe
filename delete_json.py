import os
path = r"D:\study\competition\LLM_based_IP_Review_Research\Project\PatRe\dataset\PatTimeline_A_Category\Temp_Process\cleaned_json_output1\Patent_12337242_App_17976389.json"
if os.path.exists(path):
    os.remove(path)
    print(f"Deleted {path}")
else:
    print(f"Not found {path}")
