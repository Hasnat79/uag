import json
import sys
import os 

sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")

from configs.configure import uag_oops_dataset_path


paths = [
"/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_null_fixed.json",
"/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_videollama2_text_rep_x_uag_oops_dataset_v1_null_fixed.json",
"/scratch/user/hasnat.md.abdullah/uag/results/video_chatgpt_pred_uag_oops_dataset_v1_null_fixed.json",
"/scratch/user/hasnat.md.abdullah/uag/results/video_llama2_pred_uag_oops_dataset_v1_null_fixed.json",
"/scratch/user/hasnat.md.abdullah/uag/results/videochat2_pred_uag_oops_dataset_v1_null_fixed.json",


]
def fix_start_time(result, uag_oops_dataset_start_fixed):
    for video_id, video_info in result.items():
        if video_id in uag_oops_dataset_start_fixed:
          video_info['start_time'] = uag_oops_dataset_start_fixed[video_id]['start_time']
        else :
          print(f"video_id {video_id} not found in uag_oops_dataset_start_fixed")
    return result
def save_result(path, result):
    with open(path, "w") as f:
        json.dump(result, f, indent=4)
def load_result(path):
    with open(path) as f:
        return json.load(f)
if __name__ == "__main__":
    with open(uag_oops_dataset_path) as f:
        uag_oops_dataset_start_fixed = json.load(f)
    for path in paths:
        print(f"path {path}")
        result = load_result(path)
        result = fix_start_time(result, uag_oops_dataset_start_fixed)
        save_result(path, result)

