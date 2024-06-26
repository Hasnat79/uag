import json
import os
import re
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/foundation_models/Ask-Anything/video_chat2/")
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from tqdm import tqdm

from configs.configure import videochat2_pred_x_funqa_dataset_path,funqa_test_humor_video_dir
from data_loaders.fun_qa_loader import FunQA_DataLoader
from model_loaders.videochat2_loader import VideoChat2Loader
def save_data(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)
        print(f"{file_name} saved successfully")
def extract_time_from_answer(answer):
    pattern_1 = r'(\d{1,2}:\d{2}(:\d{2})?)'
    pattern_2 = r"start_time:\s*(\d+\.\d+),\s*end_time:\s*(\d+\.\d+)"
    times = re.findall(pattern_1, answer)
    match2 = re.search(pattern_2, answer)
    # times = re.findall(pattern_1, answer)
    # if len(times) == 2:
    #     pred_start = times[0][0]
    #     pred_end = times[1][0]
    if len(times) == 2:
      pred_start= times[0][0]
      pred_end = times[1][0]
    elif match2:
        pred_start = float(match2.group(1))  # Extract and convert start_time to float
        pred_end = float(match2.group(2)) 
    else: 
        pred_start = None
        pred_end = None

    return pred_start, pred_end

if __name__ == "__main__":
    funqa_data = FunQA_DataLoader()
    video_chat2 = VideoChat2Loader()

    results = {}
    if os.path.exists(videochat2_pred_x_funqa_dataset_path):
        with open(videochat2_pred_x_funqa_dataset_path, "r") as f:
            results = json.load(f)
        print(f"size of results: {len(results)}")
    
    for video_info in tqdm(funqa_data):
        id = video_info["ID"]
        if id not in results:
            video_path = funqa_test_humor_video_dir+video_info["visual_input"]
            description = video_info["instruction"]
            query = f"""Find the start time and end time of the query below from the video.
                Query: {description}"""
            answer =video_chat2.infer(video_path, query)
            video_info["videochat2_pred"] =answer
            pred_start, pred_end = extract_time_from_answer(answer)
            video_info['pred_start'] = pred_start
            video_info['pred_end'] = pred_end
            results[id] = video_info
            save_data(results, videochat2_pred_x_funqa_dataset_path)
    # check the results
    print("verifying results")
    with open(videochat2_pred_x_funqa_dataset_path, "r") as f:
        results = json.load(f)
        print(f"size of results: {len(results)}")
    # check if there is any none value for start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            none_count +=1
    print(f"none_count: {none_count}")
    
