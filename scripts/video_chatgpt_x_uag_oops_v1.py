import json
import os 
import re
from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import results_dir
from scripts.model_loaders.video_chatgpt_loader import VideoChatGPTLoader
from scripts.data_loaders.uag_oops_v1_loader import UagOopsV1_DataLoader

import os

from tqdm import tqdm

def extract_time_from_answer(answer):
    pattern_1 = r'(\d{1,2}:\d{2}(:\d{2})?)'
    times = re.findall(pattern_1, answer)
    if len(times) == 2:
        pred_start = times[0][0]
        pred_end = times[1][0]
    else: 
        pred_start = None
        pred_end = None
    return pred_start, pred_end
if __name__ == "__main__":
    video_chatgpt= VideoChatGPTLoader()
    uag_oops_v1 = UagOopsV1_DataLoader()
    print(f"Total videos in UAG OOPS V1: {len(uag_oops_v1)}")

    results = {}
    ##: if result file exists then load the file and update the results
    if os.path.exists(f"{results_dir}video_chatgpt_pred_uag_oops_dataset_v1.json"):
        with open(f"{results_dir}video_chatgpt_pred_uag_oops_dataset_v1.json", "r") as f:
            results = json.load(f)
        print(f"size of results: {len(results)}")
    for video_id, video_info in tqdm(uag_oops_v1):
        if video_id not in results: 
            video_path = video_info["video_path"]
            video_frames = video_chatgpt.get_video_frames(video_path,num_frm=100)
            # print(len(video_frames))

            description = video_info["description"]
            query = f"""Find the start time and end time of the query below from the video.         
            Query: {description}
            Instruction: Provide your start and end timestamps in json format."""
            video_info["query"] = query
            try:
                answer = video_chatgpt.infer(video_frames, query)
                video_info["video_chatgpt_pred"] = answer
                pred_start, pred_end = extract_time_from_answer(answer)
                video_info['pred_start'] = pred_start   
                video_info['pred_end'] = pred_end
                
                results[video_id] = video_info
            except Exception as e:
                print(e)
            
            with open(f"{results_dir}video_chatgpt_pred_uag_oops_dataset_v1.json", "w") as f:
                json.dump(results, f, indent=4)
                print(f"Results saved successfully in {results_dir}video_chatgpt_pred_uag_oops_dataset_v1.json")
    

    # print(results)
    # check the results
    print("verifying results")
    with open(f"{results_dir}video_chatgpt_pred_uag_oops_dataset_v1.json", "r") as f:
        results = json.load(f)
    print(f"size of results: {len(results)}") #1589
    # check if there is any none value for start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            # print("video_id: ", video_id)
            none_count +=1
    print(f"none_count: {none_count}")
            


    