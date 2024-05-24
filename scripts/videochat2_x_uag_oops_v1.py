import json
import torch
import os
import re
from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
# import gradio as gr
# from gradio.themes.utils import colors, fonts, sizes
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/foundation_models/Ask-Anything/video_chat2/")
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from tqdm import tqdm

from configs.configure import results_dir
from data_loaders.uag_oops_v1_loader import UagOopsV1_DataLoader
from model_loaders.videochat2_loader import VideoChat2Loader



def save_data(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)
        print(f"{file_name} saved successfully")
if __name__ == "__main__":
    uag_oops_dataset_v1 = UagOopsV1_DataLoader()
    video_chat2 = VideoChat2Loader()


    results = {}
    # if result file exists
    # if os.path.exists(f"{results_dir}videochat2_pred_uag_oops_dataset.json"):
    #     with open(f"{results_dir}videochat2_pred_uag_oops_dataset.json", "r") as f:
    #         results = json.load(f)
    #     print(f"size of results: {len(results)}")
        
    for video_id,video_info in tqdm(uag_oops_dataset_v1.items()):
        # if video id not in results
        if video_id not in results:
            video_path = video_info["video_path"]
            description = video_info["description"]
            query = f"""Find the start time and end time of the query below from the given video.
            Query: {description} 
            Provide your start and end timestamps in json format.        
            """
            answer =video_chat2.infer(video_path, query)
            # answer = "demo"
            video_info["videochat2_pred"] =answer
            video_info['pred_start']=None
            video_info['pred_end']=None
            pattern_1 = r'(\d{1,2}:\d{2}(:\d{2})?)'
            times = re.findall(pattern_1, answer)
            if len(times) == 2:
                video_info['pred_start'] = times[0][0]
                video_info['pred_end'] = times[1][0]


            results[video_id] = video_info
            # save_data(results, f"{results_dir}videochat2_pred_uag_oops_dataset.json")
    
    # check the results
    print("verifying results")
    with open(f"{results_dir}videochat2_pred_uag_oops_dataset.json", "r") as f:
        results = json.load(f)
    print(f"size of results: {len(results)}") #1589
    # check if there is any none value for start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            # print("video_id: ", video_id)
            none_count +=1
    print(f"none_count: {none_count}") #322

    