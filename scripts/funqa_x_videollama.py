import json
import torch
import os
import re
from joblib import Memory
from tqdm import tqdm
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/foundation_models/Video-LLaMA")
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import videollama_pred_x_funqa_dataset_path,funqa_test_humor_video_dir
from scripts.model_loaders.video_llama2_loader import VideoLlama2Loader
from scripts.data_loaders.fun_qa_loader import FunQA_DataLoader

def load_file (file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
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
       pred_start = times[0][0]
       pred_end = times[1][0]
    elif match2:
      pred_start = float(match2.group(1))  # Extract and convert start_time to float
      pred_end = float(match2.group(2)) 
    else: 
        pred_start = None
        pred_end = None
    
    return pred_start, pred_end
if __name__ == "__main__":
    video_llama2 = VideoLlama2Loader()
    funqa_data = FunQA_DataLoader()
   

    result = {}
    if os.path.exists(videollama_pred_x_funqa_dataset_path):
        with open(videollama_pred_x_funqa_dataset_path, "r") as f:
            result = json.load(f)
        print(f"Loaded {len(result)} results")
    for video_info in tqdm(funqa_data):
        id = video_info["ID"]
        if id not in result:
            video_path = funqa_test_humor_video_dir+video_info["visual_input"]
            gr_img = None
            description = video_info["instruction"]
            # text_input = f"""Find the start time and end time of the query below from the video.
            text_input = f"""Find the start time and end time of the query below from the video.
            Query: {description}"""
            # Query: {description}"""
            audio_flag = True
            num_beams = 1  # Replace with the number of beams for decoding
            temperature = 0.5  # Replace with the temperature for decoding
            try:
                answer = video_llama2.infer(video_path, gr_img, text_input, audio_flag, num_beams, temperature)
                video_info["videollama_pred"] = answer
                pred_start, pred_end = extract_time_from_answer(answer)
                print(f"Predicted start: {pred_start}, Predicted end: {pred_end}")
                video_info['pred_start'] = pred_start
                video_info['pred_end'] = pred_end
                result[id] = video_info
                
            except Exception as e:
                print(e)
            with open(videollama_pred_x_funqa_dataset_path, "w") as f:
                json.dump(result, f, indent=4)
                print(f"Results saved successfully in {videollama_pred_x_funqa_dataset_path}")
    # check the results
    print("verifying results")
    with open(videollama_pred_x_funqa_dataset_path, "r") as f:
        results = json.load(f)
        print(f"size of results: {len(results)}") #1589
    # check if there is any none value for the start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            # print("video_id: ", video_id)
            none_count +=1
    print(f"none_count: {none_count}") #430
