import json
import os 
import re
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from tqdm import tqdm
from configs.configure import video_chatgpt_pred_x_funqa_dataset_path,funqa_test_humor_video_dir
from scripts.model_loaders.video_chatgpt_loader import VideoChatGPTLoader
from scripts.data_loaders.fun_qa_loader import FunQA_DataLoader

def save_data(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)
        print(f"{file_name} saved successfully")
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
    funqa_data = FunQA_DataLoader()

    results = {}
    if os.path.exists(video_chatgpt_pred_x_funqa_dataset_path):
        with open(video_chatgpt_pred_x_funqa_dataset_path, "r") as f:
            results = json.load(f)
        print(f"size of results: {len(results)}")
    for video_info in tqdm(funqa_data):
        id = video_info["ID"]
        if id not in results: 
            video_path = funqa_test_humor_video_dir+video_info["visual_input"]
            video_frames = video_chatgpt.get_video_frames(video_path,num_frm=100)
            description = video_info["instruction"]
            query = f"""{description}. provide a start and end time stamp of the moment. Do not provide any null predictions. Format: start_time: mm:ss, end_time: mm:ss."""
            answer = video_chatgpt.infer(video_frames, query)
            video_info["video_chatgpt_pred"] =answer
            pred_start, pred_end = extract_time_from_answer(answer)
            video_info['pred_start'] = pred_start   
            video_info['pred_end'] = pred_end
            results[id] = video_info
            save_data(results, video_chatgpt_pred_x_funqa_dataset_path)
                
    # check the results
    print("verifying results")
    with open(video_chatgpt_pred_x_funqa_dataset_path, "r") as f:
        results = json.load(f)
        print(f"size of results: {len(results)}")
    # check if there is any none value for the start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            none_count += 1
    print(f"None count: {none_count}")
            
