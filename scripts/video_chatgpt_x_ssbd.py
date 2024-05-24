import sys
import json
import re
import os

from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/foundation_models/Video-ChatGPT/")

from tqdm import tqdm
from configs.configure import results_dir
from data_loaders.ssbd_loader import Ssbd_DataLoader
from model_loaders.video_chatgpt_loader import VideoChatGPTLoader


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
    ssbd_data = Ssbd_DataLoader()
    video_chatgpt = VideoChatGPTLoader()

    results = {}
    if os.path.exists(f"{results_dir}video_chatgpt_pred_ssbd_dataset.json"):
        with open(f"{results_dir}video_chatgpt_pred_ssbd_dataset.json", "r") as f:
            results = json.load(f)
            print(f"Results loaded successfully from {results_dir}video_chatgpt_pred_ssbd_dataset.json")
    for video_id,video_info in tqdm(ssbd_data):
        
        if video_id not in results:
            video_path = video_info["video_path"]
            video_frames = video_chatgpt.get_video_frames(video_path,num_frm=100)

            category =video_info['behaviour']['category']
            intensity =video_info['behaviour']['intensity']
            
            query = f"""Find the start time and end time of the query below from the given video.
            Query: A person is {category} with {intensity} intensity.
            Instruction: Provide your start and end timestamps in json format.
            """
            video_info["query"] = query
            try:
                answer =video_chatgpt.infer(video_frames, query)
                video_info["video_chatgpt_pred"] = answer
                pred_start, pred_end = extract_time_from_answer(answer)
                video_info['pred_start'] = pred_start
                video_info['pred_end'] = pred_end

                results[video_id] = video_info
            except Exception as e:
                print(e)
            with open(f"{results_dir}   video_chatgpt_pred_ssbd_dataset.json", "w") as f:
                json.dump(results, f, indent=4)
                print(f"Results saved successfully in{results_dir}video_chatgpt_pred_ssbd_dataset.json")


    # check the results
    print("verifying results")
    with open(f"{results_dir}video_chatgpt_pred_ssbd_dataset.json", "r") as f:
        results = json.load(f)
    print(f"size of results: {len(results)}") #1589
    # check if there is any none value for start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            # print("video_id: ", video_id)
            none_count +=1
    print(f"None count: {none_count}")#22


