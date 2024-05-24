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
from configs.configure import video_llama2_pred_x_ssbd_result_path
from data_loaders.ssbd_loader import Ssbd_DataLoader
from model_loaders.video_llama2_loader import VideoLlama2Loader



def load_file (file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
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
    video_llama2 = VideoLlama2Loader()


    results = {}
    if os.path.exists(video_llama2_pred_x_ssbd_result_path):
        with open(video_llama2_pred_x_ssbd_result_path, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results")
    for video_id,video_info in tqdm(ssbd_data):
        
        if video_id not in results:
            video_path = video_info['video_path']
            category =video_info['behaviour']['category']
            intensity =video_info['behaviour']['intensity']
            query = f"""Find the start time and end time of the query below from the given video.
            Query: A person is {category} with {intensity} intensity.
            """
            video_info["query"] = query
            audio_flag = True
            num_beams = 1  # Replace with the number of beams for decoding
            temperature = 0.5  # Replace with the temperature for decoding
            text_input = query
            gr_img = None
            try:
                answer = video_llama2.infer(video_path, gr_img, text_input, audio_flag, num_beams, temperature)
                print(answer)
                video_info["video_llama2_pred"] = answer
                pred_start, pred_end = extract_time_from_answer(answer)
                print(f"Predicted start: {pred_start}, Predicted end: {pred_end}")
                video_info['pred_start'] = pred_start
                video_info['pred_end'] = pred_end
                results[video_id] = video_info
            except Exception as e:
                print(e)
            # save results
            with open(video_llama2_pred_x_ssbd_result_path, "w") as f:
                json.dump(results, f, indent=4)
                print(f"Results saved successfully in {video_llama2_pred_x_ssbd_result_path}")
        
    # check the results
    print("verifying results")
    with open(video_llama2_pred_x_ssbd_result_path, "r") as f:
        results = json.load(f)
        print(f"size of results: {len(results)}") #1589
    # check if there is any none value for the start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            # print("video_id: ", video_id)
            none_count +=1
    print(f"none_count: {none_count}")#70








    # # try for a single video 
    # gr_video ="/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/oops_video/val/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0.mp4"  # Replace with your video input
    # gr_img = None  # Replace with your image input
    # description ="A guy jumps onto a bed where his son is. When the guy jumps, the son flies up and hits the wall."
    # text_input =f"""Find the start time and end time of the query below from the given video.
    #         Query: {description}     
    #         """  # Replace with your text input
    # audio_flag = True  # Replace with True if your video has audio, False otherwise
    # num_beams = 1  # Replace with the number of beams for decoding
    # temperature = 0.5  # Replace with the temperature for decoding
    
    # chatbot = main(gr_video, gr_img, text_input, audio_flag, num_beams, temperature)
    
    # for message, response in chatbot:
    #     print("User: ", message)
    #     print("Chatbot: ", response)
    #     print()
