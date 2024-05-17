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
import argparse
from tqdm import tqdm
from configs.configure import ssbd_data_path,llava_lightning_7b_v1_1_path, video_chatgpt_weights_path, results_dir
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=8)
    parser.add_argument("--model-list-mode", type=str, default="once", choices=["once", "reload"])
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    parser.add_argument("--embed", action="store_true")
    parser.add_argument("--model-name", type=str, default=llava_lightning_7b_v1_1_path)
    parser.add_argument("--vision_tower_name", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--conv-mode", type=str, default="video-chatgpt_v1")
    parser.add_argument("--projection_path", type=str, required=False, default=video_chatgpt_weights_path)

    args = parser.parse_args()
    return args
@memory.cache
def init_model(model_name, projection_path):
    model, vision_tower, tokenizer, image_processor, video_token_len = \
        initialize_model(model_name, projection_path)
    print("Model initialized")
    return model, vision_tower, tokenizer, image_processor, video_token_len
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
    with open(ssbd_data_path, 'r') as f:
        ssbd_data = json.load(f)
    print(f"SSBD data loaded with {len(ssbd_data)} videos")

    #args setup
    args = parse_args()
    conv_mode = args.conv_mode

    # initialize the model
    model, vision_tower, tokenizer, image_processor, video_token_len = init_model(args.model_name, args.projection_path)
    results = {}
    ##TODO: if result file exists then load the file and update the results
    if os.path.exists(f"{results_dir}video_chatgpt_pred_ssbd_dataset.json"):
        with open(f"{results_dir}video_chatgpt_pred_ssbd_dataset.json", "r") as f:
            results = json.load(f)
        print(f"Results loaded successfully from {results_dir}video_chatgpt_pred_ssbd_dataset.json")
    for sample in tqdm(ssbd_data):
        print(f"sample: {sample}")
        video_tag = sample[0]
        behaviour_id = sample[2]['id']
        video_id = video_tag+'_'+str(behaviour_id)
        video_info = {}
        video_info = {'video_path': sample[1], 'behaviour': sample[2]}
        if video_id not in results:
            video_path = sample[1]
            video_frames = load_video(video_path, num_frm =100)
            category = sample[2]['category']
            intensity = sample[2]['intensity']
            query = f"""Find the start time and end time of the query below from the given video.
            Query: A person is {category} with {intensity} intensity.
            Instruction: Provide your start and end timestamps in json format.
            """
            video_info["query"] = query
            try:
                answer = video_chatgpt_infer(video_frames, query, conv_mode, model, vision_tower, tokenizer, image_processor, video_token_len)
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


