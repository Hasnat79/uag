import json
import re
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from tqdm import tqdm
from configs.configure import video_chatgpt_pred_x_uag_oops_dataset_path,video_chatgpt_pred_x_ssbd_result_path, video_chatgpt_pred_x_uag_oops_dataset_null_fixed_path,video_chatgpt_pred_x_ssbd_result_null_fixed_path

def get_uag_oops_dataset(uag_oops_dataset_path):
    with open(uag_oops_dataset_path, "r") as f:
        uag_oops_dataset = json.load(f)
    return uag_oops_dataset



def get_null_value_count(videochat2_pred_uag_oops_dataset):
    none_count = 0
    for video_id,video_info in videochat2_pred_uag_oops_dataset.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            none_count +=1
    return none_count
def save_data(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)
        print(f"{file_name} saved successfully")
def extract_times(sample_text):
    pattern_1 = r'\"start\": \"(\d{2}:\d{2}:\d{2})\",\n\"end\": \"(\d{2}:\d{2}:\d{2})\"'
    pattern_2 = r'\"start_time\": \"(\d{2}:\d{2}:\d{2})\",\n\"end_time\": \"(\d{2}:\d{2}:\d{2})\"'
    pattern_3 =  r'Start Time: (\d{2}:\d{2}:\d{2})\nEnd Time: (\d{2}:\d{2}:\d{2})'
    pattern_4 =  r'Start time: (\d{2}:\d{2}:\d{2})\nEnd time: (\d{2}:\d{2}:\d{2})'
    pattern_5 = r'Start: (\d{2}:\d{2}:\d{2})\nEnd: (\d{2}:\d{2}:\d{2})'
    pattern_6 = r'\"start_time\": \"(\d{2}:\d{2}:\d{2})\",\n  \"end_time\": \"(\d{2}:\d{2}:\d{2})\"'


    match = re.search(pattern_1, sample_text)
    match_2 = re.search(pattern_2, sample_text)
    match_3 = re.search(pattern_3, sample_text)
    match_4 = re.search(pattern_4, sample_text)
    match_5 = re.search(pattern_5, sample_text)
    match_6 = re.search(pattern_6, sample_text)
    if match:
        start_time = match.group(1)
        end_time = match.group(2)
        return start_time, end_time
    elif match_2:
        start_time = match_2.group(1)
        end_time = match_2.group(2)
        return start_time, end_time
    elif match_3:
        start_time = match_3.group(1)
        end_time = match_3.group(2)
        return start_time, end_time
    elif match_4:
        start_time = match_4.group(1)
        end_time = match_4.group(2)
        return start_time, end_time
    elif match_5:
        start_time = match_5.group(1)
        end_time = match_5.group(2)
        return start_time, end_time
    elif match_6:
        start_time = match_6.group(1)
        end_time = match_6.group(2)
        return start_time, end_time
    else:
        return None, None


def run_null_fix_videochat2_pred_uag_oops_dataset():
    videochat2_pred_uag_oops_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/results/videochat2_pred_uag_oops_dataset_v1_null_fixed.json"
    output_path = "/scratch/user/hasnat.md.abdullah/uag/results/videochat2_pred_uag_oops_dataset_null_fixed.json"
    # videochat2_pred_uag_oops_dataset = get_uag_oops_dataset(videochat2_pred_uag_oops_dataset_path)
    videochat2_pred_uag_oops_dataset = load_file(video_chatgpt_pred_x_uag_oops_dataset_path)

    none_count = get_null_value_count(videochat2_pred_uag_oops_dataset)
    print(f"none_count: {none_count}") #322

    for video_id,video_info in tqdm(videochat2_pred_uag_oops_dataset.items()):
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            print("video_id: ", video_id)
            print(f"videochat2_pred: {video_info['videochat2_pred']}")
            pred = video_info['videochat2_pred']
            start, end = extract_times(pred)
            print(f"start: {start}, end: {end}")
            video_info['pred_start'] = start
            video_info['pred_end'] = end
            # break
    save_data(videochat2_pred_uag_oops_dataset, output_path)
    print(f" null values: {get_null_value_count(videochat2_pred_uag_oops_dataset)}")#124
    # total non null values
    print(f"total non null values: {len(videochat2_pred_uag_oops_dataset) - none_count}")#1465

def load_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
def run_null_fix_video_chatgpt_pred_uag_oops_dataset():
    video_chatgpt_pred_x_uag_oops_dataset  = load_file(video_chatgpt_pred_x_uag_oops_dataset_path)
    none_count = get_null_value_count(video_chatgpt_pred_x_uag_oops_dataset)
    print(f"none_count: {none_count}")#274
    for video_id,video_info in tqdm(video_chatgpt_pred_x_uag_oops_dataset.items()):
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            print("video_id: ", video_id)
            print(f"video_chatgpt_pred: {video_info['video_chatgpt_pred']}")
            pred = video_info['video_chatgpt_pred']
            start, end = extract_times(pred)
            print(f"start: {start}, end: {end}")
            video_info['pred_start'] = start
            video_info['pred_end'] = end
            # break
    save_data(video_chatgpt_pred_x_uag_oops_dataset, video_chatgpt_pred_x_uag_oops_dataset_null_fixed_path)
    print(f" null values: {get_null_value_count(video_chatgpt_pred_x_uag_oops_dataset)}")#274
    # total non null values
    print(f"total non null values: {len(video_chatgpt_pred_x_uag_oops_dataset) - none_count}")#1315

def run_null_fix_video_chatgpt_pred_ssbd_dataset():
    video_chatgpt_pred_x_ssbd_result = load_file(video_chatgpt_pred_x_ssbd_result_path)
    none_count = get_null_value_count(video_chatgpt_pred_x_ssbd_result)
    print(f"none_count: {none_count}")#0
    for video_id,video_info in tqdm(video_chatgpt_pred_x_ssbd_result.items()):

        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            print("video_id: ", video_id)
            print(f"video_chatgpt_pred: {video_info['video_chatgpt_pred']}")
            pred = video_info['video_chatgpt_pred']
            start, end = extract_times(pred)
            print(f"start: {start}, end: {end}")
            video_info['pred_start'] = start
            video_info['pred_end'] = end
            
    save_data(video_chatgpt_pred_x_ssbd_result, video_chatgpt_pred_x_ssbd_result_null_fixed_path)
    print(f" null values: {get_null_value_count(video_chatgpt_pred_x_ssbd_result)}")#22
    # total non null values
    print(f"total non null values: {len(video_chatgpt_pred_x_ssbd_result) - none_count}")#82

if __name__ == "__main__":
    # done
    # run_null_fix_videochat2_pred_uag_oops_dataset()
    # done
    # run_null_fix_video_chatgpt_pred_uag_oops_dataset()
    # currently doing
    run_null_fix_video_chatgpt_pred_ssbd_dataset()


