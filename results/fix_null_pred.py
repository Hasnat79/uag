import json
import re
import sys
import os
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from tqdm import tqdm
from configs.configure import video_chatgpt_pred_x_uag_oops_dataset_path,video_chatgpt_pred_x_ssbd_result_path, video_chatgpt_pred_x_uag_oops_dataset_null_fixed_path,video_chatgpt_pred_x_ssbd_result_null_fixed_path,video_llama2_pred_x_uag_oops_dataset_path,video_llama2_pred_x_uag_oops_dataset_null_fixed_path,video_llama2_pred_x_ssbd_result_path,video_llama2_pred_x_ssbd_result_null_fixed_path,llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path,llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path_null_fixed,llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path,llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path_null_fixed


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

def extract_time_ranges(pred):
    # Define regular expression patterns to match time ranges
    time_range_pattern = r'(\d+:\d+) - (\d+:\d+)'
    time_range_pattern_1 = r'(\d+:\d+)-(\d+:\d+)'
    time_range_pattern_2 = r"(\d+:\d+:\d+).*?(\d+:\d+:\d+)"
    

    # Find all matches of time ranges in the input text
    matches = re.findall(time_range_pattern, pred)
    matches_1 = re.findall(time_range_pattern_1, pred)
    matches_2 = re.findall(time_range_pattern_2, pred)
    # Extract start and end times from matches
    time_ranges = []
    if len(matches)== 2:
        for match in matches:
            start_time = match[0]
            end_time = match[1]
            time_ranges.append((start_time, end_time))
        return time_ranges
    elif len(matches_1) ==2:
        for match in matches_1:
            start_time = match[0]
            end_time = match[1]
            time_ranges.append((start_time, end_time))
        return time_ranges
    elif len(matches) >= 2:
        for match in matches:
            start_time = match[0]
            end_time = match[1]
            time_ranges.append((start_time, end_time))
        return time_ranges
    elif len(matches_1) >= 2:
        for match in matches_1:
            start_time = match[0]
            end_time = match[1]
            time_ranges.append((start_time, end_time))
        return time_ranges
    # elif len(matches_2) >= 2:
    #     print(matches_2)
    #     exit()
    #     for match in matches_2:
    #         start_time = match[0]
    #         end_time = match[1]
    #         time_ranges.append((start_time, end_time))
    #     return time_ranges
    else :
       return None

    
def calculate_middle_time(time_range):
    # Parse start and end times into numerical values
    start_hours, start_minutes = map(int, time_range[0].split(':'))
    end_hours, end_minutes = map(int, time_range[1].split(':'))

    # Calculate the middle point
    middle_hours = (start_hours + end_hours) // 2
    middle_minutes = (start_minutes + end_minutes) // 2

    # Convert the middle point back into time stamp format
    middle_time_stamp = '{:02d}:{:02d}'.format(middle_hours, middle_minutes)

    return middle_time_stamp

def extract_time_for_videollama2_preds(pred):

    time_ranges = extract_time_ranges(pred)
    print(f"time_ranges: {time_ranges}")
    if time_ranges!=None and len(time_ranges)==2:
        middle_times =[]
        for time_range in time_ranges:
            middle_time = calculate_middle_time(time_range)
            middle_times.append(middle_time)
        pred_start, pred_end = middle_times
        print(f"pred_start: {pred_start}, pred_end: {pred_end}")
        return pred_start, pred_end
    else: 
        return None, None
    

def run_null_fix_video_llama2_pred_uag_oops_dataset():
    # video_llama2_pred_x_uag_oops_dataset = load_file(video_llama2_pred_x_uag_oops_dataset_path)
    video_llama2_pred_x_uag_oops_dataset = load_file(video_llama2_pred_x_uag_oops_dataset_null_fixed_path)
    none_count = get_null_value_count(video_llama2_pred_x_uag_oops_dataset)
    print(f"none_count: {none_count}")#0
    for video_id,video_info in tqdm(video_llama2_pred_x_uag_oops_dataset.items()):
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            print("video_id: ", video_id)
            print(f"video_llama2_pred: {video_info['video_llama2_pred']}")
            pred = video_info['video_llama2_pred']

            start, end = extract_time_for_videollama2_preds(pred)
            print(f"start: {start}, end: {end}\n\n")
            video_info['pred_start'] = start
            video_info['pred_end'] = end
            
    save_data(video_llama2_pred_x_uag_oops_dataset, video_llama2_pred_x_uag_oops_dataset_null_fixed_path)
    print(f" null values: {get_null_value_count(video_llama2_pred_x_uag_oops_dataset)}")#42 // 15
    # total non null values
    print(f"total non null values: {len(video_llama2_pred_x_uag_oops_dataset) - none_count}")#1547 // 1574
def run_null_fix_video_llama2_pred_ssbd_dataset():
    # video_llama2_pred_x_ssbd_result = load_file(video_llama2_pred_x_ssbd_result_path)
    video_llama2_pred_x_ssbd_result = load_file(video_llama2_pred_x_ssbd_result_null_fixed_path)
    none_count = get_null_value_count(video_llama2_pred_x_ssbd_result)
    print(f"none_count: {none_count}")#0
    for video_id,video_info in tqdm(video_llama2_pred_x_ssbd_result.items()):
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            # print("video_id: ", video_id)
            # print(f"video_llama2_pred: {video_info['video_llama2_pred']}")
            pred = video_info['video_llama2_pred']

            start, end = extract_time_for_videollama2_preds(pred)
            # print(f"start: {start}, end: {end}\n\n")
            video_info['pred_start'] = start
            video_info['pred_end'] = end
            
            
    # save_data(video_llama2_pred_x_ssbd_result, video_llama2_pred_x_ssbd_result_null_fixed_path)
    print(f" null values: {get_null_value_count(video_llama2_pred_x_ssbd_result)}")#3 // 0
    print(f"total non null values: {len(video_llama2_pred_x_ssbd_result) - none_count}")#101 // 104
def extract_time_from_llama3(pred):
    pattern = r'"startTime": (\d+\.\d+),\s+"endTime": (\d+\.\d+)'
    pattern_2 = r'"start_time": (\d+\.?\d*),\s*"end_time": (\d+\.?\d*)'
    pattern_3 = r'Start time: (\d+\.?\d*)s\s+End time: (\d+\.?\d*)s'

    match = re.search(pattern, pred)
    match_2 = re.search(pattern_2, pred)
    match_3 = re.search(pattern_3, pred)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        return start_time, end_time
    elif match_2:
        start_time = float(match_2.group(1))
        end_time = float(match_2.group(2))
        return start_time, end_time   
    elif match_3:
        start_time = float(match_3.group(1))
        end_time = float(match_3.group(2))
        return start_time, end_time
    return None, None
def run_null_fix_uag_oops(model_name,result_data_path="",output_path = "",time_extract_model=""):

    print(f"===================={model_name} null fix====================")
    if os.path.exists(output_path):
        result_data = load_file(output_path)
        print("ff")
    else:
        result_data = load_file(result_data_path)
    none_count = get_null_value_count(result_data)
    print(f"model_name: {model_name}, none_count: {none_count}")
    for video_id,video_info in tqdm(result_data.items()):
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            # print("video_id: ", video_id)

            # print(f"""{model_name}_pred: {video_info[f"{model_name.split('_')[0]}_pred"]}""")
            pred = video_info[f"""{model_name.split('_')[0]}_pred"""]
            # print(f"pred: {pred}")
            if time_extract_model=="llama3":
                start, end = extract_time_from_llama3(pred)
            # start, end = extract_time_for_videollama2_preds(pred)
            # print(f"start: {start}, end: {end}\n\n")
            video_info['pred_start'] = start
            video_info['pred_end'] = end

            # temp = input("press enter")
    # save_data(result_data, output_path)
    print(f"null values: {get_null_value_count(result_data)}")#77
    # total non null values
    print(f"total non null values: {len(result_data) - none_count}")#1512
def run_null_fix_ssbd(model_name,result_data_path="",output_path = "",time_extract_model=""):
    print(f"===================={model_name} null fix====================")
    if os.path.exists(output_path):
        result_data = load_file(output_path)
        print("ff")
    else:
        result_data = load_file(result_data_path)
    none_count = get_null_value_count(result_data)
    print(f"model_name: {model_name}, none_count: {none_count}")
    for video_id,video_info in tqdm(result_data.items()):
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            # print("video_id: ", video_id)

            # print(f"""{model_name}_pred: {video_info[f"{model_name.split('_')[0]}_pred"]}""")
            pred = video_info[f"""{model_name.split('_')[0]}_pred"""]
            # print(f"pred: {pred}")
            if time_extract_model=="llama3":
                start, end = extract_time_from_llama3(pred)
            # start, end = extract_time_for_videollama2_preds(pred)
            # print(f"start: {start}, end: {end}\n\n")
            video_info['pred_start'] = start
            video_info['pred_end'] = end

            # temp = input("press enter")
    save_data(result_data, output_path)
    print(f"null values: {get_null_value_count(result_data)}")#77
    # total non null values
    print(f"total non null values: {len(result_data) - get_null_value_count(result_data)}")#1512
if __name__ == "__main__":
    # done
    # run_null_fix_videochat2_pred_uag_oops_dataset()
    # done
    # run_null_fix_video_chatgpt_pred_uag_oops_dataset()
    # done
    # run_null_fix_video_chatgpt_pred_ssbd_dataset()
    #done
    # run_null_fix_video_llama2_pred_uag_oops_dataset()
    #done
    # run_null_fix_video_llama2_pred_ssbd_dataset()
    # ##TODO
    # run_null_fix_uag_oops("llama3_pred_x_blip2_text_rep_x_uag_oops", llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path,llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path_null_fixed,time_extract_model="llama3")
    run_null_fix_ssbd("llama3_pred_x_blip2_text_rep_x_ssbd", llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path,llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path_null_fixed,time_extract_model="llama3")


