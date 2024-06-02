import sys
import json
import random
import numpy as np
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import videochat2_pred_uag_oops_v1_path,video_chatgpt_pred_x_ssbd_result_null_fixed_path,video_chatgpt_pred_x_uag_oops_dataset_null_fixed_path, video_llama2_pred_x_uag_oops_dataset_null_fixed_path,video_llama2_pred_x_ssbd_result_null_fixed_path,llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path_null_fixed,llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path_null_fixed

# def get_null_removed_from_res(videochat2_pred_uag_oops_v1):
#     videochat2_pred_uage_oops_v1_null_fixed = {}
#     for video_id,video_info in videochat2_pred_uag_oops_v1.items():
#         if video_info['pred_start'] is not None and video_info['pred_end'] is not None:
#             videochat2_pred_uage_oops_v1_null_fixed[video_id] = video_info
#     print(f"videochat2_pred_uage_oops_v1_null_fixed: {len(videochat2_pred_uage_oops_v1_null_fixed)}") #1465
#     return videochat2_pred_uage_oops_v1_null_fixed

def get_null_removed_from_res(result_dict):
    null_fixed_dict = {}
    for video_id, video_info in result_dict.items():
        if video_info['pred_start'] is not None and video_info['pred_end'] is not None:
            null_fixed_dict[video_id] = video_info
    print(f"After removing null predictions, total instances: {len(null_fixed_dict)}")
    return null_fixed_dict
def process_gts(gt_start,gt_end):
    # if gt_start < 0, set it to positive
    if gt_start < 0:
        gt_start = -1 * gt_start
    return gt_start,gt_end
def proces_preds(pred_start,pred_end):
    # if the start format is hour:minute:seconds, convert the format to seconds
    # check which format the start time is in
    # if pred_start and pred_end are float
    if isinstance(pred_start,float) and isinstance(pred_end,float):
        return pred_start,pred_end
    if ":" in pred_start:
        pred_start = pred_start.split(":")
        if len(pred_start) == 3:
            pred_start = int(pred_start[0])*3600 + int(pred_start[1])*60 + int(pred_start[2])
        elif len(pred_start) == 2:
            pred_start = int(pred_start[0])*60 + int(pred_start[1])
    # if the end format is hour:minute:seconds, convert the format to seconds
    if ":" in pred_end:
        pred_end = pred_end.split(":")
        if len(pred_end) == 3:
            pred_end = int(pred_end[0])*3600 + int(pred_end[1])*60 + int(pred_end[2])
        elif len(pred_end) == 2:
            pred_end = int(pred_end[0])*60 + int(pred_end[1])
    return pred_start,pred_end
def calculate_iou(gt_start,gt_end,pred_start,pred_end):
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = (gt_end - gt_start) + (pred_end - pred_start) - intersection
    iou = 1.0* (intersection/union)
    return iou
def calculate_abs_dist(gt_start,gt_end , pred_start, pred_end):
    return abs(gt_start - pred_start)+ abs(gt_end - pred_end)



def generate_random_prediction(duration):
    pd_start =f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
    pd_end = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
    
    while pd_start >= pd_end:
        pd_start = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
        pd_end = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
    # print(f"pd_start: {pd_start} pd_end: {pd_end}")
    pred_start,pred_end = proces_preds(pd_start,pd_end)
    return pred_start,pred_end
def run_eval_random_pred_oops(model):
    print(f"\n\n============== {model} ==============")
    #set seed to 42
    random.seed(24)
    with open(videochat2_pred_uag_oops_v1_path, 'r') as f:
        videochat2_pred_uag_oops_v1 = json.load(f)
    print(f"videochat2_pred_uag_oops_v1: {len(videochat2_pred_uag_oops_v1)}")

    abs_distances = []
    IoU_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    toggle = True
    for threshold in IoU_threshold:
        # calculating iou recall@1 for threshold
        correct_count = 0

        for video_id,video_info in videochat2_pred_uag_oops_v1.items(): 
            gt_start,gt_end = process_gts(video_info['start_time'],video_info['end_time'])
            # generate random pd_start and pd_end within the range of labels[result["label"]]["duration"]
            duration = gt_end
            pred_start,pred_end = generate_random_prediction(duration)
            
            original_distance = gt_end - gt_start
            abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end , pred_start, pred_end)
            if abs_distance_between_gt_pred <=  original_distance+ threshold*original_distance:
                # print(f"abs_start_distance: {abs_start_distance} gt_start: {gt_start} threshold: {gt_start+gt_start*threshold}")
                correct_count += 1
            if toggle:
                abs_distances.append(abs_distance_between_gt_pred)
        toggle = False
        abs_recall_top_1 = 100*(correct_count / len(videochat2_pred_uag_oops_v1))
        print(f"Threshold m = {threshold} R@1: {abs_recall_top_1:.2f} mean abs distance: {np.mean(abs_distances):.2f}")
def run_eval_random_pred_ssbd(model):
    print(f"\n\n============== {model} ==============")
    #set seed to 42
    random.seed(42)
    with open(video_chatgpt_pred_x_ssbd_result_null_fixed_path, 'r') as f:
        video_chatgpt_ssbd_pred = json.load(f)
    print(f"videochat2_pred_uag_oops_v1: {len(video_chatgpt_ssbd_pred)}")
    
    abs_distances = []
    IoU_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    toggle = True
    for threshold in IoU_threshold:
        # calculating iou recall@1 for threshold
        correct_count = 0

        for video_id,video_info in video_chatgpt_ssbd_pred.items(): 
            gt_start,gt_end =process_ssbd_gts(video_info['behaviour']['time'])
            # generate random pd_start and pd_end within the range of labels[result["label"]]["duration"]
            with open ("/scratch/user/hasnat.md.abdullah/uag/data/ssbd/ssbd_labels.json", 'r') as f:
                ssbd_labels = json.load(f)

            duration = float(ssbd_labels[video_id[:-5]]["duration"][:-1])
            pred_start,pred_end = generate_random_prediction(duration)
            original_distance = gt_end - gt_start
            abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end , pred_start, pred_end)
            if abs_distance_between_gt_pred <=  original_distance+ threshold*original_distance:
                # print(f"abs_start_distance: {abs_start_distance} gt_start: {gt_start} threshold: {gt_start+gt_start*threshold}")
                correct_count += 1
            if toggle:
                abs_distances.append(abs_distance_between_gt_pred)

        toggle = False
        abs_recall_top_1 = 100*(correct_count / len(video_chatgpt_ssbd_pred))
        print(f"Threshold m = {threshold} R@1 {abs_recall_top_1:.2f} mean abs distance: {np.mean(abs_distances):.2f}")




def process_ssbd_gts(time):
    gt_start = f"{time[:2]}:{time[2:4]}"
    gt_end = f"{time[5:7]}:{time[7:]}"
    if ":" in gt_start:
        gt_start = gt_start.split(":")
        if len(gt_start) == 2:
            gt_start = int(gt_start[0])*60 + int(gt_start[1])
    if ":" in gt_end:
        gt_end = gt_end.split(":")
        if len(gt_end) == 2:
            gt_end = int(gt_end[0])*60 + int(gt_end[1])
    return gt_start,gt_end
    




def run_eval_on_ssbd(model_name, result_path, ignore_null=False):
    print(f"\n\n============== {model_name} ==============")
    with open(result_path, 'r') as f: 
        ssbd_result = json.load(f)
    print(f"model name: {model_name} x ssbd_result: {len(ssbd_result)}")

    result_null_fix = get_null_removed_from_res(ssbd_result)
    if ignore_null:
        sample_len = len(result_null_fix)
    else:
        sample_len = len(ssbd_result)
    
    IoU_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    abs_distances = []
    toggle = True
    for threshold in IoU_threshold:
        correct_count = 0
        for video_id,video_info in result_null_fix.items():
            if model == "llama3_blip2_text_rep_x_ssbd_dataset":
                gt_start,gt_end = process_ssbd_gts(video_info["behavior"]["time"])
            else: 
                gt_start,gt_end = process_ssbd_gts(video_info["behaviour"]["time"])
            pred_start,pred_end = proces_preds(video_info['pred_start'],video_info['pred_end'])
            original_distance = gt_end - gt_start
            abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end,pred_start,pred_end)
            if toggle:
                abs_distances.append(abs_distance_between_gt_pred)
            if abs_distance_between_gt_pred <=  original_distance+ threshold*original_distance:
                # print(f"abs_start_distance: {abs_start_distance} gt_start: {gt_start} threshold: {gt_start+gt_start*threshold}")
                correct_count += 1
        toggle = False
        print(f"correct_count: {correct_count} len(result): {sample_len}")
        recall_1_abs_dist_m = 100*(correct_count / sample_len)
        print(f"Threshold m = {threshold} R@1 abs dist: {recall_1_abs_dist_m:.2f} mean abs distance: {np.mean(abs_distances):.2f}")
    

def run_eval_on_uag_oops(model_name,result_path,ignore_null=False):
    print(f"\n\n============== {model_name} ==============")
    with open(result_path, 'r') as f:
        result = json.load(f)
    print(f"model name: {model_name} x uag_oops_result: {len(result)}")
    result_null_fix = get_null_removed_from_res(result)
    if ignore_null:
        sample_len = len(result_null_fix)
    else:
        sample_len = len(result)
    abs_distances = []
    toggle = True
    abs_distance_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    for threshold in abs_distance_threshold:
        correct_count = 0
        for video_id,video_info in result_null_fix.items():
            gt_start,gt_end = process_gts(video_info['start_time'],video_info['end_time'])
            pred_start,pred_end = proces_preds(video_info['pred_start'],video_info['pred_end'])
            # iou = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            original_distance = gt_end - gt_start
            abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end,pred_start,pred_end)
            if toggle:
                abs_distances.append(abs_distance_between_gt_pred)
            if abs_distance_between_gt_pred <=  original_distance+ threshold*original_distance:
                # print(f"abs_start_distance: {abs_start_distance} gt_start: {gt_start} threshold: {gt_start+gt_start*threshold}")
                correct_count += 1
        toggle = False
        print(f"correct_count: {correct_count} len(result): {sample_len}")
        recall_1_abs_dist_m = 100*(correct_count / sample_len)
        print(f"Threshold m = {threshold} R@1 abs dist: {recall_1_abs_dist_m:.2f} mean abs distance: {np.mean(abs_distances):.2f}")

def  run_eval_on_ssbd_videochat2(model, result_path ="/scratch/user/hasnat.md.abdullah/uag/results/videochat2_ssbd_res_pred_gt.json"):
    print(f"\n\n============== {model} ==============")
    with open(result_path, 'r') as f:
        videochat2_ssbd_pred = json.load(f)
    print(f"videochat2_ssbd_pred: {len(videochat2_ssbd_pred)}")
    sample_len = len(videochat2_ssbd_pred)
    abs_distances = []
    toggle = True
    abs_distance_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    for threshold in abs_distance_threshold:
        correct_count = 0
        for res in videochat2_ssbd_pred:
            gt_start,gt_end = proces_preds(res['gt'][0],res['gt'][1])

            pred_start, pred_end = proces_preds(res['pred'][0],res['pred'][1]) 
            original_distance = gt_end - gt_start
            abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end,pred_start,pred_end)
            if toggle:
                abs_distances.append(abs_distance_between_gt_pred)
            if abs_distance_between_gt_pred <=  original_distance+ threshold*original_distance:
                # print(f"abs_start_distance: {abs_start_distance} gt_start: {gt_start} threshold: {gt_start+gt_start*threshold}")
                correct_count += 1
        toggle = False
        print(f"correct_count: {correct_count} len(result): {sample_len}")
        recall_1_abs_dist_m = 100*(correct_count / sample_len)
        print(f"Threshold m = {threshold} R@1 abs dist: {recall_1_abs_dist_m:.2f} mean abs distance: {np.mean(abs_distances):.2f}")
if __name__ == "__main__":
    foundation_models =["random prediction_oops",
                    "random prediction_ssbd",
                    "videochat2_uag_oops_v1",
                    "videochat2_ssbd",
                    "video_chatgpt_uag_oops_v1",
                    "video_chatgpt_ssbd",
                    "video_llama2_uag_oops_v1",
                    "video_llama2_ssbd",
                    "llama3_blip2_text_rep_x_uag_oops_dataset_v1",
                    "llama3_blip2_text_rep_x_ssbd_dataset"
                    ]
    for model in foundation_models:
        # done
        if model == "random prediction_oops":
            run_eval_random_pred_oops(model)
        if model == "random prediction_ssbd":
            run_eval_random_pred_ssbd(model)
        if model == "videochat2_uag_oops_v1":
            run_eval_on_uag_oops(model,result_path = videochat2_pred_uag_oops_v1_path)
        if model == "videochat2_ssbd":
            run_eval_on_ssbd_videochat2(model, result_path ="/scratch/user/hasnat.md.abdullah/uag/results/videochat2_ssbd_res_pred_gt.json")
        if model == "video_chatgpt_uag_oops_v1":
            run_eval_on_uag_oops(model,result_path = video_chatgpt_pred_x_uag_oops_dataset_null_fixed_path)
        if model == "video_chatgpt_ssbd":
            run_eval_on_ssbd(model, result_path = video_chatgpt_pred_x_ssbd_result_null_fixed_path)
            
        if model == "video_llama2_uag_oops_v1":
            run_eval_on_uag_oops(model,result_path = video_llama2_pred_x_uag_oops_dataset_null_fixed_path)
            
        if model == "video_llama2_ssbd":
            run_eval_on_ssbd(model, result_path = video_llama2_pred_x_ssbd_result_null_fixed_path)
        
        ## TODO
        if model == "llama3_blip2_text_rep_x_uag_oops_dataset_v1":  
            run_eval_on_uag_oops(model,result_path = llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path_null_fixed)
            # run_eval_llama3_blip2_text_rep_x_uag_oops_dataset_v1(model)
        if model == "llama3_blip2_text_rep_x_ssbd_dataset":
            run_eval_on_ssbd(model, result_path = llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path_null_fixed)
            
            
        

