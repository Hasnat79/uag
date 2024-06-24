import sys
import json
import random
random.seed(42)
import numpy as np
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import videochat2_pred_uag_oops_v1_path,video_chatgpt_pred_x_ssbd_result_null_fixed_path,video_chatgpt_pred_x_uag_oops_dataset_null_fixed_path, video_llama2_pred_x_uag_oops_dataset_null_fixed_path,video_llama2_pred_x_ssbd_result_null_fixed_path,llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path_null_fixed,llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path_null_fixed,llama3_pred_x_videollama2_text_rep_x_ssbd_dataset_path_null_fixed,llama3_pred_x_videollama2_text_rep_x_uag_oops_dataset_v1_path_null_fixed,finetuned_videollama_pred_x_uag_oops_dataset_null_fixed_path,finetuned_videollama_pred_x_ssbd_dataset_null_fixed_path

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
    if isinstance(pred_start,float) :
        pred_start = pred_start
    elif ":" in pred_start :
        pred_start = pred_start.split(":")
        if len(pred_start) == 3:
            pred_start = int(pred_start[0])*3600 + int(pred_start[1])*60 + int(pred_start[2])
        elif len(pred_start) == 2:
            pred_start = int(pred_start[0])*60 + int(pred_start[1])
    if isinstance(pred_end,float):
        pred_end = pred_end
    # if the end format is hour:minute:seconds, convert the format to seconds
    elif ":" in pred_end:
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
# def run_eval_random_pred_oops(model):
#     print(f"\n\n============== {model} ==============")
#     #set seed to 42
#     random.seed(24)
#     with open(videochat2_pred_uag_oops_v1_path, 'r') as f:
#         videochat2_pred_uag_oops_v1 = json.load(f)
#     print(f"videochat2_pred_uag_oops_v1: {len(videochat2_pred_uag_oops_v1)}")

#     abs_distance_percentages = []
#     IoU_threshold = [0,0.001,0.01,0.1,0.3 ,0.5, 0.7]
#     # IoU_threshold = [0.7]
#     toggle = True
#     for threshold in IoU_threshold:
#         # calculating iou recall@1 for threshold
#         correct_count = 0
#         # print(f"threshold m: {threshold}")
#         for video_id,video_info in videochat2_pred_uag_oops_v1.items(): 
#             gt_start,gt_end = process_gts(video_info['start_time'],video_info['end_time'])
#             threshold_gt, threshold_end = max(0,gt_start-gt_start*threshold), gt_end+gt_end*threshold
#             # generate random pd_start and pd_end within the range of labels[result["label"]]["duration"]
#             duration = gt_end
#             pred_start,pred_end = generate_random_prediction(duration)


            
#             # print(f"gt_start: {gt_start} gt_end: {gt_end} pred_start: {pred_start} pred_end: {pred_end}")
#             # print(f"temp_gt: {threshold_gt} temp_end: {threshold_end}")
#             # print(f"absolute_distance_percentage: {absolute_distance_percentage}")
#             if pred_start >= threshold_gt and pred_end <= threshold_end:
#                 # print(f"gt_start: {gt_start} gt_end: {gt_end} pred_start: {pred_start} pred_end: {pred_end}")
#                 # print("True")
#                 # print(f"abs_start_distance: {abs_start_distance} gt_start: {gt_start} threshold: {gt_start+gt_start*threshold}")
#                 correct_count += 1

            
#             # if toggle:
#             #     abs_distance_percentages.append(absolute_distance_percentage)
#         toggle = False
#         abs_recall_top_1 = 100*(correct_count / len(videochat2_pred_uag_oops_v1))
#         print(f"Threshold m = {threshold} R@1: {abs_recall_top_1:.2f} ")
        # print(f"Threshold m = {threshold} R@1: {abs_recall_top_1:.2f} mean abs distance percentage: {np.mean(abs_distance_percentages):.2f}")
def run_eval_random_pred_oops(model):
    print(f"\n\n============== {model} ==============")
    #set seed to 42
    with open(videochat2_pred_uag_oops_v1_path, 'r') as f:
        videochat2_pred_uag_oops_v1 = json.load(f)
    print(f"videochat2_pred_uag_oops_v1: {len(videochat2_pred_uag_oops_v1)}")

    abs_distances = []
    threshold_seconds = [0,1,3,5,7]
    toggle = True
    for threshold in threshold_seconds:
        # calculating iou recall@1 for threshold
        correct_count = 0
        for video_id,video_info in videochat2_pred_uag_oops_v1.items(): 
            gt_start,gt_end = process_gts(video_info['start_time'],video_info['end_time'])
            # generate random pd_start and pd_end within the range of labels[result["label"]]["duration"]
            duration = gt_end
            pred_start,pred_end = generate_random_prediction(duration)
            abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end , pred_start, pred_end)
            if abs_distance_between_gt_pred <= threshold:
                correct_count += 1
            if toggle:
                abs_distances.append(abs_distance_between_gt_pred)
        toggle = False
        abs_recall_top_1 = 100*(correct_count / len(videochat2_pred_uag_oops_v1))
        print(f"correct_count: {correct_count} len(result): {len(videochat2_pred_uag_oops_v1)}")
        print(f"Threshold m = {threshold}s R@1: {abs_recall_top_1:.2f} mean abs distance: {np.mean(abs_distances):.2f}")
def run_eval_random_pred_ssbd(model):
    print(f"\n\n============== {model} ==============")
    #set seed to 42
    
    with open(video_chatgpt_pred_x_ssbd_result_null_fixed_path, 'r') as f:
        video_chatgpt_ssbd_pred = json.load(f)
    print(f"videochat2_pred_uag_oops_v1: {len(video_chatgpt_ssbd_pred)}")
    
    abs_distances = []
    threshold_seconds = [0,1,3,5,7]
    toggle = True
    for threshold in threshold_seconds:
        # calculating iou recall@1 for threshold
        correct_count = 0
        for video_id,video_info in video_chatgpt_ssbd_pred.items(): 
            gt_start,gt_end =process_ssbd_gts(video_info['behaviour']['time'])
            # generate random pd_start and pd_end within the range of labels[result["label"]]["duration"]
            with open ("/scratch/user/hasnat.md.abdullah/uag/data/ssbd/ssbd_labels.json", 'r') as f:
                ssbd_labels = json.load(f)

            duration = float(ssbd_labels[video_id[:-5]]["duration"][:-1])
            pred_start,pred_end = generate_random_prediction(duration)
            abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end , pred_start, pred_end)
            if abs_distance_between_gt_pred <= threshold:
                correct_count += 1
            if toggle:
                abs_distances.append(abs_distance_between_gt_pred)
        toggle = False
        print(f"correct_count: {correct_count} len(result): {len(video_chatgpt_ssbd_pred)}")
        abs_recall_top_1 = 100*(correct_count / len(video_chatgpt_ssbd_pred))
        print(f"Threshold m = {threshold}s R@1: {abs_recall_top_1:.2f} mean abs distance: {np.mean(abs_distances):.2f}")

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
    with open ("/scratch/user/hasnat.md.abdullah/uag/data/ssbd/ssbd_labels.json",'r') as f:
        ssbd_labels = json.load(f)
    sample_len = len(ssbd_result)
    threshold_seconds = [0,1,3,5,7]
    abs_distances = []
    toggle = True
    for threshold in threshold_seconds:
        correct_count = 0
        for video_id,video_info in ssbd_result.items():
            if model == "llama3_blip2_text_rep_x_ssbd_dataset":
                gt_start,gt_end = process_ssbd_gts(video_info["behavior"]["time"])
            else: 
                gt_start,gt_end = process_ssbd_gts(video_info["behaviour"]["time"])
            duration = float(ssbd_labels[video_id[:-5]]["duration"][:-1])
            if video_info['pred_start'] is None and video_info['pred_end'] is None:
                # if both pred_start and pred_end are None, generate random prediction for both
                pred_start,pred_end = generate_random_prediction(duration)
                # print(f"video_id: {video_id}")
                # print(f"pred_start: {pred_start} pred_end: {pred_end}")
                # break
            elif video_info['pred_start'] is None and video_info['pred_end'] is not None:
                # if pred_start is None, generate random prediction for pred_start
                _, pred_end = proces_preds("00:00",video_info['pred_end'])
                rand_pred_start,_ = generate_random_prediction(duration)
                while rand_pred_start > pred_end:
                    rand_pred_start,_ = generate_random_prediction(duration)
                pred_start = rand_pred_start
            elif video_info['pred_end'] is None and video_info['pred_start'] is not None:
                # if pred_end is None, generate random prediction for pred_end
                pred_start,_ = proces_preds(video_info['pred_start'],"00:00")
                _,rand_pred_end = generate_random_prediction(duration)
                while pred_start > rand_pred_end:
                    _,rand_pred_end = generate_random_prediction(duration)
                pred_end = rand_pred_end
            else:
              pred_start,pred_end = proces_preds(video_info['pred_start'],video_info['pred_end'])
            abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end,pred_start,pred_end)
            if abs_distance_between_gt_pred <= threshold:
                correct_count += 1
            if toggle:
                abs_distances.append(abs_distance_between_gt_pred)
        toggle = False
        print(f"correct_count: {correct_count} len(result): {sample_len}")
        
        recall_1_abs_dist_m = 100*(correct_count / sample_len)
        # print(f"Threshold m = {threshold} R@1 abs dist: {recall_1_abs_dist_m:.2f} mean abs distance: {np.mean(abs_distances):.2f}")
        print(f"Threshold m = {threshold}s R@1: {recall_1_abs_dist_m:.2f} mean abs distance: {np.mean(abs_distances):.2f}")

    

def run_eval_on_uag_oops(model_name,result_path):
    print(f"\n\n============== {model_name} ==============")
    with open(result_path, 'r') as f:
        result = json.load(f)
    print(f"model name: {model_name} x uag_oops_result: {len(result)}")
    sample_len = len(result)
    abs_distances = []
    toggle = True
    threshold_seconds = [0,1,3,5,7]
    for threshold in threshold_seconds:
        correct_count = 0
        for video_id,video_info in result.items():
            gt_start,gt_end = process_gts(video_info['start_time'],video_info['end_time'])
            threshold_gt, threshold_end = max(0,gt_start-gt_start*threshold), gt_end+gt_end*threshold
            duration = gt_end
            if video_info['pred_start'] is None and video_info['pred_end'] is None:
                # if both pred_start and pred_end are None, generate random prediction for both
                pred_start,pred_end = generate_random_prediction(duration)
                # print(f"video_id: {video_id}")
                # print(f"pred_start: {pred_start} pred_end: {pred_end}")
                # break
            elif video_info['pred_start'] is None and video_info['pred_end'] is not None:
                # if pred_start is None, generate random prediction for pred_start
                _, pred_end = proces_preds("00:00",video_info['pred_end'])
                rand_pred_start,_ = generate_random_prediction(duration)
                while rand_pred_start > pred_end:
                    rand_pred_start,_ = generate_random_prediction(duration)
                pred_start = rand_pred_start
            elif video_info['pred_end'] is None and video_info['pred_start'] is not None:
                # if pred_end is None, generate random prediction for pred_end
                pred_start,_ = proces_preds(video_info['pred_start'],"00:00")
                _,rand_pred_end = generate_random_prediction(duration)
                while pred_start > rand_pred_end:
                    _,rand_pred_end = generate_random_prediction(duration)
                pred_end = rand_pred_end
            else:
              pred_start,pred_end = proces_preds(video_info['pred_start'],video_info['pred_end'])

            abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end , pred_start, pred_end)
            if abs_distance_between_gt_pred <= threshold:
                correct_count += 1
            if toggle:
                abs_distances.append(abs_distance_between_gt_pred)
        toggle = False
        abs_recall_top_1 = 100*(correct_count / sample_len)
        print(f"correct_count: {correct_count} len(result): {sample_len}")
        print(f"Threshold m = {threshold}s R@1: {abs_recall_top_1:.2f} mean abs distances: {np.mean(abs_distances):.2f}")

            
def  run_eval_on_ssbd_videochat2(model, result_path ="/scratch/user/hasnat.md.abdullah/uag/results/videochat2_ssbd_res_pred_gt.json"):
    print(f"\n\n============== {model} ==============")
    with open(result_path, 'r') as f:
        videochat2_ssbd_pred = json.load(f)
    print(f"videochat2_ssbd_pred: {len(videochat2_ssbd_pred)}")
    sample_len = len(videochat2_ssbd_pred)
    abs_distances = []
    toggle = True
    threshold_seconds = [0,1,3,5,7]
    for threshold in threshold_seconds:
        correct_count = 0
        for res in videochat2_ssbd_pred:
            gt_start,gt_end = proces_preds(res['gt'][0],res['gt'][1])

            pred_start, pred_end = proces_preds(res['pred'][0],res['pred'][1]) 
            original_distance = gt_end - gt_start
            abs_distance_between_gt_pred = calculate_abs_dist(gt_start,gt_end,pred_start,pred_end)
            if toggle:
                abs_distances.append(abs_distance_between_gt_pred)
            if abs_distance_between_gt_pred <= threshold:
                correct_count += 1
        toggle = False
        print(f"correct_count: {correct_count} len(result): {sample_len}")
        recall_1_abs_dist_m = 100*(correct_count / sample_len)
        print(f"Threshold m = {threshold}s R@1 abs dist: {recall_1_abs_dist_m:.2f} mean abs distance: {np.mean(abs_distances):.2f}")
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
                    "llama3_blip2_text_rep_x_ssbd_dataset",
                    "llama3_videollama2_text_rep_x_ssbd_dataset",
                    "llama3_videollama2_text_rep_x_uag_oops_dataset_v1",
                    "fine_tuned_videollama_uag_oops_dataset",
                    "fine_tuned_videollama_ssbd_dataset"
                    ]
    for model in foundation_models:
        # # done
        # if model == "random prediction_oops":
        #     run_eval_random_pred_oops(model)
        # if model == "random prediction_ssbd":
        #     run_eval_random_pred_ssbd(model)
        # if model == "videochat2_uag_oops_v1":
        #     run_eval_on_uag_oops(model,result_path = videochat2_pred_uag_oops_v1_path)
        # if model == "videochat2_ssbd":
        #     run_eval_on_ssbd_videochat2(model, result_path ="/scratch/user/hasnat.md.abdullah/uag/results/videochat2_ssbd_res_pred_gt.json")
        # if model == "video_chatgpt_uag_oops_v1":
        #     run_eval_on_uag_oops(model,result_path = video_chatgpt_pred_x_uag_oops_dataset_null_fixed_path)
        # if model == "video_chatgpt_ssbd":
        #     run_eval_on_ssbd(model, result_path = video_chatgpt_pred_x_ssbd_result_null_fixed_path)
            
        # if model == "video_llama2_uag_oops_v1":
        #     run_eval_on_uag_oops(model,result_path = video_llama2_pred_x_uag_oops_dataset_null_fixed_path)
            
        # if model == "video_llama2_ssbd":
        #     run_eval_on_ssbd(model, result_path = video_llama2_pred_x_ssbd_result_null_fixed_path)
        
        # if model == "llama3_blip2_text_rep_x_uag_oops_dataset_v1":  
        #     run_eval_on_uag_oops(model,result_path = llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path_null_fixed)
        #     # run_eval_llama3_blip2_text_rep_x_uag_oops_dataset_v1(model)
        # if model == "llama3_blip2_text_rep_x_ssbd_dataset":
        #     run_eval_on_ssbd(model, result_path = llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path_null_fixed)
        # if model == "llama3_videollama2_text_rep_x_ssbd_dataset":
        #     run_eval_on_ssbd(model, result_path = llama3_pred_x_videollama2_text_rep_x_ssbd_dataset_path_null_fixed)
        # if model == "llama3_videollama2_text_rep_x_uag_oops_dataset_v1":
        #     run_eval_on_uag_oops(model,result_path = llama3_pred_x_videollama2_text_rep_x_uag_oops_dataset_v1_path_null_fixed)
        if model == "fine_tuned_videollama_uag_oops_dataset":
            run_eval_on_uag_oops(model,result_path = finetuned_videollama_pred_x_uag_oops_dataset_null_fixed_path)
        if model == "fine_tuned_videollama_ssbd_dataset":
            run_eval_on_ssbd(model, result_path = finetuned_videollama_pred_x_ssbd_dataset_null_fixed_path)
        

