import sys
import json
import random
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import videochat2_pred_uag_oops_v1_path,video_chatgpt_pred_x_ssbd_result_null_fixed_path,video_chatgpt_pred_x_uag_oops_dataset_null_fixed_path, video_llama2_pred_x_uag_oops_dataset_null_fixed_path,video_llama2_pred_x_ssbd_result_null_fixed_path

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


def run_eval_videochat2(model_name):
    print(f"\n\n============== {model_name} ==============")
    with open(videochat2_pred_uag_oops_v1_path, 'r') as f:
        videochat2_pred_uag_oops_v1 = json.load(f)
    print(f"videochat2_pred_uag_oops_v1: {len(videochat2_pred_uag_oops_v1)}")
    # we will ignore the null 
    videochat2_pred_uage_oops_v1_null_fixed = get_null_removed_from_res(videochat2_pred_uag_oops_v1)

    IoU_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    for threshold in IoU_threshold:
        # calculating iou recall@1 for threshold
        correct_count = 0
        for video_id,video_info in videochat2_pred_uage_oops_v1_null_fixed.items(): 
            gt_start,gt_end = process_gts(video_info['start_time'],video_info['end_time'])
        
            pred_start,pred_end = proces_preds(video_info['pred_start'],video_info['pred_end'])
            iou_recall_top_1 = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            iou = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            # print(f"iou: {iou}")
            if iou >= threshold:
                correct_count += 1
        iou_recall_top_1 = 100*(correct_count / len(videochat2_pred_uage_oops_v1_null_fixed))
        print(f" IoU = {threshold} R@1: {iou_recall_top_1:.2f}")
def generate_random_prediction(duration):
    pd_start =f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
    pd_end = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
    
    while pd_start >= pd_end:
        pd_start = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
        pd_end = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
    # print(f"pd_start: {pd_start} pd_end: {pd_end}")
    pred_start,pred_end = proces_preds(pd_start,pd_end)
    return pred_start,pred_end
def run_eval_random_pred(model):
    print(f"\n\n============== {model} ==============")
    #set seed to 42
    random.seed(42)
    with open(videochat2_pred_uag_oops_v1_path, 'r') as f:
        videochat2_pred_uag_oops_v1 = json.load(f)
    print(f"videochat2_pred_uag_oops_v1: {len(videochat2_pred_uag_oops_v1)}")
    # we will ignore the null 
    videochat2_pred_uage_oops_v1_null_fixed = get_null_removed_from_res(videochat2_pred_uag_oops_v1)

    IoU_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    for threshold in IoU_threshold:
        # calculating iou recall@1 for threshold
        correct_count = 0

        for video_id,video_info in videochat2_pred_uage_oops_v1_null_fixed.items(): 
            gt_start,gt_end = process_gts(video_info['start_time'],video_info['end_time'])
            # generate random pd_start and pd_end within the range of labels[result["label"]]["duration"]
            duration = gt_end
            pred_start,pred_end = generate_random_prediction(duration)
            
            iou = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            if iou >= threshold:
                correct_count += 1
        iou_recall_top_1 = 100*(correct_count / len(videochat2_pred_uage_oops_v1_null_fixed))
        print(f" IoU = {threshold} R@1: {iou_recall_top_1:.2f}")


def run_eval_video_chatgpt_oops(model,result_path =""):
    print(f"\n\n============== {model} ==============")
    with open(result_path, 'r') as f:
        video_chatgpt_pred_x_uag_oops_dataset = json.load(f)
    print(f"video_chatgpt_pred_x_uag_oops_dataset: {len(video_chatgpt_pred_x_uag_oops_dataset)}")
    # we will ignore the null 
    video_chatgpt_pred_x_uag_oops_dataset_null_fixed = get_null_removed_from_res(video_chatgpt_pred_x_uag_oops_dataset)

    IoU_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    for threshold in IoU_threshold:
        # calculating iou recall@1 for threshold
        correct_count = 0
        for video_id,video_info in video_chatgpt_pred_x_uag_oops_dataset_null_fixed.items(): 
            gt_start,gt_end = process_gts(video_info['start_time'],video_info['end_time'])

            
            pred_start,pred_end = proces_preds(video_info['pred_start'],video_info['pred_end'])
            # print(f"gt_start: {gt_start} gt_end: {gt_end} pred_start: {pred_start} pred_end: {pred_end}")
            iou_recall_top_1 = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            iou = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            # print(f"iou: {iou}")
            if iou >= threshold:
                correct_count += 1
        iou_recall_top_1 = 100*(correct_count / len(video_chatgpt_pred_x_uag_oops_dataset_null_fixed))
        print(f" IoU = {threshold} R@1: {iou_recall_top_1:.2f}")
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
    
def run_eval_video_chatgpt_ssbd(model):
    print(f"\n\n============== {model} ==============")
    with open(video_chatgpt_pred_x_ssbd_result_null_fixed_path, 'r') as f:
        video_chatgpt_pred_x_ssbd_result = json.load(f)
    print(f"video_chatgpt_pred_x_ssbd_result: {len(video_chatgpt_pred_x_ssbd_result)}")
    # we will ignore the null 
    video_chatgpt_pred_x_ssbd_result_null_fixed = get_null_removed_from_res(video_chatgpt_pred_x_ssbd_result)

    IoU_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    for threshold in IoU_threshold:
        # calculating iou recall@1 for threshold
        correct_count = 0
        for video_id,video_info in video_chatgpt_pred_x_ssbd_result_null_fixed.items(): 
            gt_start,gt_end = process_ssbd_gts(video_info["behaviour"]["time"])
            # print(f"gt_start: {gt_start} gt_end: {gt_end}")
            pred_start,pred_end = proces_preds(video_info['pred_start'],video_info['pred_end'])
            # print(f"gt_start: {gt_start} gt_end: {gt_end} pred_start: {pred_start} pred_end: {pred_end}")
            iou_recall_top_1 = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            iou = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            # print(f"iou: {iou}")
            if iou >= threshold:
                correct_count += 1
        iou_recall_top_1 = 100*(correct_count / len(video_chatgpt_pred_x_ssbd_result_null_fixed))
        print(f" IoU = {threshold} R@1: {iou_recall_top_1:.2f}")
def run_eval_video_llama2_oops(model,result_path =""):
    print(f"\n\n============== {model} ==============")
    with open(result_path, 'r') as f:
        video_llama2_pred_x_uag_oops_dataset = json.load(f)
    print(f"video_llama2_pred_x_uag_oops_dataset: {len(video_llama2_pred_x_uag_oops_dataset)}")
    # we will ignore the null 
    video_llama2_pred_x_uag_oops_dataset_null_fixed = get_null_removed_from_res(video_llama2_pred_x_uag_oops_dataset)
    

    IoU_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    for threshold in IoU_threshold:
        # calculating iou recall@1 for threshold
        correct_count = 0
        for video_id,video_info in video_llama2_pred_x_uag_oops_dataset_null_fixed.items(): 
            gt_start,gt_end = process_gts(video_info['start_time'],video_info['end_time'])

            
            pred_start,pred_end = proces_preds(video_info['pred_start'],video_info['pred_end'])
            # print(f"gt_start: {gt_start} gt_end: {gt_end} pred_start: {pred_start} pred_end: {pred_end}")
            iou_recall_top_1 = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            iou = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            # print(f"iou: {iou}")
            if iou >= threshold:
                correct_count += 1
        iou_recall_top_1 = 100*(correct_count / len(video_llama2_pred_x_uag_oops_dataset_null_fixed))
        print(f" IoU = {threshold} R@1: {iou_recall_top_1:.2f}")

def run_eval_video_llama2_ssbd(model,result_path = ""):
    print(f"\n\n============== {model} ==============")
    with open(result_path, 'r') as f:
        video_llama2_pred_x_ssbd_result = json.load(f)
    print(f"video_llama2_pred_x_ssbd_result: {len(video_llama2_pred_x_ssbd_result)}")
    # we will ignore the null 
    video_llama2_pred_x_ssbd_result_null_fixed = get_null_removed_from_res(video_llama2_pred_x_ssbd_result)
    IoU_threshold = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    for threshold in IoU_threshold:
        # calculating iou recall@1 for threshold
        correct_count = 0
        for video_id,video_info in video_llama2_pred_x_ssbd_result_null_fixed.items(): 
            gt_start,gt_end = process_ssbd_gts(video_info["behaviour"]["time"])
            # print(f"gt_start: {gt_start} gt_end: {gt_end}")
            pred_start,pred_end = proces_preds(video_info['pred_start'],video_info['pred_end'])
            # print(f"gt_start: {gt_start} gt_end: {gt_end} pred_start: {pred_start} pred_end: {pred_end}")
            iou_recall_top_1 = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            iou = calculate_iou(gt_start,gt_end,pred_start,pred_end)
            # print(f"iou: {iou}")
            if iou >= threshold:
                correct_count += 1
        iou_recall_top_1 = 100*(correct_count / len(video_llama2_pred_x_ssbd_result_null_fixed))
        print(f" IoU = {threshold} R@1: {iou_recall_top_1:.2f}")
if __name__ == "__main__":
    foundation_models =["random prediction",
                    "videochat2_uag_oops_v1",
                    "video_chatgpt_uag_oops_v1",
                    "video_chatgpt_ssbd",
                    "video_llama2_uag_oops_v1",
                    "video_llama2_ssbd"
                    ]
    for model in foundation_models:
        # done
        if model == "random prediction":
            run_eval_random_pred(model)
        if model == "videochat2_uag_oops_v1":
            run_eval_videochat2(model)
        if model == "video_chatgpt_uag_oops_v1":
            run_eval_video_chatgpt_oops(model,result_path = video_chatgpt_pred_x_uag_oops_dataset_null_fixed_path)
        if model == "video_chatgpt_ssbd":
            run_eval_video_chatgpt_ssbd(model)
        if model == "video_llama2_uag_oops_v1":
            run_eval_video_llama2_oops(model,result_path = video_llama2_pred_x_uag_oops_dataset_null_fixed_path)
        if model == "video_llama2_ssbd":
            run_eval_video_llama2_ssbd(model,result_path = video_llama2_pred_x_ssbd_result_null_fixed_path)
            
            
        

