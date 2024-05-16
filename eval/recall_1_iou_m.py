import sys
import json
import random
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import videochat2_pred_uag_oops_v1_path

def get_null_removed_from_res(videochat2_pred_uag_oops_v1):
    videochat2_pred_uage_oops_v1_null_fixed = {}
    for video_id,video_info in videochat2_pred_uag_oops_v1.items():
        if video_info['pred_start'] is not None and video_info['pred_end'] is not None:
            videochat2_pred_uage_oops_v1_null_fixed[video_id] = video_info
    print(f"videochat2_pred_uage_oops_v1_null_fixed: {len(videochat2_pred_uage_oops_v1_null_fixed)}") #1465
    return videochat2_pred_uage_oops_v1_null_fixed
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
        iou_recall_top_1 = correct_count / len(videochat2_pred_uage_oops_v1_null_fixed)
        print(f" IoU = {threshold} R@1: {iou_recall_top_1:.4f}")
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
        iou_recall_top_1 = correct_count / len(videochat2_pred_uage_oops_v1_null_fixed)
        print(f" IoU = {threshold} R@1: {iou_recall_top_1:.4f}")


if __name__ == "__main__":
    foundation_models =["random prediction","videochat2_uag_oops_v1"]
    for model in foundation_models:
        if model == "random prediction":
            run_eval_random_pred(model)
        if model == "videochat2_uag_oops_v1":
            run_eval_videochat2(model)
        

