import json
import os
from tqdm import tqdm
import re
import random
def calculate_iou(prediction, ground_truth):
    # Convert minutes:seconds format to seconds
    pred_start = int(prediction[0].split(':')[0]) * 60 + int(prediction[0].split(':')[1])
    pred_end = int(prediction[1].split(':')[0]) * 60 + int(prediction[1].split(':')[1])
    gt_start = int(ground_truth[0].split(':')[0]) * 60 + int(ground_truth[0].split(':')[1])
    gt_end = int(ground_truth[1].split(':')[0]) * 60 + int(ground_truth[1].split(':')[1])

    # print(f"pred_start: {pred_start}, pred_end: {pred_end}")
    # print(f"gt_start: {gt_start}, gt_end: {gt_end}")
    # Calculate the intersection
    intersection_start = max(pred_start, gt_start)
    intersection_end = min(pred_end, gt_end)
    
    # Calculate the union
    union_start = min(pred_start, gt_start)
    union_end = max(pred_end, gt_end)

    intersection = max(0, intersection_end - intersection_start)
    union = union_end - union_start

    # Calculate IoU
    iou = intersection / union if union != 0 else 0
    return iou

def evaluate_iou(predictions, ground_truths, threshold):
    tp = 0 # True positive 
    for i in range(len(predictions)):
        iou = calculate_iou(predictions[i], ground_truths[i])
        # print(f"iou: {iou}")
        if iou >= threshold:
            tp+=1
    


    recall = tp / float(len(ground_truths)) # true positive rate
    return recall 
    # iou = calculate_iou(predictions[0], ground_truths[0])
    # print(f"iou: {iou}")
    # if iou >= threshold:
    #     return 1
    # else:
    #     return 0

# # Example data
# predicted = [("01:30", "02:00")]
# ground_truth = [("01:20", "02:10")]


def main():
    # results_path = "/scratch/user/hasnat.md.abdullah/uag/Foundation_model_evaluation/videochat2/results/ssbd_test_result.json"
    # with open(results_path, "r") as f:
    #     results = json.load(f)
    # r_save = []
    # gts = []
    # pred = []
    # progress_bar = tqdm(total=len(results))
    # pattern_1 = r'(\d{1,2}:\d{2}(:\d{2})?)'
    # # pattern_2 = r'(\d{2}:\d{2}:\d{2})'
    # for result in results:
    #     gt = result["query"]["time"]
    #     gt_start = gt.split(":")[0][:2]+":"+gt.split(":")[0][2:]
    #     gt_end = gt.split(":")[1][:2]+":"+gt.split(":")[1][2:]
    #     # print(f"gt_start: {gt_start}, gt_end: {gt_end}")
    #     gts.append((gt_start, gt_end))
        

    #     times = re.findall(pattern_1, result["answer"])
    #     # print(f"times patter_1: {times}")
    #     if len(times) >= 2:
    #         if len(times[0][0])< 5:
    #             pd_start = times[0][0][-4:]
    #             pd_end = times[1][0][-4:]
    #         else: 
    #             pd_start = times[0][0][-5:]
    #             pd_end = times[1][0][-5:]
    #         # print(f"start_time: {start_time}, end_time: {end_time}")
    #     else: 
    #         # print(f"result label: {result['label']}")
    #         # print(result)
    #         pd_start = "00:00"
    #         pd_end = "00:00"
    #     pred.append((pd_start, pd_end))
    #     r_save.append({"label": result["label"], "gt": (gt_start, gt_end), "pred": (pd_start, pd_end)})
    #     progress_bar.update(1)
    # save pred and gts
    with open("res_pred_gt.json", "r") as f:
        res = json.load(f)
    gts = []
    pred = []
    for r in res:
        gts.append(r["gt"])
        pred.append(r["pred"])

    # Calculate IoU for different thresholds
    thresholds = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    for threshold in thresholds:
        iou = evaluate_iou(pred, gts, threshold)
        print(f"IoU@{threshold}: {iou}")

## generate random predictions but keep the gts same
def random_predictions():
    # results_path = "/scratch/user/hasnat.md.abdullah/uag/Foundation_model_evaluation/videochat2/results/ssbd_test_result.json"
    labels = "/scratch/user/hasnat.md.abdullah/uag/data/ssbd/ssbd_labels.json"
    # set up seed
    random.seed(42)
    with open(labels, "r") as f:
        labels = json.load(f)

    # with open(results_path, "r") as f:
    #     results = json.load(f)

    with open("res_pred_gt.json", "r") as f:
        res = json.load(f)
    gts = []
    pred = []
    for r in res:
        gts.append(r["gt"])
        # pred.append(r["pred"])
        # generate random pd_start and pd_end within the range of labels[result["label"]]["duration"]
        duration = labels[r["label"]]["duration"][:-1]
        # print(f"duration: {duration}")

        # make sure start time is less than end time
        pd_start = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
        pd_end = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
        while pd_start >= pd_end:
            pd_start = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
            pd_end = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
        pred.append((pd_start, pd_end))


    # r_save = []
    # gts = []
    # pred = []
    # progress_bar = tqdm(total=len(results))
    # # pattern_2 = r'(\d{2}:\d{2}:\d{2})'
    # for result in results:
    #     gt = result["query"]["time"]
    #     gt_start = gt.split(":")[0][:2]+":"+gt.split(":")[0][2:]
    #     gt_end = gt.split(":")[1][:2]+":"+gt.split(":")[1][2:]
    #     # print(f"gt_start: {gt_start}, gt_end: {gt_end}")
    #     gts.append((gt_start, gt_end))
        

    #     # generate random pd_start and pd_end within the range of labels[result["label"]]["duration"]
    #     duration = labels[result["label"]]["duration"][:-1]
    #     # print(f"duration: {duration}")
    #     pd_start = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
    #     pd_end = f"{random.randint(0, int(duration)//60):02d}:{random.randint(0, int(duration)%60):02d}"
        
    #     # print(f"pd_start: {pd_start}, pd_end: {pd_end}")
    #     pred.append((pd_start, pd_end))
    #     # r_save.append({"label": result["label"], "gt": (gt_start, gt_end), "pred": (pd_start, pd_end)})
    #     progress_bar.update(1)


    # Calculate IoU for different thresholds
    thresholds = [0.001,0.01,0.1,0.3 ,0.5, 0.7]
    for threshold in thresholds:
        iou = evaluate_iou(pred, gts, threshold)
        print(f"IoU@{threshold}: {iou}")

if __name__ == '__main__':
    # pass
    main()
    random_predictions()

# videochat2 results
#     IoU@0.3: 0.028846153846153848
# IoU@0.5: 0.009615384615384616
# IoU@0.7: 0.0

#------------------------------------------
# Random predictions 
# IoU@0.3: 0.057692307692307696
# IoU@0.5: 0.019230769230769232
# IoU@0.7: 0.009615384615384616