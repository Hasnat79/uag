

import json
import os
import statistics
import sys
import json
import os
import numpy as np
import statistics
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import ssbd_data_path
# from scenedetect import detect, ContentDetector, SceneManager, VideoManager


with open("/scratch/user/hasnat.md.abdullah/uag/data/ssbd/ssbd_labels.json") as f:
    ssbd_label = json.load(f)

with open("/scratch/user/hasnat.md.abdullah/uag/data/ssbd/ssbd_test_list.json") as f:
    ssbd_test_list = json.load(f)

# find average and standard deviation of time span of unusual activity in ssbd dataset
time_spans = []

for sample in ssbd_test_list:
    info = sample[2]
    if ":" in info["time"]:
        
        start_time = info["time"].split(":")[0]
        end_time = info["time"].split(":")[1]
    if "-" in info["time"]:
        start_time = info["time"].split("-")[1]
        end_time = info["time"].split("-")[1]

    # print(start_time,end_time)
    start_time = int(start_time[:2])*60 + int(start_time[2:])
    end_time = int(end_time[:2])*60 + int(end_time[2:])
    time_span = end_time - start_time
    time_spans.append(time_span)

print(f"ssbd dataset: The average time span of unusual activity is {np.mean(time_spans):.3f} seconds.")# 0 samples
print(f"ssbd dataset: The standard deviation of time span of unusual activity is {np.std(time_spans):.3f} seconds.")# 0 samples


# they are three category, head banging , armflapping and spinning.
# find the number of samples in each category
head_banging = 0
hand_flapping = 0
spinning = 0
for sample in ssbd_test_list:
    info = sample[2]
    if "headbanging" in info["category"]:
        head_banging+=1
    if "armflapping" in info["category"]:
        hand_flapping+=1
    if "spinning" in info["category"]:
        spinning+=1

print(f"ssbd dataset: The number of samples with headbanging activity is {head_banging}.")# 0 samples
print(f"ssbd dataset: The number of samples with armflapping activity is {hand_flapping}.")# 0 samples
print(f"ssbd dataset: The number of samples with spinning activity is {spinning}.")# 0 samples

# find the category wise average time span of unusual activity
head_banging_time_spans = []
hand_flapping_time_spans = []
spinning_time_spans = []

for sample in ssbd_test_list:
    info = sample[2]
    if "headbanging" in info["category"]:
        if ":" in info["time"]:
            start_time = info["time"].split(":")[0]
            end_time = info["time"].split(":")[1]
        if "-" in info["time"]:
            start_time = info["time"].split("-")[1]
            end_time = info["time"].split("-")[1]

        start_time = int(start_time[:2])*60 + int(start_time[2:])
        end_time = int(end_time[:2])*60 + int(end_time[2:])
        time_span = end_time - start_time
        head_banging_time_spans.append(time_span)
    if "armflapping" in info["category"]:
        if ":" in info["time"]:
            start_time = info["time"].split(":")[0]
            end_time = info["time"].split(":")[1]
        if "-" in info["time"]:
            start_time = info["time"].split("-")[1]
            end_time = info["time"].split("-")[1]

        start_time = int(start_time[:2])*60 + int(start_time[2:])
        end_time = int(end_time[:2])*60 + int(end_time[2:])
        time_span = end_time - start_time
        hand_flapping_time_spans.append(time_span)
    if "spinning" in info["category"]:
        if ":" in info["time"]:
            start_time = info["time"].split(":")[0]
            end_time = info["time"].split(":")[1]
        if "-" in info["time"]:
            start_time = info["time"].split("-")[1]
            end_time = info["time"].split("-")[1]

        start_time = int(start_time[:2])*60 + int(start_time[2:])
        end_time = int(end_time[:2])*60 + int(end_time[2:])
        time_span = end_time - start_time
        spinning_time_spans.append(time_span)

print(f"The average time span of headbanging activity is {np.mean(head_banging_time_spans):.3f} seconds.")# 0 samples
print(f"The average time span of headbanging activity is {np.mean(head_banging_time_spans):.3f} seconds.")# 0 samples
print(f"The average time span of armflapping activity is {np.mean(hand_flapping_time_spans):.3f} seconds.")# 0 samples


# find average duration of the whole video in ssbd dataset
video_durations = []
for id, info in ssbd_label.items():
    duration = float(info["duration"][:-1])
    video_durations.append(duration)

print(f"ssbd dataset: The average duration of the whole video is {np.mean(video_durations):.3f} seconds.")# 0 samples
