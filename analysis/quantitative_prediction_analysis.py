#approximately 0:14-0:21.",

import json
import os
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")


from configs.configure import *

with open("/scratch/user/hasnat.md.abdullah/uag/results/video_llama2_pred_uag_oops_dataset_v1.json") as f:
    data = json.load(f)
#print on how many samples the "video_llama2_pred" key had the word "approximately" in the ssbd dataset
c = 0
for id,info in data.items():
    if "video_llama2_pred" in info:
        if "approximately" in info["video_llama2_pred"]:
            c+=1

print(f" uag oops v1 dataset: The word 'approximately' was found in the 'video_llama2_pred' key of {c} samples.")# 585 samples

with open("/scratch/user/hasnat.md.abdullah/uag/results/video_llama2_pred_ssbd_dataset.json") as f:
    data = json.load(f)
#print on how many samples the "video_llama2_pred" key had the word "approximately" in the ssbd dataset
c = 0
for id,info in data.items():
    if "video_llama2_pred" in info:
        if "approximately" in info["video_llama2_pred"]:
            c+=1

print(f" ssbd dataset: The word 'approximately' was found in the 'video_llama2_pred' key of {c} samples.")# 585 samples


# how many uag_oops_v1 dataset samples have "don't know don't know" in the description key
with open("/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/uag_oops_dataset/uag_oops_dataset_v1.json") as f:
    data = json.load(f)
c = 0
for id, info in data.items():
    if "description" in info:
        if "Don't know Don't know" in info["description"]:
            c+=1
print(f"uag_oops_v1 dataset: The phrase 'don't know don't know' was found in the 'description' key of {c} samples.")# 0 samples


# how many llama3 + blip2 predictions has "I cannot provide.." in the prediction
with open ("/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_null_fixed.json") as f:
    data = json.load(f)

c = 0
d = 0
e = 0
for id, info in data.items():
    if "llama3_pred" in info:
        if "explicit content" in info["llama3_pred"]:
            c+=1
        if "I cannot provide" in info["llama3_pred"]:
            d+=1
        if "identify an individual" in info["llama3_pred"]:
            e+=1
print(f"uag_oops_v1 dataset: The phrase 'explicit content' was found in the 'llama3_pred' key of {c} samples.")# 3 samples
print(f"uag_oops_v1 dataset: The phrase 'I cannot provide' was found in the 'llama3_pred' key of {d} samples.")# 12 samples
