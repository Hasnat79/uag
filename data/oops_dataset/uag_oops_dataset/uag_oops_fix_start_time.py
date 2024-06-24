import json
import os
import sys

sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import oops_transition_times_path

uag_oops_v1_path  = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/uag_oops_dataset/uag_oops_dataset_v1.json"


def fix_start_time():
    with open(uag_oops_v1_path, 'r') as f:
        uag_oops_dataset = json.load(f)
    with open(oops_transition_times_path, 'r') as f:
        transition_times = json.load(f)
    for video in uag_oops_dataset:
        if video in transition_times:
            start_time = transition_times[video]['t'][0]
            uag_oops_dataset[video]["start_time"] = start_time
    with open('uag_oops_dataset_v1_start_time_fixed.json', 'w') as f:
        json.dump(uag_oops_dataset, f, indent=4)
    print("start time fixed")


if __name__ == "__main__":
    fix_start_time()
