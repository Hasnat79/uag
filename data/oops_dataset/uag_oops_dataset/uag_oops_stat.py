import json
import numpy as np
with open("/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/uag_oops_dataset/uag_oops_dataset_v1.json") as f:
    data = json.load(f)

# average time span and standard deviation of unusual activity in uag_oops_v1 dataset
time_spans = []

for id, info in data.items():
    start_time = info["start_time"]
    if start_time <0:
        start_time= -1*start_time
    end_time = info["end_time"]
    time_span = end_time - start_time
    time_spans.append(time_span)

print(f"uag_oops_v1 dataset: The average time span of unusual activity is {np.mean(time_spans):.3f} seconds.")# 0 samples
print(f"uag_oops_v1 dataset: The standard deviation of time span of unusual activity is {np.std(time_spans):.3f} seconds.")# 0 samples
