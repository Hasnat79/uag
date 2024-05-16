import json
path = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/svos/val.json"
with open(path,'r') as f:
    data = json.load(f)

with open(path, 'w') as f:
    json.dump(data, f, indent=4)