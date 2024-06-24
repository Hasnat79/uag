import json

with open("/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/train_descriptions.json", 'r') as f:
    train_descriptions = json.load(f)
with open("/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/train_descriptions.json", 'w') as f:
    json.dump(train_descriptions, f, indent=4)
    
