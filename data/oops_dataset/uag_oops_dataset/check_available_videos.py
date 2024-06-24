import json 
import sys
import os

sys.path.append(sys.path.append("/scratch/user/hasnat.md.abdullah/uag/"))


from configs.configure import uag_oops_dataset_path

if __name__ == "__main__":
    with open(uag_oops_dataset_path, 'r') as f:
        data = json.load(f)

    data_available = 0
    for id,info in data.items():
        if os.path.exists(info['video_path']):
            data_available+=1

        else: 
            print(f"id {id}\nvideo path {info['video_path']} not found")
            print()

    print(f"Total data available: {data_available} out of {len(data)}")
