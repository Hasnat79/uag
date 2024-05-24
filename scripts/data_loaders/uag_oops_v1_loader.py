
import json
import os
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")

from configs.configure import uag_oops_dataset_path


class UagOopsV1_DataLoader():
    def __init__(self):
        self.data = self.load_data()
    def load_data(self):
        with open(uag_oops_dataset_path, 'r') as f:
            data = json.load(f)
        return data

    def __iter__(self):
        for video_id,video_info in self.data.items():
            yield video_id,video_info
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    uag_oops_v1 = UagOopsV1_DataLoader()
    print(f"Total videos in UAG OOPS V1: {len(uag_oops_v1)}")
    for video_id, video_info in uag_oops_v1:
        print(video_id, video_info)
        