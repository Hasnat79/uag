import json
import os
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")

from configs.configure import ssbd_data_path

class Ssbd_DataLoader():
    def __init__(self):
        self.data = self.load_data()
    def load_data(self):
        with open(ssbd_data_path, 'r') as f:
            data = json.load(f)
        return data

    def __iter__(self):
        for sample in self.data:
            video_tag = sample[0]
            behaviour_id = sample[2]['id']
            video_id = video_tag+'_'+str(behaviour_id)
            video_info = {}
            video_info = {'video_path': sample[1], 'behaviour': sample[2]}
            yield video_id,video_info
       
    def __len__(self):
        return len(self.data)
    
if __name__ == "__main__":
    ssbd_data = Ssbd_DataLoader()
    print(f"Total videos in SSBD: {len(ssbd_data)}")
    for video_id, video_info in ssbd_data:
        ''' DATA PREVIEW:
        video_id: v_ArmFlapping_01_b_01 
        video_info:  {'video_path': '/scratch/user/hasnat.md.abdullah/uag/data/ssbd/videos/v_ArmFlapping_01.mp4',
          'behaviour': {'id': 'b_01', 'time': '0018:0024', 'bodypart': 'hand', 'category': 'armflapping', 'intensity': 'high', 'modality': 'video'}}
'''
        print(video_id, video_info)
        break
