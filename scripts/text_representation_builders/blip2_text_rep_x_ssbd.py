import json
import sys
import os
from tqdm import tqdm
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import ssbd_data_path,blip2_text_rep_x_ssbd_dataset_path

blip2_ssbd_text_rep_dataset_spiderman_server_path="/scratch/user/hasnat.md.abdullah/uag/scripts/text_representation_builders/ssbd_text_rep_dataset.json"

def get_text_representation(blip2_ssbd_text_rep_dataset,video_id):
  i=0
  text_rep =""
  for k, caption in blip2_ssbd_text_rep_dataset[video_id].items():
    text_rep+=f"{i}.0s: "
    text_rep+=caption
    i+=1

  # print(text_rep)
  
  return text_rep

if __name__ == "__main__":
  with open(blip2_ssbd_text_rep_dataset_spiderman_server_path) as f:
      blip2_ssbd_text_rep_dataset = json.load(f)
  with open(ssbd_data_path) as f:
      ssbd_data = json.load(f)
  
  
  
  blip2_text_rep_x_ssbd_dataset = {}
  if os.path.exists(blip2_text_rep_x_ssbd_dataset_path):
    with open(blip2_text_rep_x_ssbd_dataset_path) as f:
      blip2_text_rep_x_ssbd_dataset = json.load(f)
      print(f"size of blip2_ssbd_text_rep_dataset: {len(blip2_text_rep_x_ssbd_dataset)}")
  for video_sample in tqdm(ssbd_data):
    video_info = {}
    video_id = video_sample[0]+"_"+video_sample[2]["id"]
    behavior = video_sample[2] #dict
    print(f"video sample: {video_sample}")
    text_rep = get_text_representation(blip2_ssbd_text_rep_dataset,video_sample[0])
    

    video_info['video_path']=video_sample[1]
    video_info['behavior']=behavior
    video_info["text_rep"]=text_rep
    blip2_text_rep_x_ssbd_dataset[video_id] = video_info
  with open(blip2_text_rep_x_ssbd_dataset_path, 'w') as f:
    json.dump(blip2_text_rep_x_ssbd_dataset, f,indent=4)
    print(f"succesfully saved blip2_text_rep_x_ssbd_dataset with {len(blip2_text_rep_x_ssbd_dataset)} samples")



  
