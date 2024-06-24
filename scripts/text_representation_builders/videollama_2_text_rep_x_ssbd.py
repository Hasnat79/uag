import json
import sys
import os
import shutil
from tqdm import tqdm
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import videollama2_text_rep_x_ssbd_dataset_path,temp_video_frames_dir
from scripts.data_loaders.ssbd_loader import Ssbd_DataLoader
from scripts.model_loaders.video_llama2_loader import VideoLlama2Loader
from scripts.text_representation_builders.utils import save_temporary_frames_from_video

def generate_text_representation_from_video(videollama2, video_path=""):
  save_temporary_frames_from_video(video_path)
  
  video_frames = [os.path.join(temp_video_frames_dir, f) for f in os.listdir(temp_video_frames_dir) if f.endswith('.png')]
  sorted_video_frames = sorted(video_frames,key = lambda x: int(x.split("/")[-1].split(".")[0]))
  text_input = "What is happeing in the image? Instruction: answer within one line and cover all the details"
  text_rep = ""
  for i,frame in enumerate(sorted_video_frames):
    text_rep += f"{i}.0s: "
    text_rep += videollama2.infer(video_path=None, gr_img = frame,text_input=text_input)+".\n"
    print(text_rep)
    # delete temp frames inside the folder
  shutil.rmtree("./temp_frames")
  print(f"text rep : {text_rep}")
  return text_rep

def build_videollama2_text_rep_ssbd_(ssbd,videollama2,output_path =""):
  videollama2_text_rep_x_ssbd_dataset = {}
  if os.path.exists(output_path):
    with open(output_path) as f:
      videollama2_text_rep_x_ssbd_dataset = json.load(f)
      print(f"size of videollama2_text_rep_x_ssbd_dataset: {len(videollama2_text_rep_x_ssbd_dataset)}")
  saved_rep ={}
  for video_id,video_info in videollama2_text_rep_x_ssbd_dataset.items():
    saved_rep[video_id[:-5]] = video_info["text_rep"]
  for video_id,video_info in tqdm(ssbd):
    if video_id not in videollama2_text_rep_x_ssbd_dataset:
      print(video_id)#v_ArmFlapping_02_b_02
      video_path = video_info["video_path"]
      if video_id[:-5] not in saved_rep:
        text_rep = generate_text_representation_from_video(videollama2,video_path)
        saved_rep[video_id[:-5]] = text_rep
      else: 
        text_rep = saved_rep[video_id[:-5]]
      video_info["text_rep"] = text_rep
      videollama2_text_rep_x_ssbd_dataset[video_id] = video_info
      with open(output_path,'w') as f:
        json.dump(videollama2_text_rep_x_ssbd_dataset,f,indent = 4)
        print(f"successfully saved videollama2_text_rep_x_ssbd_dataset with {len(videollama2_text_rep_x_ssbd_dataset)} samples")
  with open(output_path) as f:
    videollama2_text_rep_x_ssbd_dataset = json.load(f)
    print(f"size of videollama2_text_rep_x_ssbd_dataset: {len(videollama2_text_rep_x_ssbd_dataset)}")
if __name__ == "__main__":
  ssbd = Ssbd_DataLoader()
  videollama2 = VideoLlama2Loader()

  
  build_videollama2_text_rep_ssbd_(ssbd,videollama2,output_path = videollama2_text_rep_x_ssbd_dataset_path)
