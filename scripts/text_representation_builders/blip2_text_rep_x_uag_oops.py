import os
import sys
import shutil
import cv2
import torch
import nltk
from tqdm import tqdm
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)

import json
from configs.configure import uag_oops_dataset_path, blip2_text_rep_x_oops_dataset_v1_path,oops_val_videos_dir,temp_video_frames_dir
from transformers import Blip2Processor, Blip2ForConditionalGeneration

@memory.cache
def initialize_blip2_model():
    '''
    initializes the blip2 model
    '''
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
"Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16) 
    model.to("cuda:0")
    return processor,model

def vqa_captioner(frame,question=""):
    '''
    given a frame / image it returns the captioned text using blip2 model utilizing visual question answering technique
    '''
    # question = "Question: What is happening in the image? Answer: "
    inputs = processor(frame, return_tensors="pt").to("cuda",torch.float16)
    out = model.generate(
    **inputs,
    num_beams=5,
    min_length=24,
    max_length=72
    # max_new_tokens=200,
    # # num_beams=5,
    # # temperature=0.7,
    # # repetition_penalty=1.3,
    # min_length=5,
    # max_length=10
)   
    generated_text = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    
    print(f"Generated text: {generated_text}\n\n")

    return generated_text

def save_temporary_frames_from_video(video_path="", output_path = "./temp_frames"):
    '''from a video path it saves the first frame and the rest of the frames at 1fps in a temp folder
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # extract frames from video: 1fps -- 0th second ~ frame 1
    os.system(f"ffmpeg -i '{video_path}' -vf fps=1 {output_path}/%d.png")
    # save frames in output_path

def load_frame(frame_path):
    '''loads a frame from a given frame path
    '''
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def generate_text_representation_from_video(video_path=""):
    ''' from a video path it saves all the frames at 1fp in a temp folder and then generates text representation from each frame using captioner function and returns the concatenated text representation of all the frames.
    '''
    save_temporary_frames_from_video(video_path,output_path = temp_video_frames_dir)
    video_frames = [os.path.join(temp_video_frames_dir, f) for f in os.listdir(temp_video_frames_dir) if f.endswith('.png')]
    text_rep = ""
    
    for i,frame in enumerate(video_frames):
        frame_image = load_frame(frame)
        text_rep += f"{i}.0s: "
        text_rep += vqa_captioner(frame_image)+".\n"
        
    # delete temp frames inside the folder
    shutil.rmtree("./temp_frames")

    return text_rep
    

def build_blip2_text_rep_x_oops_dataset_v1(dataset = uag_oops_dataset_path, output_path = blip2_text_rep_x_oops_dataset_v1_path):
    blip2_text_rep_x_oops_dataset_v1 = {}
    
    if os.path.exists(output_path):
        with open(output_path,'r') as f:
            blip2_text_rep_x_oops_dataset_v1 = json.load(f)
        print(f"Loaded blip2_text_rep_x_oops_dataset_v1 with {len(blip2_text_rep_x_oops_dataset_v1)} samples")
    
    for video_id, video_info in tqdm(dataset.items()):
        if video_id not in blip2_text_rep_x_oops_dataset_v1:
            video_path = video_info ["video_path"]
            print(f"video_path: {video_path}")
            text_rep = generate_text_representation_from_video(video_path)
            video_info["text_rep"] = text_rep
            blip2_text_rep_x_oops_dataset_v1[video_id] = video_info
            
        # with open(output_path, 'w') as f:
        #     json.dump(blip2_text_rep_x_oops_dataset_v1, f,indent=4)
        #     print(f"succesfully saved blip2_text_rep_x_oops_dataset_v1 with {len(blip2_text_rep_x_oops_dataset_v1)} samples")
    #verifying the result
  
if __name__ == "__main__":
    with open(uag_oops_dataset_path, 'r') as f:
        uag_oops_dataset = json.load(f)
    print(f"UAG oops dataset loaded with {len(uag_oops_dataset)} videos")
    
    ## intitialize the captioner model
    #global model
    processor,model = initialize_blip2_model()

    build_blip2_text_rep_x_oops_dataset_v1(uag_oops_dataset, blip2_text_rep_x_oops_dataset_v1_path)
