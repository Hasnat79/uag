import json 
import sys
import os 

sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import blip2_text_rep_x_oops_dataset_v1_path, blip2_text_rep_x_ssbd_dataset_path


def process_sample(sample):
    text_representation = sample["text_rep"]
    comments = ""
    accuracy = ""
    context = ""

    return {"text_representation": text_representation,  "accuracy": accuracy, "context": context,"comments": comments}
if __name__ == "__main__":
    # generate a dictionary with 50 text_represerntation and comments for manual content analysis: uag oops dataset v1
    with open (blip2_text_rep_x_oops_dataset_v1_path) as f:
        blip2_text_rep_x_oops_dataset_v1 = json.load(f)

    # sample 50 video_representation from blip2_text_rep_x_oops_dataset_v1 and generate a dictionary and save it in current folder
    blip2_text_rep_x_oops_dataset_v1_sampled = {}
    for video_id, sample in blip2_text_rep_x_oops_dataset_v1.items():
        # generate a dictionary with text_represerntation and comments
        sample = process_sample(sample)
        blip2_text_rep_x_oops_dataset_v1_sampled[video_id] = sample
        if len(blip2_text_rep_x_oops_dataset_v1_sampled) == 50:
            break
      
    if not os.path.exists("blip2_text_rep_x_oops_dataset_v1_sampled_50.json"):
      with open("blip2_text_rep_x_oops_dataset_v1_sampled_50.json", "w") as f:
          json.dump(blip2_text_rep_x_oops_dataset_v1_sampled, f, indent=4)


# generate a dictionary with 30 text_represerntation and comments for manual content analysis: ssbd dataset
    with open(blip2_text_rep_x_ssbd_dataset_path) as f:
        blip2_text_rep_x_ssbd_dataset = json.load(f)
    # sample 30 video_representation from blip2_text_rep_x_ssbd_dataset and generate a dictionary and save it in current folder
    blip2_text_rep_x_ssbd_dataset_sampled = {}
    for video_id, sample in blip2_text_rep_x_ssbd_dataset.items():
        # generate a dictionary with text_represerntation and comments
        sample = process_sample(sample)
        blip2_text_rep_x_ssbd_dataset_sampled[video_id] = sample
        if len(blip2_text_rep_x_ssbd_dataset_sampled) == 30:
            break
    if not os.path.exists("blip2_text_rep_x_ssbd_dataset_sampled_30.json"):
      with open("blip2_text_rep_x_ssbd_dataset_sampled_30.json", "w") as f:
          json.dump(blip2_text_rep_x_ssbd_dataset_sampled, f, indent=4)
