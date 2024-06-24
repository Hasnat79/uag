import json 
import sys
import os 

sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path_null_fixed,llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path_null_fixed

def process_sample(sample):
    text_representation = sample["text_rep"]
    comments = ""
    accuracy = ""
    return {"text_representation": text_representation,  "accuracy": accuracy, "comments": comments}
if __name__ == "__main__":
  # load llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path_null_fixed
  with open(llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path_null_fixed) as f:
    llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1 = json.load(f)
  # check null values count
  null_count = 0
  for video_id, sample  in llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1.items():
    if sample['pred_start'] is None:
      null_count += 1
  print(f"null values count in llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1: {null_count}")#59

  # take 50 samples with null values, and generate a dictionary with text_represerntation, accuracy and comments
  llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_sampled = {}
  for video_id, sample in llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1.items():
    if sample['pred_start'] is None:
      # generate a dictionary with text_represerntation, accuracy and comments
      sample = process_sample(sample)
      llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_sampled[video_id] = sample
    if len(llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_sampled) == 50:
      break

  if not os.path.exists("llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_sampled_50.json"):
    with open("llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_sampled_50.json", "w") as f:
      json.dump(llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_sampled, f, indent=4)

  # load llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path_null_fixed
  with open(llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path_null_fixed) as f:
    llama3_pred_x_blip2_text_rep_x_ssbd_dataset = json.load(f)
  # check null values count
  null_count = 0
  for video_id, sample in llama3_pred_x_blip2_text_rep_x_ssbd_dataset.items():
    if sample['pred_start'] is None:
      null_count += 1
  print(f"null values count in llama3_pred_x_blip2_text_rep_x_ssbd_dataset: {null_count}")#29
  # take 29 samples with null values, and generate a dictionary with text_represerntation, accuracy and comments
  llama3_pred_x_blip2_text_rep_x_ssbd_dataset_sampled = {}
  for video_id, sample in llama3_pred_x_blip2_text_rep_x_ssbd_dataset.items():
    if sample['pred_start'] is None:
      # generate a dictionary with text_represerntation, accuracy and comments
      sample = process_sample(sample)
      llama3_pred_x_blip2_text_rep_x_ssbd_dataset_sampled[video_id] = sample
    if len(llama3_pred_x_blip2_text_rep_x_ssbd_dataset_sampled) == 29:
      break
  if not os.path.exists("llama3_pred_x_blip2_text_rep_x_ssbd_dataset_sampled_29.json"):
    with open("llama3_pred_x_blip2_text_rep_x_ssbd_dataset_sampled_29.json", "w") as f:
      json.dump(llama3_pred_x_blip2_text_rep_x_ssbd_dataset_sampled, f, indent=4)
