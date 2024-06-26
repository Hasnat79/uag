## oops data processing training
oops_train_videos_dir = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/oops_video/train/"
oops_train_filtred_list_txt = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/annotations/train_filtered.txt"
oops_train_description = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/svos/train.json"
## oops data processing validation
oops_val_videos_dir = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/oops_video/val/"
oops_val_video_list_txt = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/annotations/val.txt"
oops_val_filtered_list_txt = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/annotations/val_filtered.txt"
oops_transition_times_path = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/annotations/transition_times_fixed.json"
oops_heldout_transition_times_path = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/annotations/heldout_transition_times_fixed.json"
oops_validation_description = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/svos/val.json"

# oops training
train_videos_dir ="./oops_video/train"
train_video_list_txt = "./annotations/train.txt"
train_filtered_list_txt = "./annotations/train_filtered.txt"
uag_oops_train_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/uag_oops_dataset/uag_oops_train_dataset_v1.json"
uag_oops_instruct_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/uag_oops_dataset/uag_oops_train_instruct_dataset_v1.json"
## processed oops dataset:  uag oops dataset v1 path
uag_oops_dataset_dir = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/uag_oops_dataset/"
# uag_oops_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/uag_oops_dataset/uag_oops_dataset_v1.json"
# it's validation
uag_oops_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/uag_oops_dataset/uag_oops_dataset_v1_start_time_fixed.json"

#video_llama2 predictions over uag oops dataset v1
video_llama2_pred_x_uag_oops_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/results/video_llama2_pred_uag_oops_dataset_v1.json"
video_llama2_pred_x_uag_oops_dataset_null_fixed_path = "/scratch/user/hasnat.md.abdullah/uag/results/video_llama2_pred_uag_oops_dataset_v1_null_fixed.json"

#video_chatgpt predictions over uag oops dataset v1
video_chatgpt_pred_x_uag_oops_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/results/video_chatgpt_pred_uag_oops_dataset_v1.json"
video_chatgpt_pred_x_uag_oops_dataset_null_fixed_path = "/scratch/user/hasnat.md.abdullah/uag/results/video_chatgpt_pred_uag_oops_dataset_v1_null_fixed.json"

# videochat 2 predictions path over uag oops dataset v1
videochat2_pred_uag_oops_v1_path = "/scratch/user/hasnat.md.abdullah/uag/results/videochat2_pred_uag_oops_dataset_v1_null_fixed.json"

# llama3 predictions over blip2_text_rep_x_uag_oops_dataset_v1
llama3_model_id = "/scratch/user/hasnat.md.abdullah/uag/scripts/meta-llama/Meta-Llama-3-8B-Instruct"
llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path = "/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1.json"
llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_path_null_fixed = "/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_blip2_text_rep_x_uag_oops_dataset_v1_null_fixed.json"
# llama3 predictions over blip2_text_rep_x_ssbd_dataset
llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_blip2_text_rep_x_ssbd_dataset.json"
llama3_pred_x_blip2_text_rep_x_ssbd_dataset_path_null_fixed = "/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_blip2_text_rep_x_ssbd_dataset_null_fixed.json"
#llama3 predictions over videollama2_text_rep_x_ssbd_dataset
llama3_pred_x_videollama2_text_rep_x_ssbd_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_videollama2_text_rep_x_ssbd_dataset.json"
llama3_pred_x_videollama2_text_rep_x_ssbd_dataset_path_null_fixed = "/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_videollama2_text_rep_x_ssbd_dataset_null_fixed.json"
llama3_pred_x_videollama2_text_rep_x_uag_oops_dataset_v1_path = "/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_videollama2_text_rep_x_uag_oops_dataset_v1.json"
llama3_pred_x_videollama2_text_rep_x_uag_oops_dataset_v1_path_null_fixed = "/scratch/user/hasnat.md.abdullah/uag/results/llama3_pred_x_videollama2_text_rep_x_uag_oops_dataset_v1_null_fixed.json"
#ssbd dataset
ssbd_data_path = "/scratch/user/hasnat.md.abdullah/uag/data/ssbd/ssbd_test_list_fixed.json"

# funQA dataset
funqa_data_path = "/scratch/user/hasnat.md.abdullah/uag/data/FunQA/funqa_dataset.json"
funqa_test_annotations_h1_path = "/scratch/user/hasnat.md.abdullah/uag/data/FunQA/annotation_with_ID/funqa_test.json"
funqa_test_humor_video_dir = "/scratch/user/hasnat.md.abdullah/uag/data/FunQA/test/test_humor/"
# funQA x videollama
videollama_pred_x_funqa_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/results/videollama_pred_x_funqa_dataset.json"
videollama_pred_x_funqa_dataset_null_fixed_path = "/scratch/user/hasnat.md.abdullah/uag/results/videollama_pred_x_funqa_dataset_null_fixed.json"
#funQA x videochat2
videochat2_pred_x_funqa_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/results/videochat2_pred_x_funqa_dataset.json"
#funQA x videochatgpt
video_chatgpt_pred_x_funqa_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/results/video_chatgpt_pred_x_funqa_dataset.json"

# finetuned videollama (on oops instruct)
finetuned_videollama_pred_x_uag_oops_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/results/finetuned_videollama_pred_x_uag_oops_dataset_v1.json"
finetuned_videollama_pred_x_uag_oops_dataset_null_fixed_path = "/scratch/user/hasnat.md.abdullah/uag/results/finetuned_videollama_pred_x_uag_oops_dataset_v1_null_fixed.json"

# finetuned videollama (on oops instruct) over ssbd dataset
finetuned_videollama_pred_x_ssbd_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/results/finetuned_videollama_pred_x_ssbd_dataset.json"
finetuned_videollama_pred_x_ssbd_dataset_null_fixed_path = "/scratch/user/hasnat.md.abdullah/uag/results/finetuned_videollama_pred_x_ssbd_dataset_null_fixed.json"

# video_llama2 predictions over ssbd dataset
video_llama2_pred_x_ssbd_result_path = "/scratch/user/hasnat.md.abdullah/uag/results/video_llama2_pred_ssbd_dataset.json"
video_llama2_pred_x_ssbd_result_null_fixed_path = "/scratch/user/hasnat.md.abdullah/uag/results/video_llama2_pred_ssbd_dataset_null_fixed.json"

# video_chatgpt predictions over ssbd dataset
video_chatgpt_pred_x_ssbd_result_path = "/scratch/user/hasnat.md.abdullah/uag/results/video_chatgpt_pred_ssbd_dataset.json"
video_chatgpt_pred_x_ssbd_result_null_fixed_path = "/scratch/user/hasnat.md.abdullah/uag/results/video_chatgpt_pred_ssbd_dataset_null_fixed.json"
# results directory
results_dir = "/scratch/user/hasnat.md.abdullah/uag/results/"


# videochat gpt experiment related paths
llava_lightning_7b_v1_1_path = "/scratch/user/hasnat.md.abdullah/uag/foundation_models/Video-ChatGPT/LLaVA-7B-Lightening-v1-1"
video_chatgpt_weights_path = "/scratch/user/hasnat.md.abdullah/uag/foundation_models/Video-ChatGPT/video_chatgpt-7B.bin"

# configs for text_representation_builders
blip2_text_rep_x_oops_dataset_v1_path =  "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/uag_oops_dataset/blip2_text_rep_x_oops_dataset_v1.json"
temp_video_frames_dir = "./temp_frames"
blip2_text_rep_x_ssbd_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/data/ssbd/blip2_text_rep_x_ssbd_dataset.json"

videollama2_text_rep_x_ssbd_dataset_path = "/scratch/user/hasnat.md.abdullah/uag/data/ssbd/videollama2_text_rep_x_ssbd_dataset.json"
videollama2_text_rep_x_oops_dataset_v1_path = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/uag_oops_dataset/videollama2_text_rep_x_oops_dataset_v1.json"
