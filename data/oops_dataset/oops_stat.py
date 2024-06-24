import json
import os
import statistics
import sys
import json
import os
import statistics
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import oops_val_videos_dir,oops_val_video_list_txt,oops_transition_times_path,oops_val_filtered_list_txt,oops_heldout_transition_times_path
# from scenedetect import detect, ContentDetector, SceneManager, VideoManager

# exit()


val_videos_dir = oops_val_videos_dir
val_video_list_txt = oops_val_video_list_txt
val_filtered_list_txt = oops_val_filtered_list_txt
transition_times_path = oops_transition_times_path
train_videos_dir ="./oops_video/train"
train_video_list_txt = "./annotations/train.txt"
train_filtered_list_txt = "./annotations/train_filtered.txt"

oops_validation_description = "./val_descriptions.json"
oops_train_description = "./train_descriptions.json"
# check how many training files are there in train video txt list
train_video_list = []
with open(train_video_list_txt, 'r') as file:
    train_video_list = file.readlines()
train_video_list = [video.strip() for video in train_video_list]
train_video_count = len(train_video_list)
print("Total training files:", train_video_count)

# # check how many train videos are actually available
# total_available_train_videos = 0
# for train_video in train_video_list:
#     # print(train_video)
#     train_video_path = train_videos_dir+"/"+train_video+".mp4"
#     if os.path.exists(train_video_path):
#         total_available_train_videos+=1
#     else:
#         print("Train video path does not exist")
# print(f"Total training videos available: {total_available_train_videos}")

# check how many training videos are there in train filtered list
train_filtered_list = []
with open(train_filtered_list_txt, 'r') as file:
    train_filtered_list = file.readlines()
train_filtered_list = [video.strip() for video in train_filtered_list]
train_filtered_count = len(train_filtered_list)
print("Total videos in train filtered list:", train_filtered_count)

# # total trained filtered videos
# total_available_train_filtered_videos = 0
# for train_filtered_video in train_filtered_list:
#     train_filtered_video_path = train_videos_dir + "/" + train_filtered_video + ".mp4"
#     if os.path.exists(train_filtered_video_path):
#         total_available_train_filtered_videos += 1
#     else:
#         print("Train filtered video path does not exist")
# print(f"Total trained filtered videos available: {total_available_train_filtered_videos}")
# print("-----------------------------------------------")
# check how many validation files are there in valvideo txt list
# val_video_list = []
# with open(val_video_list_txt, 'r') as file:
#     val_video_list = file.readlines()
# val_video_list = [video.strip() for video in val_video_list]

# val_video_count = len(val_video_list)
# print("Total validation files:", val_video_count)
# # check how many videos in validation file are available
# total_available_val_videos = 0
# for val_video in val_video_list:
#     val_video_path = val_videos_dir + "/" + val_video + ".mp4"
#     if os.path.exists(val_video_path):
#         total_available_val_videos += 1
#     else:
#         print("Validation video path does not exist")
# print(f"Total validation videos available: {total_available_val_videos}")



# check how many failed training filtered videos
# Load transition times from JSON file
with open(transition_times_path, 'r') as file:
    transition_times = json.load(file)
train_filtered_list = []
with open(train_filtered_list_txt, 'r') as file:
    train_filtered_list = file.readlines()
train_filtered_list = [video.strip() for video in train_filtered_list]
train_filtered_count = len(train_filtered_list)
# Count the number of failed and not failed training filtered videos
failed_videos = 0
not_failed_videos = 0

for val_filtered_video in train_filtered_list:
    if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
        failed_videos += 1
    else:
        not_failed_videos += 1
print(f"Total failed training filtered videos: {failed_videos}")#3855


# check how many failed filtered training videos with descriptions
with open(oops_train_description, 'r') as file:
    train_description = json.load(file)
failed_videos_with_description = 0
for train_filtered_video in train_filtered_list:
    if train_filtered_video in transition_times and transition_times[train_filtered_video]["n_notfound"] == 0:
        if train_filtered_video in train_description:
            print(f"train_filtered_video: {train_filtered_video}")
            failed_videos_with_description += 1
print("Total failed filtered training videos with descriptions:", failed_videos_with_description)#3846
# check how many failed filtered training videos with descriptions are available
# total_available_failed_filtered_train_videos_with_descriptions = 0
# for train_filtered_video in train_filtered_list:
#     if train_filtered_video in transition_times and transition_times[train_filtered_video]["n_notfound"] == 0:
#         if train_filtered_video in train_description:
#             train_filtered_video_path = train_videos_dir + "/" + train_filtered_video + ".mp4"
#             if os.path.exists(train_filtered_video_path):
#                 total_available_failed_filtered_train_videos_with_descriptions += 1
#             else:
#                 print("Train filtered video path does not exist")
# print(f"Total available failed filtered training videos with descriptions: {total_available_failed_filtered_train_videos_with_descriptions}")#3846
print("-----------------------------------------------")
exit()
#------------------------------------------

# check how many videos are there in val filtered list
val_filtered_list = []
with open(val_filtered_list_txt, 'r') as file:
    val_filtered_list = file.readlines()
val_filtered_list = [video.strip() for video in val_filtered_list]
val_filtered_count = len(val_filtered_list)
print("Total videos in val filtered list:", val_filtered_count)

# check how many validation filtered videos are available
total_available_val_filtered_videos = 0
for val_filtered_video in val_filtered_list:
    val_filtered_video_path = val_videos_dir + "/" + val_filtered_video + ".mp4"
    if os.path.exists(val_filtered_video_path):
        total_available_val_filtered_videos += 1
    else:
        print("Validation filtered video path does not exist")
print(f"Total validation filtered videos available: {total_available_val_filtered_videos}")
print("------------------------------------------")
#check how many failed / not failed videos are there in validation filtered videos
# Load transition times from JSON file
with open(transition_times_path, 'r') as file:
    transition_times = json.load(file)

# Count the number of failed and not failed validation filtered videos
failed_videos = 0
not_failed_videos = 0

for val_filtered_video in val_filtered_list:
    if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
        failed_videos += 1
    else:
        not_failed_videos += 1

total_val_filtered_videos = len(val_filtered_list)

# Calculate the percentages
failed_percentage = (failed_videos / total_val_filtered_videos) * 100
not_failed_percentage = (not_failed_videos / total_val_filtered_videos) * 100

print("Failed validation filtered videos:", failed_videos)
print("Not failed validation filtered videos:", not_failed_videos)
print("Percentage of failed validation filtered videos:", failed_percentage)
print("Percentage of not failed validation filtered videos:", not_failed_percentage)
print("----------------------------------------------")
# Calculate the average duration of failed validation filtered videos
failed_videos = 0
total_failed_duration = 0
for val_filtered_video in val_filtered_list:
    if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
        failed_videos += 1
        # Assuming the duration of each video is stored in the transition_times dictionary
        total_failed_duration += transition_times[val_filtered_video]["len"]
# print(f"failed videos : {failed_videos}")
# print(f"total failed duration: {total_failed_duration}")
average_failed_duration = total_failed_duration / failed_videos
print("Average duration of failed validation filtered videos:", average_failed_duration, "seconds")
# Calculate the standard deviation of failed validation filtered videos duration
failed_durations = []
for val_filtered_video in val_filtered_list:
    if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
        failed_durations.append(transition_times[val_filtered_video]["len"])
std_deviation = statistics.stdev(failed_durations)
print("Standard deviation of failed validation filtered videos duration:", std_deviation)
print("-----------------------------------------------")

# Calculate the average duration of failed annotation videos
failed_annotation_videos = 0
total_failed_annotation_duration = 0
for val_filtered_video in val_filtered_list:
    if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
        if "t" in transition_times[val_filtered_video]:
            start_times = transition_times[val_filtered_video]["t"]
            if len(start_times) >= 2:
                first_start_time = start_times[0]
                last_start_time = start_times[-1]
                annotation_duration = last_start_time - first_start_time
                total_failed_annotation_duration += annotation_duration
                failed_annotation_videos += 1
print(f"failed annotation videos: {failed_annotation_videos}")
average_failed_annotation_duration = total_failed_annotation_duration / failed_annotation_videos
print("Average duration of failed annotation videos:", average_failed_annotation_duration, "seconds")
# Calculate the standard deviation of failed annotation videos duration
failed_annotation_durations = []
for val_filtered_video in val_filtered_list:
    if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
        if "t" in transition_times[val_filtered_video]:
            start_times = transition_times[val_filtered_video]["t"]
            if len(start_times) >= 2:
                first_start_time = start_times[0]
                last_start_time = start_times[-1]
                annotation_duration = last_start_time - first_start_time
                failed_annotation_durations.append(annotation_duration)
std_deviation_annotation = statistics.stdev(failed_annotation_durations)
print("Standard deviation of failed annotation videos duration:", std_deviation_annotation)
print("----------------------------------------------------------------")
with open(oops_validation_description, 'r') as file:
    validation_description = json.load(file)

# Calculate how many filtered validation failed videos are in heldout transition times set
with open(oops_heldout_transition_times_path,'r') as f: 
    heldout_transition_times = json.load(f)
heldout_failed_videos = 0
for val_filtered_video in val_filtered_list:
    if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
        if val_filtered_video in heldout_transition_times:
            heldout_failed_videos += 1
print("Total filtered validation failed videos in heldout transition times set:", heldout_failed_videos)

print("-------------------------------------------------------")
# Calculate how many validation videos have description available
videos_with_description = 0
for val_video in val_video_list:
    if val_video in validation_description:
        videos_with_description += 1
print("Total validation videos with description:", videos_with_description)

# Calculate how many failed videos have validation descriptions
failed_videos_with_description = 0
for val_filtered_video in val_filtered_list:
    if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
        if val_filtered_video in validation_description:
            failed_videos_with_description += 1
print("Total failed filtered validation videos with validation descriptions:", failed_videos_with_description)

print("--------------------------------------------------------")
filtered_videos_with_description = 0
for val_filtered_video in val_filtered_list:
    if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
        
        if val_filtered_video in validation_description:
            if "t" in heldout_transition_times[val_filtered_video] and "len" in transition_times[val_filtered_video]:
                start_time = heldout_transition_times[val_filtered_video]["t"][0]
                video_length = transition_times[val_filtered_video]["len"]

                if start_time < 5 and video_length >= 10.0:
                    # print("beep")
                    filtered_videos_with_description += 1
print("Total filtered validation videos with descriptions, start time < 5s, and video length > 15s:", filtered_videos_with_description)
print("--------------------------------------------------")
# Calculate how many validation filtered failed videos with description have multiple scenes inside the videos
# def detect_multiple_scenes(video_path):
#     video_manager = VideoManager([video_path])
#     scene_manager = SceneManager()
#     scene_manager.add_detector(ContentDetector(threshold = 50))
#     video_manager.set_downscale_factor()
#     video_manager.start()
#     scene_manager.detect_scenes(frame_source=video_manager)
#     scene_list = scene_manager.get_scene_list()
#     if len(scene_list) == 0:
#         return True
# failed_videos_with_description_multiple_scenes = 0
# failed_videos_with_description_single_scene = 0
# i=0
# for val_filtered_video in val_filtered_list:
#     if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
#         if val_filtered_video in validation_description:
#             # Assuming you have a function or library to detect multiple scenes in a video
#             video_path = val_videos_dir + "/" + val_filtered_video + ".mp4"            
#             print(video_path)
#             print(f"i: {i}")
#             try: 
#                 if detect_multiple_scenes(video_path):
#                     # failed_videos_with_description_multiple_scenes += 1
#                     # print(f"current failed videos with description and multiple scenes: {failed_videos_with_description_multiple_scenes}")
#                     failed_videos_with_description_single_scene+=1
#                     print(f"current failed videos with description and single scene: {failed_videos_with_description_single_scene}")

#             except Exception as e: 
#                 print(e)
#                 print(video_path)
#                 print(f"i: {i}")

#             i+=1
#             #     print(video_path)
#             # if failed_videos_with_description_multiple_scenes > 2: 
#             #     break
# print("Total failed filtered validation videos with description and multiple scenes:", failed_videos_with_description_multiple_scenes)
