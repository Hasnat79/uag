import sys
import json
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
import os
from tqdm import tqdm
from configs.configure import oops_train_videos_dir,oops_transition_times_path,oops_train_filtred_list_txt,oops_train_description, uag_oops_dataset_dir, uag_oops_train_dataset_path


def save_data(data, file_name):
    with open(uag_oops_dataset_dir+file_name, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {file_name}")
def get_train_filtered_video_list(oops_train_filtred_list_txt):
    with open(oops_train_filtred_list_txt, 'r') as f:
        train_filtered_list = f.readlines()
    train_filtered_list = [video.strip() for video in train_filtered_list]
    train_filtered_count = len(train_filtered_list)
    print(f"Train filtered video count: {train_filtered_count}")
    return train_filtered_list
def get_transition_times(oops_transition_times_path):
    with open(oops_transition_times_path, 'r') as f:
        transition_times = json.load(f)
    print("transition times loaded")
    return transition_times
def get_train_descriptions(oops_train_description):
    with open(oops_train_description, 'r') as f:
        train_descriptions = json.load(f)
    print("train descriptions loaded")
    return train_descriptions
def get_failed_train_filtered_videos_with_description():
    
    train_filtered_list = get_train_filtered_video_list(oops_train_filtred_list_txt)
    transition_times = get_transition_times(oops_transition_times_path)
    train_descriptions = get_train_descriptions(oops_train_description)

    failed_videos_with_description_count = 0
    failed_videos_with_description_list = []
    for train_filtered_video in train_filtered_list:
        if train_filtered_video in transition_times and transition_times[train_filtered_video]["n_notfound"] == 0:
            if train_filtered_video in train_descriptions:
                failed_videos_with_description_count +=1.
                failed_videos_with_description_list.append(train_filtered_video)
    print(f"Failed train videos with description count: {len(failed_videos_with_description_list)}")

    return failed_videos_with_description_list


def build_uag_oops_train_dataset():
    if not os.path.exists(uag_oops_dataset_dir+"uag_oops_train_dataset_v1.json"):
        failed_train_filtered_videos_with_description = get_failed_train_filtered_videos_with_description()
        print(f"Failed train videos with description count: {len(failed_train_filtered_videos_with_description)}")
        transition_times = get_transition_times(oops_transition_times_path)
        train_description = get_train_descriptions(oops_train_description)
        uag_oops_train_dataset = {}
        for video in tqdm(failed_train_filtered_videos_with_description):
            sample = {}
            # print(video)

            # extract the description from train description
            for desc in train_description[video]:
                if desc['goal'] != "Don't know" and desc['wentwrong'] != "Don't know":
                    sample["video_path"] = oops_train_videos_dir + video+".mp4"
                    sample["description"] = desc['goal'] + " " + desc['wentwrong']
                    if video in transition_times:
                        start_time = transition_times[video]['t'][0]
                        end_time = transition_times[video]['len']
                    else:
                        start_time = None
                        end_time = None
                    sample["start_time"] = start_time
                    sample["end_time"] = end_time
                    uag_oops_train_dataset[video] = sample
                    break
            save_data(uag_oops_train_dataset, "uag_oops_train_dataset_v1.json")
    # check final train dataset size
    with open(uag_oops_dataset_dir+"uag_oops_train_dataset_v1.json", 'r') as f:
        uag_oops_train_dataset = json.load(f)
    print(f"Final train dataset size: {len(uag_oops_train_dataset)}")        
# Final train dataset size: 3778
           
                            

def build_uag_oops_train_instruct_dataset():
    with open(uag_oops_train_dataset_path, 'r') as f:
        uag_oops_train_dataset = json.load(f)
    instruct_set_list = []
    if not os.path.exists(uag_oops_dataset_dir+"uag_oops_train_instruct_dataset_v1.json"):
      for video_id, video_info in tqdm(uag_oops_train_dataset.items()):
          sample = {}
          sample['video'] = video_info['video_path'].split("/")[-1]
          qa = {}
          qa['q'] = f"""Find the start time and end time of the query below from the video.
          Query: {video_info['description']}"""
          qa['a'] = f"""start_time: {video_info['start_time']}, end_time: {video_info['end_time']}"""
          sample["QA"] = [qa]
          instruct_set_list.append(sample)
          save_data(instruct_set_list, "uag_oops_train_instruct_dataset_v1.json")
    # check final train instruct dataset size
    with open(uag_oops_dataset_dir+"uag_oops_train_instruct_dataset_v1.json", 'r') as f:
        instruct_set_list = json.load(f)
    print(f"Final train instruct dataset size: {len(instruct_set_list)}")
      

if __name__ == "__main__":
    # get_failed_train_filtered_videos_with_description()
    build_uag_oops_train_dataset()
    build_uag_oops_train_instruct_dataset()
