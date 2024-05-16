import sys
import json
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
import os
from configs.configure import oops_val_videos_dir,oops_transition_times_path,oops_val_filtered_list_txt,oops_heldout_transition_times_path, oops_validation_description, uag_oops_dataset_dir
from scenedetect import detect, ContentDetector, SceneManager, VideoManager
from tqdm import tqdm

def save_data(data, file_name):

    with open(uag_oops_dataset_dir+file_name, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {file_name}")

def get_validation_filtered_video_list(oops_val_video_list_txt):
    with open(oops_val_video_list_txt, 'r') as f:
        val_video_list = f.readlines()
    val_filtered_list = [video.strip() for video in val_video_list]
    val_filtered_count = len(val_filtered_list)
    print(f"Validation filtered video count: {val_filtered_count}")
    return val_filtered_list

def get_transition_times(oops_transition_times_path):
    with open(oops_transition_times_path, 'r') as f:
        transition_times = json.load(f)
    print("transition times loaded")
    return transition_times

def get_heldout_transition_times(oops_heldout_transition_times_path):
    with open(oops_heldout_transition_times_path, 'r') as f:
        held_out_transition_times = json.load(f)
    print("held out transition times loaded")
    return held_out_transition_times
def get_validation_descriptions(oops_validation_description):
    with open(oops_validation_description, 'r') as f:
        validation_descriptions = json.load(f)
    print("validation descriptions loaded")
    return validation_descriptions
def get_failed_validation_filtered_videos_with_description():
    if not os.path.exists(uag_oops_dataset_dir+"failed_validation_filtered_videos_with_description_list.json"):
        val_filtered_list = get_validation_filtered_video_list(oops_val_filtered_list_txt)
        transition_times = get_transition_times(oops_transition_times_path)
        held_out_transition_times = get_heldout_transition_times(oops_heldout_transition_times_path)
        validation_descriptions = get_validation_descriptions(oops_validation_description)

        failed_videos_with_description_count = 0
        failed_videos_with_description_list = []
        for val_filtered_video in val_filtered_list:
            if val_filtered_video in transition_times and transition_times[val_filtered_video]["n_notfound"] == 0:
                if val_filtered_video in validation_descriptions:
                    failed_videos_with_description_count +=1.
                    failed_videos_with_description_list.append(val_filtered_video)
        print(f"Failed videos with description count: {len(failed_videos_with_description_list)}")
        save_data(failed_videos_with_description_list, "failed_validation_filtered_videos_with_description_list.json")
        return failed_videos_with_description_list
    else:
        with open(uag_oops_dataset_dir+"failed_validation_filtered_videos_with_description_list.json", 'r') as f:
            failed_validation_filtered_videos_with_descriptions = json.load(f)
        print(f"Failed videos with description count: {len(failed_validation_filtered_videos_with_descriptions)}")
        return failed_validation_filtered_videos_with_descriptions
def detect_single_scene(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold = 50))
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    scene_list = scene_manager.get_scene_list()
    if len(scene_list) == 0:
        return True
    
def get_failed_validation_filtered_videos_with_single_scene(failed_validation_filtered_videos_with_descriptions):
   
   if not os.path.exists(uag_oops_dataset_dir+"failed_validation_filtered_videos_with_single_scene.json"):
        failed_videos_with_description_single_scene = 0
        failed_validation_videos_with_single_scene = []
        for val_filtered_video in tqdm(failed_validation_filtered_videos_with_descriptions):
            video_path = oops_val_videos_dir+val_filtered_video+".mp4"
            try: 
                if detect_single_scene(video_path):
                    failed_validation_videos_with_single_scene.append(val_filtered_video)
                    failed_videos_with_description_single_scene +=1
                    print(f"current failed videos with single scene count: {failed_videos_with_description_single_scene}")

                    print(f"len of failed_validation_videos_with_single_scene: {len(failed_validation_videos_with_single_scene)}")
            except Exception as e:
                print(f"Error in video: {val_filtered_video}")
                print(e)
            save_data(failed_validation_videos_with_single_scene, "failed_validation_filtered_videos_with_single_scene.json")

        return failed_validation_videos_with_single_scene
   else:
        with open(uag_oops_dataset_dir+"failed_validation_filtered_videos_with_single_scene.json", 'r') as f:
            failed_validation_videos_with_single_scene = json.load(f)
        print(f"Failed videos with single scene count: {len(failed_validation_videos_with_single_scene)}")
        return failed_validation_videos_with_single_scene

    # return failed_validation_videos_with_single_scene
def get_description(validation_descriptions,val_filtered_video):
    description = ""
    if val_filtered_video in validation_descriptions:
        description+= validation_descriptions[val_filtered_video][0]['goal']
        description+= " "+validation_descriptions[val_filtered_video][0]['wentwrong']
    return description

def generate_dataset_v1(failed_validation_videos_with_single_scene):
    if not os.path.exists(uag_oops_dataset_dir+"uag_oops_dataset_v1.json"):
        uag_oops_dataset = {}
        held_out_transition_times = get_heldout_transition_times(oops_heldout_transition_times_path)
        transition_times = get_transition_times(oops_transition_times_path)
        validation_descriptions = get_validation_descriptions(oops_validation_description)
        for val_filtered_video in tqdm(failed_validation_videos_with_single_scene):
            print(f"val filtered video: {val_filtered_video}")
            video_path = oops_val_videos_dir+val_filtered_video+".mp4"
            # get unusual activity start time from heldout transition times
            if val_filtered_video in held_out_transition_times:
                start_time = held_out_transition_times[val_filtered_video]['t'][0]
            else: 
                start_time = None
            # get end_time = duration time from transition times
            if val_filtered_video in transition_times:
                end_time = transition_times[val_filtered_video]['len']
            else:
                end_time = None
            description = get_description(validation_descriptions,val_filtered_video)
            uag_oops_dataset[val_filtered_video] = {"video_path": video_path, "start_time": start_time, "end_time": end_time, "description": description}
            
            save_data(uag_oops_dataset, "uag_oops_dataset_v1.json")
        print(f"UAG OOPS dataset v1 count: {len(uag_oops_dataset)}")
    else:
        with open(uag_oops_dataset_dir+"uag_oops_dataset_v1.json", 'r') as f:
            uag_oops_dataset = json.load(f)
        print(f"UAG OOPS dataset v1 count: {len(uag_oops_dataset)}")
        return uag_oops_dataset
def generate_dataset_v2(failed_validation_filtered_videos_with_descriptions):
    """ traverse each failed video with description, get the video path check if there is multiple scene in the video.
    If there is multiple scene, get the first scene. the start time of unusual activity will be taken from heldout transition times. the end time will be taken from the end time of first scene.
    if there is no multiple scene, get the start time from heldout transition times and end time will be the len of the video from transition times"""
    pass
if __name__ == "__main__":
    
    failed_validation_filtered_videos_with_descriptions = get_failed_validation_filtered_videos_with_description()

    # uncomment this after the hprc job is done
    failed_validation_videos_with_single_scene = get_failed_validation_filtered_videos_with_single_scene(failed_validation_filtered_videos_with_descriptions)

    
    # with open ("/scratch/user/hasnat.md.abdullah/uag/backup/failed_validation_filtered_videos_with_single_scene.json", 'r') as f:
    #     temp_failed_validation_videos_with_single_scene = json.load(f)
    # print(f"len of failed_validation_videos_with_single_scene: {len(temp_failed_validation_videos_with_single_scene)}")

    generate_dataset_v1(failed_validation_videos_with_single_scene)

    # generate_dataset_v2(failed_validation_filtered_videos_with_descriptions)


