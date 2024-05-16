
import os
import ffmpeg
from scenedetect import detect, ContentDetector, SceneManager, VideoManager
video_path_1= "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/oops_video/val/Don't Get Zapped - Throwback Thursday (August 2017)102.mp4"
video_path_2 ="/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/oops_video/val/Don't Get Zapped - Throwback Thursday (August 2017)101.mp4"
video_path_3 ="/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/oops_video/val/Don't Get Zapped - Throwback Thursday (August 2017)104.mp4"
video_path_4 ="/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/oops_video/val/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)21.mp4"
video_manager = VideoManager([video_path_4])
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold = 50))
video_manager.set_downscale_factor()
video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)
scene_list = scene_manager.get_scene_list()

print(scene_list)
print(scene_list[0][1])
start_time = scene_list[0][0].get_timecode(precision=0)
end_time = scene_list[0][1].get_timecode(precision =0)
print(start_time, end_time)
print(len(scene_list))
output_path = "processed_34FunnyKidNominees-FailArmyHallOfFame(May2017)21.mp4"


ffmpeg.input(video_path_4, ss=start_time).output(output_path, to=end_time).run()
