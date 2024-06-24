import cv2
import os

def load_frame(frame_path):
    '''loads a frame from a given frame path
    '''
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def save_temporary_frames_from_video(video_path="", output_path = "./temp_frames"):
    '''from a video path it saves the first frame and the rest of the frames at 1fps in a temp folder
    '''

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # video_path = video_path.replace("'",'"')
    # video_path = video_path.replace("'","\'")        
    # escaped_video_path = shlex.quote(video_path)
    # print(escaped_video_path)
    
    # extract frames from video: 1fps -- 0th second ~ frame 1

    os.system(f"""ffmpeg -i "{video_path}" -vf fps=1 {output_path}/%d.png""")
