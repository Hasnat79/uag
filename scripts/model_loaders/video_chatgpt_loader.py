
import os
from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/foundation_models/Video-ChatGPT/")
from configs.configure import llava_lightning_7b_v1_1_path, video_chatgpt_weights_path
from video_chatgpt.eval.model_utils import initialize_model, load_video
from video_chatgpt.inference import video_chatgpt_infer



class VideoChatGPTLoader():

    def __init__(self):
        # args = self.parse_args()
        self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len = initialize_model(llava_lightning_7b_v1_1_path, video_chatgpt_weights_path)
    # @memory.cache
    # def init_model(self, model_name, projection_path):
    #     model, vision_tower, tokenizer, image_processor, video_token_len = \
    #         initialize_model(model_name, projection_path)
    #     print("Model initialized")
    #     return model, vision_tower, tokenizer, image_processor, video_token_len
    def __call__(self):
        return self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len
    def get_video_frames (self, video_path, num_frm=100):
        print(f"Loading video from {video_path}")
        return load_video(video_path,num_frm=num_frm)
      
    def infer(self,video_frames,query):
        conv_mode = "video-chatgpt_v1"
        answer = video_chatgpt_infer(video_frames, query, conv_mode, self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len)
        print(f"Answer: {answer}")
        return answer
    

if __name__ == "__main__":
    video_chatgpt = VideoChatGPTLoader()
    print("Model loaded")
    video_path = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/oops_video/val/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0.mp4"
    video_frames = video_chatgpt.get_video_frames(video_path)
    query = "What is happening in this video?"
    answer = video_chatgpt.infer(video_frames, query)
    print(f"Answer: {answer}")
    # Model loaded
    # Answer: The person is arm flapping.
