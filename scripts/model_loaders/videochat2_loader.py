
import os
from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/foundation_models/Ask-Anything/video_chat2/")
from conversation import Chat
from tqdm import tqdm
# videochat
import torch
from utils.config import Config
from utils.easydict import EasyDict
from models.videochat2_it import VideoChat2_it
from peft import get_peft_model, LoraConfig, TaskType



class VideoChat2Loader():
    def __init__(self):
        self.chat = self.init_model()
    def __call__(self):
        return self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len
    @memory.cache
    def init_model():
        print('Initializing VideoChat')
        config_file = "/scratch/user/hasnat.md.abdullah/uag/configs/video_chat2_config.json"
        cfg = Config.from_file(config_file)
        cfg.model.vision_encoder.num_frames = 4
        # cfg.model.videochat2_model_path = ""
        # cfg.model.debug = True
        model = VideoChat2_it(config=cfg.model)
        model = model.to(torch.device(cfg.device))

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=16, lora_alpha=32, lora_dropout=0.
        )
        model.llama_model = get_peft_model(model.llama_model, peft_config)
        state_dict = torch.load("/scratch/user/hasnat.md.abdullah/uag/foundation_models/Ask-Anything/video_chat2/videochat2_7b_stage3.pth", "cpu")
        if 'model' in state_dict.keys():
            msg = model.load_state_dict(state_dict['model'], strict=False)
        else:
            msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        model = model.eval()

        chat = Chat(model)
        print('Initialization Finished')
        return chat
    def infer(self,video_path, user_message):
        num_beams = 1
        temperature = 0.1
        num_segments = 4 

        chat_state = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
        img_list = []
        llm_message, img_list, chat_state = self.chat.upload_video(video_path, chat_state, img_list, num_segments)
        chat_state =  self.chat.ask(user_message, chat_state)
        llm_message,llm_message_token, chat_state = self.chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
        llm_message = llm_message.replace("<s>", "")

        # reset 
        if chat_state is not None:
            chat_state.messages = []
        if img_list is not None:
            img_list = []
        print(f"video_path: {video_path}")
        print(f"prompt: {user_message}")
        print(f"Answer: {llm_message}")
        return llm_message
    
if __name__ == "__main__":
    video_chat2 = VideoChat2Loader()
    print("Model loaded")
    video_path = "/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/oops_video/val/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0.mp4"
    user_message = "What is happening in this video?"
    answer = video_chat2.infer(video_path, user_message)
    print(f"Answer: {answer}")
    # Model loaded
    # Answer: The person is arm flapping.
