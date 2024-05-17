import json
import torch
import os
import re
from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
# import gradio as gr
# from gradio.themes.utils import colors, fonts, sizes
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/foundation_models/Ask-Anything/video_chat2/")
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from conversation import Chat
from tqdm import tqdm
# videochat
from utils.config import Config
from utils.easydict import EasyDict
from models.videochat2_it import VideoChat2_it
from peft import get_peft_model, LoraConfig, TaskType
from configs.configure import uag_oops_dataset_path,results_dir

# ========================================
#             Model Initialization
# ========================================
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


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return None, gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), gr.update(placeholder='Please upload your video first', interactive=False),gr.update(value="Upload & Start Chat", interactive=True), chat_state, img_list


def upload_img(gr_img, gr_video, chat_state, num_segments):
    print(gr_img, gr_video)
    chat_state = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    img_list = []
    if gr_img is None and gr_video is None:
        return None, None, gr.update(interactive=True),gr.update(interactive=True, placeholder='Please upload video/image first!'), chat_state, None
    if gr_video: 
        llm_message, img_list, chat_state = chat.upload_video(gr_video, chat_state, img_list, num_segments)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list
    if gr_img:
        llm_message, img_list,chat_state = chat.upload_img(gr_img, chat_state, img_list)
        return gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True, placeholder='Type and press Enter'), gr.update(value="Start Chatting", interactive=False), chat_state, img_list


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return gr.update(interactive=True, placeholder='Input should not be empty!'), chatbot, chat_state
    chat_state =  chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message,llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
    llm_message = llm_message.replace("<s>", "") # handle <s>
    chatbot[-1][1] = llm_message
    print(chat_state)
    print(f"Answer: {llm_message}")
    return chatbot, chat_state, img_list


# class OpenGVLab(gr.themes.base.Base):
    def __init__(
        self,
        *,
        primary_hue=colors.blue,
        secondary_hue=colors.sky,
        neutral_hue=colors.gray,
        spacing_size=sizes.spacing_md,
        radius_size=sizes.radius_sm,
        text_size=sizes.text_md,
        font=(
            fonts.GoogleFont("Noto Sans"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono=(
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            body_background_fill="*neutral_50",
        )

def main(video_path,user_message,chat ):
        
    # gvlabtheme = OpenGVLab(primary_hue=colors.blue,
    #         secondary_hue=colors.sky,
    #         neutral_hue=colors.gray,
    #         spacing_size=sizes.spacing_md,
    #         radius_size=sizes.radius_sm,
    #         text_size=sizes.text_md,
    #         )

    # video_path = "/scratch/user/hasnat.md.abdullah/uag/data/ssbd/videos/v_ArmFlapping_01.mp4"
    # video_path = "/scratch/user/hasnat.md.abdullah/uag/data/ssbd/videos/v_ArmFlapping_08.mp4"
    num_beams = 1
    temperature = 0.1
    num_segments = 4 # pretrain ckpt num fram = 4

    # chat = init_model()
    ## upload ##
    chat_state = EasyDict({
        "system": "",
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })
    img_list = []
    llm_message, img_list, chat_state = chat.upload_video(video_path, chat_state, img_list, num_segments)
    # print(f"llm_message: {llm_message}")
    # ask question 
    # user_message = """Find the start time and end time of the query below from the video.     Query: A person is flapping his arms with high intensity."""
    # user_message = """"Given a video description, predict the precise start and end timestamps of the described activity within the video. Description: A person is flapping his arms with high."""
    chat_state =  chat.ask(user_message, chat_state)
    llm_message,llm_message_token, chat_state = chat.answer(conv=chat_state, img_list=img_list, max_new_tokens=1000, num_beams=num_beams, temperature=temperature)
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

def get_uag_oops_dataset(uag_oops_dataset_path):
    with open(uag_oops_dataset_path, "r") as f:
        uag_oops_dataset = json.load(f)
    return uag_oops_dataset
def save_data(data, file_name):
    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)
        print(f"{file_name} saved successfully")
if __name__ == "__main__":
    uag_oops_dataset = get_uag_oops_dataset(uag_oops_dataset_path)
    print(f"size of uag_oops_dataset: {len(uag_oops_dataset)}") #1589
    chat = init_model()
    results = {}
    # if result file exists
    if os.path.exists(f"{results_dir}videochat2_pred_uag_oops_dataset.json"):
        with open(f"{results_dir}videochat2_pred_uag_oops_dataset.json", "r") as f:
            results = json.load(f)
        print(f"size of results: {len(results)}")
        
    for video_id,video_info in tqdm(uag_oops_dataset.items()):
        # if video id not in results
        if video_id not in results:
            video_path = video_info["video_path"]
            description = video_info["description"]
            query = f"""Find the start time and end time of the query below from the given video.
            Query: {description} 
            Provide your start and end timestamps in json format.        
            """
            answer = main(video_path, query,chat)
            # answer = "demo"
            video_info["videochat2_pred"] =answer
            video_info['pred_start']=None
            video_info['pred_end']=None
            pattern_1 = r'(\d{1,2}:\d{2}(:\d{2})?)'
            times = re.findall(pattern_1, answer)
            if len(times) == 2:
                video_info['pred_start'] = times[0][0]
                video_info['pred_end'] = times[1][0]


            results[video_id] = video_info
            save_data(results, f"{results_dir}videochat2_pred_uag_oops_dataset.json")
    
    # check the results
    print("verifying results")
    with open(f"{results_dir}videochat2_pred_uag_oops_dataset.json", "r") as f:
        results = json.load(f)
    print(f"size of results: {len(results)}") #1589
    # check if there is any none value for start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            # print("video_id: ", video_id)
            none_count +=1
    print(f"none_count: {none_count}") #322

    