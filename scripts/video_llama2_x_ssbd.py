import json
import torch
import os
import re
from joblib import Memory
from tqdm import tqdm
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/foundation_models/Video-LLaMA")
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import ssbd_data_path, video_llama2_pred_x_ssbd_result_path


import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='/scratch/user/hasnat.md.abdullah/uag/foundation_models/Video-LLaMA/eval_configs/video_llama_eval_withaudio.yaml', help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='llama2', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args
def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
# ========================================
#             Model Initialization
# ========================================
@memory.cache
def initialize_model(args):
    print('Initializing Chat')
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')
    return chat


# ========================================
#             Function Definitions
# ========================================

def upload_imgorvideo(gr_video, gr_img, text_input, chat_state, chatbot, audio_flag):
    if args.model_type == 'vicuna':
        chat_state = default_conversation.copy()
    else:
        chat_state = conv_llava_llama_2.copy()
    if gr_img is None and gr_video is None:
        return None, None, None, chat_state, None
    elif gr_img is not None and gr_video is None:
        print(gr_img)
        chatbot = chatbot + [((gr_img,), None)]
        chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        img_list = []
        llm_message = chat.upload_img(gr_img, chat_state, img_list)
        return chat_state, img_list, chatbot
    elif gr_video is not None and gr_img is None:
        # print(gr_video)
        chatbot = chatbot + [((gr_video,), None)]
        chat_state.system =  ""
        img_list = []
        if audio_flag:
            llm_message = chat.upload_video(gr_video, chat_state, img_list)
        else:
            llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list)
        return chat_state, img_list, chatbot
    else:
        return chat_state, None, chatbot

def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return chat_state, chatbot, chat_state
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return chat_state, chatbot, chat_state

def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    chatbot[-1][1] = llm_message
    # print(chat_state.get_prompt())
    # print(chat_state)
    return chatbot, chat_state, img_list

# ========================================
#             Main Function
# ========================================

def main(gr_video, gr_img, text_input, audio_flag, num_beams, temperature):
    chat_state = None
    img_list = None
    
    chatbot = []
    
    chat_state, img_list, chatbot = upload_imgorvideo(gr_video, gr_img, text_input, chat_state, chatbot, audio_flag)
    chat_state, chatbot, chat_state = gradio_ask(text_input, chatbot, chat_state)
    chatbot, chat_state, img_list = gradio_answer(chatbot, chat_state, img_list, num_beams, temperature)
    
    return chatbot

# ========================================
#             Run the Main Function
# ========================================
def load_file (file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data
def extract_time_from_answer(answer):
    pattern_1 = r'(\d{1,2}:\d{2}(:\d{2})?)'
    times = re.findall(pattern_1, answer)
    if len(times) == 2:
        pred_start = times[0][0]
        pred_end = times[1][0]
    else: 
        pred_start = None
        pred_end = None
    return pred_start, pred_end
if __name__ == "__main__":
    # Load the UAG OOPS dataset
    ssbd_data = load_file(ssbd_data_path)
    
    print(f"SSBD data loaded with {len(ssbd_data)} videos")
    args = parse_args()
    chat = initialize_model(args)

    results = {}
    if os.path.exists(video_llama2_pred_x_ssbd_result_path):
        with open(video_llama2_pred_x_ssbd_result_path, "r") as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results")
    for sample in tqdm(ssbd_data):
        print(f"sample: {sample}")
        video_tag = sample[0]
        behaviour_id = sample[2]['id']
        video_id = video_tag+'_'+str(behaviour_id)
        video_info = {}
        video_info = {'video_path': sample[1], 'behaviour': sample[2]}
        if video_id not in results:
            video_path = sample[1]
            category = sample[2]['category']
            intensity = sample[2]['intensity']
            query = f"""Find the start time and end time of the query below from the given video.
            Query: A person is {category} with {intensity} intensity.
            """
            video_info["query"] = query
            audio_flag = True
            num_beams = 1  # Replace with the number of beams for decoding
            temperature = 0.5  # Replace with the temperature for decoding
            gr_video = video_path
            text_input = query
            gr_img = None
            try:
                chatbot = main(gr_video, gr_img, text_input, audio_flag, num_beams, temperature)
                print(chatbot[-1][1])
                video_info["video_llama2_pred"] = chatbot[-1][1]
                pred_start, pred_end = extract_time_from_answer(chatbot[-1][1])
                print(f"Predicted start: {pred_start}, Predicted end: {pred_end}")
                video_info['pred_start'] = pred_start
                video_info['pred_end'] = pred_end
                results[video_id] = video_info
            except Exception as e:
                print(e)
            # save results
            with open(video_llama2_pred_x_ssbd_result_path, "w") as f:
                json.dump(results, f, indent=4)
                print(f"Results saved successfully in {video_llama2_pred_x_ssbd_result_path}")
        
    # check the results
    print("verifying results")
    with open(video_llama2_pred_x_ssbd_result_path, "r") as f:
        results = json.load(f)
        print(f"size of results: {len(results)}") #1589
    # check if there is any none value for the start or end preds
    none_count = 0
    for video_id,video_info in results.items():
        if video_info['pred_start'] is None or video_info['pred_end'] is None:
            # print("video_id: ", video_id)
            none_count +=1
    print(f"none_count: {none_count}")#70








    # # try for a single video 
    # gr_video ="/scratch/user/hasnat.md.abdullah/uag/data/oops_dataset/oops_video/val/34 Funny Kid Nominees - FailArmy Hall Of Fame (May 2017)0.mp4"  # Replace with your video input
    # gr_img = None  # Replace with your image input
    # description ="A guy jumps onto a bed where his son is. When the guy jumps, the son flies up and hits the wall."
    # text_input =f"""Find the start time and end time of the query below from the given video.
    #         Query: {description}     
    #         """  # Replace with your text input
    # audio_flag = True  # Replace with True if your video has audio, False otherwise
    # num_beams = 1  # Replace with the number of beams for decoding
    # temperature = 0.5  # Replace with the temperature for decoding
    
    # chatbot = main(gr_video, gr_img, text_input, audio_flag, num_beams, temperature)
    
    # for message, response in chatbot:
    #     print("User: ", message)
    #     print("Chatbot: ", response)
    #     print()
