import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from joblib import Memory
cache_dir = './cachedir'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
memory = Memory(cache_dir)
import sys
sys.path.append("/scratch/user/hasnat.md.abdullah/uag/")
from configs.configure import llama3_model_id


class Llama3Loader():
    def __init__(self) -> None:
        
        self.tokenizer, self.model = self.setup_llama_pipeline()

    @memory.cache
    def setup_llama_pipeline():
      '''
      setup the llama pipeline
      '''
      # model_id = "./meta-llama/Meta-Llama-3-8B-Instruct"
      model_id = llama3_model_id

      tokenizer = AutoTokenizer.from_pretrained(model_id)
      model = AutoModelForCausalLM.from_pretrained(model_id,
          torch_dtype=torch.bfloat16,
          device_map="auto",
      )
      return tokenizer, model
    
    def infer(self,content):
      messages = [
    {"role": "system", "content": "You are a video analyst. You can read the the video text representation and predict the start and end time of any activity in the video."},
    {"role": "user", "content": f"{content}"},
]
      input_ids = self.tokenizer.apply_chat_template(
      messages,
      add_generation_prompt=True,
      return_tensors="pt"
  ).to(self.model.device)
      terminators = [
      self.tokenizer.eos_token_id,
      self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]
      outputs = self.model.generate(
      input_ids,
      max_new_tokens=256,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )
      response = outputs[0][input_ids.shape[-1]:]
      print(self.tokenizer.decode(response, skip_special_tokens=True))
      generated_text = self.tokenizer.decode(response, skip_special_tokens=True)

      return generated_text
    

if __name__ == "__main__":
  llama3 = Llama3Loader()
  content = "Find the start time and end time of the query below from the video. Query: A guy jumps onto a bed where his son is. When the guy jumps, the son flies up and hits the wall. 0.0s: a man is helping a little boy put a quilt on a bed in a room with a man standing next to him.\n1.0s: a person laying on a bed in front of a window in a room with a quilt on top of the bed.\n2.0s: a man playing with a child on a bed in front of a window in a room with a bed and a window.\n3.0s: a man doing a handstand on a bed in front of a window in a room with a window in the background.\n"
  answer = llama3.infer(content)
  print(answer)
