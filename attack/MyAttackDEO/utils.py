import random

import numpy as np
import pandas as pd
from PIL import Image
import torch
import os
from datasets import Dataset, load_dataset
from transformers import DataCollatorForLanguageModeling


import argparse
import logging
import random
import time
import numpy as np
from accelerate import Accelerator
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoProcessor,LlavaForConditionalGeneration, get_scheduler, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration # for llava/mistral
from PIL import Image
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import pandas as pd


"""
prompts: a list of textual prompts (batch)
images: a list of image names for batch inference e.g., ["objs.png", "ball.png", "hello.png"]

images should be the same length as the prompts. Think of it as the first prompt goes with the first image,
the second prompt goes with the second image, ...

returns: Responses in a list
"""
def batch_generate_response(prompts, model, processor, device, new_tokens=100, images=None):
    if images == None:
        batch = processor.tokenizer(prompts, return_tensors='pt', padding = True).to(device)
        len_prompt = batch['input_ids'].shape[1]

        # maybe it's also better to put: with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output_tokens = model.generate(**batch, max_new_tokens=new_tokens, do_sample = True, temperature = 0.6, top_p = 0.9)
            response = processor.tokenizer.batch_decode(output_tokens[:,len_prompt:], skip_special_tokens=True) 
            return response 
    
    else:
        raw_images = [Image.open(img).resize((3100,1438)) for img in images]
        batch = processor(prompts, raw_images, return_tensors='pt', padding = True).to(device, torch.float16)
        len_prompt = batch['input_ids'].shape[1]
        # with torch.cuda.amp.autocast(): # interesting! for batch generation, this led to errors! so I commented it out!
        with torch.no_grad():
            output_tokens = model.generate(**batch, max_new_tokens=new_tokens, do_sample = True, temperature = 0.6, top_p = 0.9)
        response = processor.tokenizer.batch_decode(output_tokens[:,len_prompt:], skip_special_tokens=True) 
        return response 