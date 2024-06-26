import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
torch.random.manual_seed(0)
model_id = "microsoft/Phi-3-small-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
assert torch.cuda.is_available(), "This model needs a GPU to run ..."
device = torch.cuda.current_device()
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
messages = [
    {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
    "pad_token_id": tokenizer.eos_token_id
}

def promptModel(prompt, max_tokens=500, temperature=0.0, do_sample=False):
    messages = [
        {"role": "user", "content": prompt},
    ]
    generation_args = {
        "max_new_tokens": max_tokens,
        "return_full_text": False,
        "temperature": temperature,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id
    }
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

import os
from os.path import exists
import pandas as pd
import numpy as np
import time

_lu_       = {'file':[], 'article':[], 'phi_response':[], 'phi_time':[]}

done = set()
if exists('phi_2014_vast.csv.partial'):
    _df_ = pd.read_csv('phi_2014_vast.csv.partial')
    _lu_['file'].extend(_df_['file'])
    _lu_['article'].extend(_df_['article'])
    _lu_['phi_response'].extend(_df_['phi_response'])
    _lu_['phi_time'].extend(_df_['phi_time'])
    done = set(_df_['article'])

_base_dir_ = '../../../data/2014_vast/MC1/News Articles'

ts0 = time.time()
files_processed = 0

_prompt_   = 'Parse the following text into an ontology of people, places, things, and events.  Extract the relationships betwen entities.  Return just the JSON structure.'
for _dir_ in os.listdir(_base_dir_):
    for _file_ in os.listdir(os.path.join(_base_dir_, _dir_)):
        _article_raw_ = open(os.path.join(_base_dir_, _dir_, _file_), 'rb').read()
        _article_     = str(_article_raw_) #.replace('\\r', '').split('\\n')
        if _article_ in done: continue
        ts0_model = time.time()
        _response_    = promptModel(_prompt_ + '\n\n' + _article_, max_tokens=4096)
        ts1_model = time.time()

        _lu_['file'].append(_file_)
        _lu_['article'].append(_article_)
        _lu_['phi_response'].append(_response_)
        _lu_['phi_time'].append(ts1_model-ts0_model)

        pd.DataFrame(_lu_).to_csv('phi_2014_vast.csv', index=False)
        files_processed += 1
        #if files_processed > 0: break
    #if files_processed > 0: break

ts1 = time.time()
open('phi_time.txt', 'wt').write(str(ts1-ts0))
