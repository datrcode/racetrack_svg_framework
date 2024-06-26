import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def promptModel(_user_, _system_='You are a helpful digital assistant.', max_tokens=256):
    messages = [
        {"role": "system", "content": _system_},
        {"role": "user",   "content": _user_},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = pipeline(
        prompt,
        max_new_tokens=max_tokens,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    return outputs[0]["generated_text"][len(prompt):]

import os
_base_dir_ = '../../../data/2014_vast/MC1/News Articles'

import pandas as pd
import numpy as np
import time

ts0 = time.time()
files_processed = 0
_lu_       = {'file':[], 'article':[], 'llama3_8b_response':[], 'llama3_8b_time':[]}
_prompt_   = 'Parse the following text into an ontology of people, places, things, and events.  Extract the relationships betwen entities.  Return just the JSON structure.'
for _dir_ in os.listdir(_base_dir_):
    for _file_ in os.listdir(os.path.join(_base_dir_, _dir_)):
        _article_raw_ = open(os.path.join(_base_dir_, _dir_, _file_), 'rb').read()
        _article_     = str(_article_raw_) #.replace('\\r', '').split('\\n')
        ts0_model = time.time()
        _response_    = promptModel(_article_, _prompt_, max_tokens=4096)
        ts1_model = time.time()

        _lu_['file'].append(_file_)
        _lu_['article'].append(_article_)
        _lu_['llama3_8b_response'].append(_response_)
        _lu_['llama3_8b_time'].append(ts1_model-ts0_model)

        pd.DataFrame(_lu_).to_csv('llama3_8b_2014_vast.csv', index=False)
        files_processed += 1
        #if files_processed > 0: break
    #if files_processed > 0: break


ts1 = time.time()
open('llama3_8b_time.txt', 'wt').write(str(ts1-ts0))
