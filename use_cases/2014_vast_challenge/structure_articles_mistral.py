from huggingface_hub import snapshot_download
from pathlib import Path
mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
mistral_models_path.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)
from mistral_inference.model import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)
def promptModel(prompt, max_tokens=64, temperature=0.0):
    completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    tokens = tokenizer.encode_chat_completion(completion_request).tokens
    out_tokens, _ = generate([tokens], model, max_tokens=max_tokens, temperature=temperature, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
    result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
    return result

import os
from os.path import exists

_base_dir_ = '../../../data/2014_vast/MC1/News Articles'

import pandas as pd
import numpy as np
import time

_lu_       = {'file':[], 'article':[], 'mistral_7b_response':[], 'mistral_7b_time':[]}

done = set()
if exists('mistral_7b_2014_vast.csv.partial'):
    _df_ = pd.read_csv('mistral_7b_2014_vast.csv.partial')
    _lu_['file'].extend(_df_['file'])
    _lu_['article'].extend(_df_['article'])
    _lu_['mistral_7b_response'].extend(_df_['mistral_7b_response'])
    _lu_['mistral_7b_time'].extend(_df_['mistral_7b_time'])
    done = set(_df_['article'])

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
        _lu_['mistral_7b_response'].append(_response_)
        _lu_['mistral_7b_time'].append(ts1_model-ts0_model)

        pd.DataFrame(_lu_).to_csv('mistral_7b_2014_vast.csv', index=False)
        files_processed += 1
        #if files_processed > 0: break
    #if files_processed > 0: break

ts1 = time.time()
open('mistral_7b_time.txt', 'wt').write(str(ts1-ts0))
