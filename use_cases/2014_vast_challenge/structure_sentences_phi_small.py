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

def breakIntoSentences(_str_):
    _sentences_ = []
    for _ in _str_.split('.'):
        if len(_) > 0:
            _sentences_.append(_.strip() + '.')
    return _sentences_

def separateArticle(_str_):
    _source_, _title_, _published_, _sentences_ = '', '', '', []
    for _part_ in _str_.split('\n'):
        if len(_part_) > 0 and _part_.startswith('<<') == False:
            if   _part_.startswith("b'SOURCE:"):   _source_    = _part_.replace("b'SOURCE: ", '')  .strip()
            elif _part_.startswith("TITLE:"):      _title_     = _part_.replace('TITLE: ', '')     .strip()
            elif _part_.startswith("PUBLISHED:"):  _published_ = _part_.replace('PUBLISHED: ', '') .strip()
            else:                                  _sentences_.extend(breakIntoSentences(_part_))
    return _source_, _title_, _published_, _sentences_

#_src_, _title_, _published_, _sentences_ = separateArticle(_lu_['article'][1].replace('\\r', '').replace('\\n', '\n'))

import pandas as pd
import numpy as np
import time
import os
from os.path import exists

_lu_       = {'file':[], 'source':[], 'title':[], 'published':[], 'sentence':[], 'sentence_no':[], 'phi_small_response':[], 'phi_small_time':[]}
done       = set()

if exists('phi_small_2014_vast_sbs.csv.partial'):
    _df_ = pd.read_csv('phi_small_2014_vast_sbs.csv.partial')
    _lu_['file'].extend(_df_['file'])
    _lu_['source'].extend(_df_['source'])
    _lu_['title'].extend(_df_['title'])
    _lu_['published'].extend(_df_['published'])
    _lu_['sentence'].extend(_df_['sentence'])
    _lu_['sentence_no'].extend(_df_['sentence_no'])
    _lu_['phi_small_response'].extend(_df_['phi_small_response'])
    _lu_['phi_small_time'].extend(_df_['phi_small_time'])
    done = set(_df_['file'])

_base_dir_ = '../../../data/2014_vast/MC1/News Articles'

_prompt_   = 'Translate the following text into an ontology represented as JSON.  Do not include the schema context.  Do not explain your response.'
for _dir_ in os.listdir(_base_dir_):
    for _file_ in os.listdir(os.path.join(_base_dir_, _dir_)):
        if _file_ in done: continue
        _article_raw_ = open(os.path.join(_base_dir_, _dir_, _file_), 'rb').read()
        _src_, _title_, _published_, _sentences_ = separateArticle(str(_article_raw_).replace('\\r', '').replace('\\n', '\n'))
        _sentence_no_ = 0
        for _ in _sentences_:
            if len(_) < 6: continue
            ts0_model   = time.time()
            _response_  = promptModel(_prompt_ + '\n\n' + _, max_tokens=4096)
            ts1_model   = time.time()

            _lu_['file'].append(_file_)
            _lu_['source'].append(_src_)
            _lu_['title'].append(_title_)
            _lu_['published'].append(_published_)
            _lu_['sentence'].append(_)
            _lu_['sentence_no'].append(_sentence_no_)
            _lu_['phi_small_response'].append(_response_)
            _lu_['phi_small_time'].append(ts1_model-ts0_model)
            _sentence_no_ += 1

        pd.DataFrame(_lu_).to_csv('phi_small_2014_vast_sbs.csv', index=False)

