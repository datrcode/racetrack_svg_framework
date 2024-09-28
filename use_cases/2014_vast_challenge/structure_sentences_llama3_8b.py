import transformers
import torch
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
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

_lu_       = {'file':[], 'source':[], 'title':[], 'published':[], 'sentence':[], 'sentence_no':[], 'llama3_8b_response':[], 'llama3_8b_time':[]}
done       = set()

if exists('llama3_8b_2014_vast_sbs.csv.partial'):
    _df_ = pd.read_csv('llama3_8b_2014_vast_sbs.csv.partial')
    _lu_['file'].extend(_df_['file'])
    _lu_['source'].extend(_df_['source'])
    _lu_['title'].extend(_df_['title'])
    _lu_['published'].extend(_df_['published'])
    _lu_['sentence'].extend(_df_['sentence'])
    _lu_['sentence_no'].extend(_df_['sentence_no'])
    _lu_['llama3_8b_response'].extend(_df_['llama3_8b_response'])
    _lu_['llama3_8b_time'].extend(_df_['llama3_8b_time'])
    done = set(_df_['file'])

_base_dir_ = '../../../data/2014_vast/MC1/News Articles'

_prompt_   = 'Translate the following text into an CCO ontology represented as JSON.  Only include the JSON structure.'
for _dir_ in os.listdir(_base_dir_):
    for _file_ in os.listdir(os.path.join(_base_dir_, _dir_)):
        if _file_ in done: continue
        _article_raw_ = open(os.path.join(_base_dir_, _dir_, _file_), 'rb').read()
        _src_, _title_, _published_, _sentences_ = separateArticle(str(_article_raw_).replace('\\r', '').replace('\\n', '\n'))
        _sentence_no_ = 0
        for _ in _sentences_:
            if len(_) < 6: continue
            ts0_model   = time.time()
            _response_  = promptModel(_, _prompt_, max_tokens=4096)
            ts1_model   = time.time()

            _lu_['file'].append(_file_)
            _lu_['source'].append(_src_)
            _lu_['title'].append(_title_)
            _lu_['published'].append(_published_)
            _lu_['sentence'].append(_)
            _lu_['sentence_no'].append(_sentence_no_)
            _lu_['llama3_8b_response'].append(_response_)
            _lu_['llama3_8b_time'].append(ts1_model-ts0_model)
            _sentence_no_ += 1

        pd.DataFrame(_lu_).to_csv('llama3_8b_2014_vast_sbs.csv', index=False)

