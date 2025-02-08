#
# Streamlit App
#
import streamlit as st
import pandas as pd
import numpy as np
import rtsvg
rt = rtsvg.RACETrack()

@st.cache_data
def loadData():
    return pd.read_parquet('../../data/moon_example.parquet')

df = loadData()

st.title('SCU Editor')
questions = df['question'].unique()
question  = st.selectbox('Question', questions)
models    = df.query(f'question == @question')['model'].unique()
model     = st.selectbox('Model', models)

st.subheader('Summary')
_df_ = df.query('question == @question and model == @model')
summary = _df_.iloc[0]['summary']
st.write(summary)

def colorizer(summary, excerpt):
    excerpt = excerpt.lower().strip()
    if len(excerpt) == 0: return 'orange'
    _parts_ = excerpt.split('...')
    for _part_ in _parts_:
        _part_ = _part_.strip() 
        if _part_ not in summary: return 'red'
    return 'blue'

def makeSafe(scu):
    _safe_ = []
    for c in scu:
        if   c >= 'a' and c <= 'z': _safe_.append(c)
        elif c >= 'A' and c <= 'Z': _safe_.append(c)
        elif c >= '0' and c <= '9': _safe_.append(c)
        else: _safe_.append('_')
    return ''.join(_safe_)

def textAreaChanged(scusafe=None):
    _exec_ = f'st.session_state.{scusafe}.value = "test"'
    eval(_exec_)

st.subheader('Summary Content Units')
summary_content_units = sorted(list(set(_df_['summary_content_unit'])))
row_i, svg_id_num = 4, 0
while svg_id_num < len(summary_content_units):
    if row_i >= 4:
        cols  = st.columns(4)
        row_i = 0
    summary_content_unit = summary_content_units[svg_id_num]
    _df_scu_   = _df_.query('summary_content_unit == @summary_content_unit').reset_index()
    _excerpt_  = _df_scu_.iloc[0]['excerpt']
    _color_    = colorizer(summary, _excerpt_)
    scu_safe   = makeSafe(summary_content_unit)
    cols[row_i].text_area(key       = scu_safe,
                          label     = f':{_color_}[{summary_content_unit}]', 
                          on_change = textAreaChanged,
                          kwargs    = {'scusafe':scu_safe},
                          value     = _excerpt_) 
    svg_id_num, row_i = svg_id_num + 1, row_i + 1


