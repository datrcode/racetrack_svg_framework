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

def validateExcerpt(excerpt):
    for _part_ in excerpt.split('...'):
        _part_ = _part_.strip().lower()
        if _part_ not in summary.lower(): return False
    return True

st.write(summary)
st.subheader('Summary Content Units')
summary_content_units = sorted(list(set(_df_['summary_content_unit'])))
for i in range(len(summary_content_units)):
    scu_col, cu_col, valid_col = st.columns(3)
    summary_content_unit = summary_content_units[i]
    scu_col.write(summary_content_unit)
    _df_scu_   = _df_.query('summary_content_unit == @summary_content_unit').reset_index()
    _excerpt_  = _df_scu_.iloc[0]['excerpt']
    cu_col.text_input(label=summary_content_unit, value=_excerpt_, label_visibility='collapsed')
    valid_col.checkbox(label=summary_content_unit+'_valid',
                       value=validateExcerpt(_excerpt_),
                       disabled=True,
                       key=summary_content_unit+'_valid',
                       label_visibility='collapsed')
