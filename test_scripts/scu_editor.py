#
# Streamlit App
#

import streamlit as st
import pandas as pd
import numpy as np
import rtsvg
rt = rtsvg.RACETrack()

st.markdown("""
    <style>
    .stTextArea [data-baseweb=base-input] {
        background-image: linear-gradient(140deg, rgb(54, 36, 31) 0%, rgb(121, 56, 100) 50%, rgb(106, 117, 25) 75%);
        -webkit-text-fill-color: white;
    }

    .stTextArea [data-baseweb=base-input] [disabled=""]{
        background-image: linear-gradient(45deg, red, purple, red);
        -webkit-text-fill-color: gray;
    }
    </style>
    """,unsafe_allow_html=True)

st.markdown("""
<style>
    .stTextInput input[aria-label="test color"] {
        background-color: #0066cc;
        color: #33ff33;
    }
    .stTextInput input[aria-label="test color2"] {
        background-color: #cc0066;
        color: #ffff33;
    }
</style>
""", unsafe_allow_html=True)

st.text_input("test color")
st.text_input("test color2")

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


def svgStatus(status, svg_id):
    _w_, _h_               = 32, 32
    _x_ins_, _y_ins_, _rx_ =  4,  4, 4
    _svg_base_    = '<svg id="{svg_id}" x="0" y="0" width="{_w_}" height="{_h_}">'
    _svg_end_     = '</svg>'
    _svg_caution_ = _svg_base_ + \
                    f'<rect id="{svg_id}_rect" x="{_x_ins_}" y="{_y_ins_}" width="{_w_ - 2*_x_ins_}" height="{_h_ - 2*_y_ins_}" rx="{_rx_}" fill="yellow" />' + \
                    _svg_end_
    _svg_error_   = _svg_base_ + \
                    f'<rect id="{svg_id}_rect" x="{_x_ins_}" y="{_y_ins_}" width="{_w_ - 2*_x_ins_}" height="{_h_ - 2*_y_ins_}" rx="{_rx_}" fill="red" />' + \
                    _svg_end_
    _svg_okay_    = _svg_base_ + \
                    f'<rect id="{svg_id}_rect" x="{_x_ins_}" y="{_y_ins_}" width="{_w_ - 2*_x_ins_}" height="{_h_ - 2*_y_ins_}" rx="{_rx_}" fill="blue" />' + \
                    _svg_end_
    if status == 'caution': return _svg_caution_
    if status == 'error':   return _svg_error_
    if status == 'okay':    return _svg_okay_


def validateExcerpt(excerpt, svg_id):
    if len(excerpt.strip()) == 0:
        return svgStatus('caution', svg_id)
    for _part_ in excerpt.split('...'):
        _part_ = _part_.strip().lower()
        if _part_ not in summary.lower(): 
            return svgStatus('error',svg_id)
    return svgStatus('okay', svg_id)

st.write(summary)
st.subheader('Summary Content Units')
summary_content_units = sorted(list(set(_df_['summary_content_unit'])))
svg_id_num = 0
for i in range(len(summary_content_units)):
    scu_col, cu_col, valid_col = st.columns(3)
    summary_content_unit = summary_content_units[i]
    scu_col.write(summary_content_unit)
    _df_scu_   = _df_.query('summary_content_unit == @summary_content_unit').reset_index()
    _excerpt_  = _df_scu_.iloc[0]['excerpt']
    cu_col.text_area(label=summary_content_unit, 
                     value=_excerpt_,
                     label_visibility='collapsed')
    valid_col.write(validateExcerpt(_excerpt_, 'status_' + str(svg_id_num)), unsafe_allow_html=True)
    svg_id_num += 1

