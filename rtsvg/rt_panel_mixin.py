# Copyright 2025 David Trimm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pandas as pd
import polars as pl

import threading

import panel as pn
import param

from panel.reactive import ReactiveHTML

from math import pi, sqrt, sin, cos
import copy

from shapely import Polygon

from .rt_stackable                import RTStackable
from .rt_graph_interactive_panel  import RTGraphInteractivePanel
from .rt_coordinated_views        import RTCoordinatedViews

__name__ = 'rt_panel_mixin'

#
# Panel Mixin
#
class RTPanelMixin(object):
    #
    # Constructor
    # - may need to modify inline=True...
    #
    def __panel_mixin_init__(self):
        pn.extension(inline=True)

    #
    # layoutPanel() - helps with the constructions of the layout
    #
    def layoutPanel(self):
        return LayoutPanel()

    #
    # interactiveGraphPanel()
    #
    def interactiveGraphPanel(self, df, ln_params, w=600, h=400, use_linknode=False, **kwargs):
        ''' Interactive Graph Layout using Panel Architecture

        Parameters
        ----------
        df : DataFrame
            The dataframe to be rendered
        ln_params : dict
            Should include relationships and pos (both same as linkNode)
            Will be passed without modification to link()
        
        w, h : int
            Width and height of the layout
        
        use_linknode : bool
            Use LinkNode() (which implements more rendering features) instead of link()

        Use print(_instance_) to show key shortcut commands

        Use saveLayout() and loadLayout() to store and retrieve layouts in parquet format
        '''
        return RTGraphInteractivePanel(self, df, ln_params, w, h, use_linknode, **kwargs)

    #
    # interactivePanel() - coordinated views with configurable components.
    #
    def interactivePanel(self,
                         df,
                         spec,                  # Layout specification
                         w,                     # Width of the panel
                         h,                     # Heght of the panel
                         rt_params      = {},   # Racetrack params -- dictionary of param:value
                         # -------------------- #
                         h_gap          = 0,    # Horizontal left/right gap
                         v_gap          = 0,    # Verticate top/bottom gap
                         widget_h_gap   = 1,    # Horizontal gap between widgets
                         widget_v_gap   = 1,    # Vertical gap between widgets
                         **kwargs):             # Other arguments to pass to the layout instance

        ''' Interactive Panel Layout using Panel Architecture

        Parameters
        ----------
        df : DataFrame
            The dataframe to be rendered

        spec : dict
            Layout specification

        w, h : int
            Width and height of the layout

        rt_params : dict
            Params passed to all of the widgets within the layout

        h_gap, v_gap, widget_h_gap, widget_v_gap : int
            Horizontal, vertical, and widget gap between views
        
        To debug, use the ".show()" version of the instance.  Then, errors (and prints) will be sent back to the notebook.
        '''
        return RTCoordinatedViews(df, self, spec, w, h, rt_params, h_gap, v_gap, widget_h_gap, widget_v_gap, **kwargs)

    #
    # RTFontMetricsPanel - determine the font metrics for a specific
    # browser / jupyter configuration
    #
    class RTFontMetricsPanel(ReactiveHTML):
        txt12_w      = param.Number(default=7)
        txt12short_w = param.Number(default=7)
        txt14_w      = param.Number(default=7)
        txt16_w      = param.Number(default=7)
        txt24_w      = param.Number(default=7)
        txt36_w      = param.Number(default=7)
        txt36short_w = param.Number(default=7)
        txt48_w      = param.Number(default=7)
     
        _template = """
            <svg width="1024" height="256">
                <text id="click" x="5" y="32"  font-family="Times"     font-size="28px" fill="#ff0000">Click Me</text>
                <text id="txt12" x="5" y="62"  font-family="Monospace" font-size="12px">abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ</text>
                <text id="txt12short" x="5" y="238"  font-family="Monospace" font-size="12px">abcdefghijklmnopqrstuvwxyz</text>

                <text id="txt14" x="5" y="76"  font-family="Monospace" font-size="14px">abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ</text>
                <text id="txt16" x="5" y="92"  font-family="Monospace" font-size="16px">abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ</text>
                <text id="txt24" x="5" y="120" font-family="Monospace" font-size="24px">abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ</text>
                <text id="txt36" x="5" y="148" font-family="Monospace" font-size="36px">abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ</text>
                <text id="txt36short" x="5" y="226"  font-family="Monospace" font-size="36px">abcdefghijklmnopqrstuvwxyz</text>
                <text id="txt48" x="5" y="186" font-family="monospace" font-size="48px">abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ</text>
                <rect id="screen" x="0" y="0" width="1024" height="256" fill-opacity="0.1"
                  onmousedown="${script('myonmousedown')}"
                />
            </svg>
        """

        _scripts = {
                'myonmousedown':"""
                    click.setAttribute("fill","#0000ff");
                    let my_num_chars       = 26*4 + 3;
                    let my_num_chars_short = 26
                    data.txt12_w      = txt12.getBoundingClientRect().width/my_num_chars;
                    data.txt12short_w = txt12short.getBoundingClientRect().width/my_num_chars_short;
                    data.txt14_w      = txt14.getBoundingClientRect().width/my_num_chars;
                    data.txt16_w      = txt16.getBoundingClientRect().width/my_num_chars;
                    data.txt24_w      = txt24.getBoundingClientRect().width/my_num_chars;
                    data.txt36_w      = txt36.getBoundingClientRect().width/my_num_chars;
                    data.txt36short_w = txt36short.getBoundingClientRect().width/my_num_chars_short;
                    data.txt48_w      = txt48.getBoundingClientRect().width/my_num_chars;
                    click.setAttribute("fill","#000000");
                """
        }

    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------
    

#
# ReactiveHTML Class for Layout Implementation
#
class LayoutPanel(ReactiveHTML):
    #
    # Contains the parameterized string
    #
    export_string = param.String(default='None')

    #
    # Print export layout ... for copying and pasting into next code block
    #
    def layoutSpec(self):
        parts = self.export_string.split('|') # pipe (|) used because line returns fail in javascript
        for part in parts:
            print(part)

    #
    # Template... annoying since iterations don't seem to fit here...  lots of repeated code blocks
    #
    _template = '''
		<svg id="placer" width="800" height="800" xmlns="http://www.w3.org/2000/svg">
          <rect               x="0"   y="0"   width="800"  height="800"  fill="#000000"/>

            <line x1="0"   y1="0" x2="0"   y2="800" stroke="#303030" />            <line x1="25"  y1="0" x2="25"  y2="800" stroke="#303030" />            <line x1="50"  y1="0" x2="50"  y2="800" stroke="#303030" />
            <line x1="75"  y1="0" x2="75"  y2="800" stroke="#303030" />            <line x1="100" y1="0" x2="100" y2="800" stroke="#303030" />            <line x1="125" y1="0" x2="125" y2="800" stroke="#303030" />
            <line x1="150" y1="0" x2="150" y2="800" stroke="#303030" />            <line x1="175" y1="0" x2="175" y2="800" stroke="#303030" />            <line x1="200" y1="0" x2="200" y2="800" stroke="#303030" />
            <line x1="225" y1="0" x2="225" y2="800" stroke="#303030" />            <line x1="250" y1="0" x2="250" y2="800" stroke="#303030" />            <line x1="275" y1="0" x2="275" y2="800" stroke="#303030" />
            <line x1="300" y1="0" x2="300" y2="800" stroke="#303030" />            <line x1="325" y1="0" x2="325" y2="800" stroke="#303030" />            <line x1="350" y1="0" x2="350" y2="800" stroke="#303030" />
            <line x1="375" y1="0" x2="375" y2="800" stroke="#303030" />            <line x1="400" y1="0" x2="400" y2="800" stroke="#303030" />            <line x1="425" y1="0" x2="425" y2="800" stroke="#303030" />
            <line x1="450" y1="0" x2="450" y2="800" stroke="#303030" />            <line x1="475" y1="0" x2="475" y2="800" stroke="#303030" />            <line x1="500" y1="0" x2="500" y2="800" stroke="#303030" />
            <line x1="525" y1="0" x2="525" y2="800" stroke="#303030" />            <line x1="550" y1="0" x2="550" y2="800" stroke="#303030" />            <line x1="575" y1="0" x2="575" y2="800" stroke="#303030" />
            <line x1="600" y1="0" x2="600" y2="800" stroke="#303030" />            <line x1="625" y1="0" x2="625" y2="800" stroke="#303030" />            <line x1="650" y1="0" x2="650" y2="800" stroke="#303030" />
            <line x1="675" y1="0" x2="675" y2="800" stroke="#303030" />            <line x1="700" y1="0" x2="700" y2="800" stroke="#303030" />            <line x1="725" y1="0" x2="725" y2="800" stroke="#303030" />
            <line x1="750" y1="0" x2="750" y2="800" stroke="#303030" />            <line x1="775" y1="0" x2="775" y2="800" stroke="#303030" />

            <line y1="0"   x1="0" y2="0"   x2="800" stroke="#303030" />            <line y1="25"  x1="0" y2="25"  x2="800" stroke="#303030" />            <line y1="50"  x1="0" y2="50"  x2="800" stroke="#303030" />
            <line y1="75"  x1="0" y2="75"  x2="800" stroke="#303030" />            <line y1="100" x1="0" y2="100" x2="800" stroke="#303030" />            <line y1="125" x1="0" y2="125" x2="800" stroke="#303030" />
            <line y1="150" x1="0" y2="150" x2="800" stroke="#303030" />            <line y1="175" x1="0" y2="175" x2="800" stroke="#303030" />            <line y1="200" x1="0" y2="200" x2="800" stroke="#303030" />
            <line y1="225" x1="0" y2="225" x2="800" stroke="#303030" />            <line y1="250" x1="0" y2="250" x2="800" stroke="#303030" />            <line y1="275" x1="0" y2="275" x2="800" stroke="#303030" />
            <line y1="300" x1="0" y2="300" x2="800" stroke="#303030" />            <line y1="325" x1="0" y2="325" x2="800" stroke="#303030" />            <line y1="350" x1="0" y2="350" x2="800" stroke="#303030" />
            <line y1="375" x1="0" y2="375" x2="800" stroke="#303030" />            <line y1="400" x1="0" y2="400" x2="800" stroke="#303030" />            <line y1="425" x1="0" y2="425" x2="800" stroke="#303030" />
            <line y1="450" x1="0" y2="450" x2="800" stroke="#303030" />            <line y1="475" x1="0" y2="475" x2="800" stroke="#303030" />            <line y1="500" x1="0" y2="500" x2="800" stroke="#303030" />
            <line y1="525" x1="0" y2="525" x2="800" stroke="#303030" />            <line y1="550" x1="0" y2="550" x2="800" stroke="#303030" />            <line y1="575" x1="0" y2="575" x2="800" stroke="#303030" />
            <line y1="600" x1="0" y2="600" x2="800" stroke="#303030" />            <line y1="625" x1="0" y2="625" x2="800" stroke="#303030" />            <line y1="650" x1="0" y2="650" x2="800" stroke="#303030" />
            <line y1="675" x1="0" y2="675" x2="800" stroke="#303030" />            <line y1="700" x1="0" y2="700" x2="800" stroke="#303030" />            <line y1="725" x1="0" y2="725" x2="800" stroke="#303030" />
            <line y1="750" x1="0" y2="750" x2="800" stroke="#303030" />            <line y1="775" x1="0" y2="775" x2="800" stroke="#303030" />

            <rect id="r0"  x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r1"  x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r2"  x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r3"  x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r4"  x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r5"  x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r6"  x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r7"  x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r8"  x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r9"  x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r10" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r11" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r12" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r13" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r14" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r15" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r16" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r17" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r18" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r19" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r20" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r21" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r22" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r23" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r24" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r25" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r26" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r27" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>
            <rect id="r28" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>  <rect id="r29" x="-10" y="-10" width="5" height="5" stroke="#0000ff" stroke-width="3" opacity="0.5"/>

            <text id="t0"  x="-10" y="-10" fill="#ffffff">r0</text>   <text id="t1"  x="-10" y="-10" fill="#ffffff">r1</text>   <text id="t2"  x="-10" y="-10" fill="#ffffff">r2</text>
            <text id="t3"  x="-10" y="-10" fill="#ffffff">r3</text>   <text id="t4"  x="-10" y="-10" fill="#ffffff">r4</text>   <text id="t5"  x="-10" y="-10" fill="#ffffff">r5</text>
            <text id="t6"  x="-10" y="-10" fill="#ffffff">r6</text>   <text id="t7"  x="-10" y="-10" fill="#ffffff">r7</text>   <text id="t8"  x="-10" y="-10" fill="#ffffff">r8</text>
            <text id="t9"  x="-10" y="-10" fill="#ffffff">r9</text>   <text id="t10" x="-10" y="-10" fill="#ffffff">r10</text>  <text id="t11" x="-10" y="-10" fill="#ffffff">r11</text>
            <text id="t12" x="-10" y="-10" fill="#ffffff">r12</text>  <text id="t13" x="-10" y="-10" fill="#ffffff">r13</text>  <text id="t14" x="-10" y="-10" fill="#ffffff">r14</text>
            <text id="t15" x="-10" y="-10" fill="#ffffff">r15</text>  <text id="t16" x="-10" y="-10" fill="#ffffff">r16</text>  <text id="t17" x="-10" y="-10" fill="#ffffff">r17</text>
            <text id="t18" x="-10" y="-10" fill="#ffffff">r18</text>  <text id="t19" x="-10" y="-10" fill="#ffffff">r19</text>  <text id="t20" x="-10" y="-10" fill="#ffffff">r20</text>
            <text id="t21" x="-10" y="-10" fill="#ffffff">r21</text>  <text id="t22" x="-10" y="-10" fill="#ffffff">r22</text>  <text id="t23" x="-10" y="-10" fill="#ffffff">r23</text>
            <text id="t24" x="-10" y="-10" fill="#ffffff">r24</text>  <text id="t25" x="-10" y="-10" fill="#ffffff">r25</text>  <text id="t26" x="-10" y="-10" fill="#ffffff">r26</text>
            <text id="t27" x="-10" y="-10" fill="#ffffff">r27</text>  <text id="t28" x="-10" y="-10" fill="#ffffff">r28</text>  <text id="t29" x="-10" y="-10" fill="#ffffff">r29</text>

          <rect id="drag"     x="-10" y="-10" width="5"    height="5"    fill="none"    stroke="#ff0000" stroke-width="1"/>
          <rect id="interact" x="0"   y="0"   width="800"  height="800"  fill="#000000" opacity="0.1"
              onmousedown="${script('myonmousedown')}"
              onmousemove="${script('myonmousemove')}"
              onmouseup="${script('myonmouseup')}"
          />
        </svg>
    '''

    #
    # Scripts for JavaScript
    #
    _scripts={
        'render':'''
          state.drag_op     = false
          state.rects       = new Set();
          state.xa          = state.xb = state.ya = state.yb = 0;
          state.x0          = state.x1 = state.y0 = state.y1 = 0;
          state.r_lu        = new Map();
          state.t_lu        = new Map();
          state.r_lu['r0']  = r0;  state.t_lu['t0']  = t0;  state.r_lu['r1']  = r1;  state.t_lu['t1']  = t1;
          state.r_lu['r2']  = r2;  state.t_lu['t2']  = t2;  state.r_lu['r3']  = r3;  state.t_lu['t3']  = t3;
          state.r_lu['r4']  = r4;  state.t_lu['t4']  = t4;  state.r_lu['r5']  = r5;  state.t_lu['t5']  = t5;
          state.r_lu['r6']  = r6;  state.t_lu['t6']  = t6;  state.r_lu['r7']  = r7;  state.t_lu['t7']  = t7;
          state.r_lu['r8']  = r8;  state.t_lu['t8']  = t8;  state.r_lu['r9']  = r9;  state.t_lu['t9']  = t9;
          state.r_lu['r10'] = r10; state.t_lu['t10'] = t10; state.r_lu['r11'] = r11; state.t_lu['t11'] = t11;
          state.r_lu['r12'] = r12; state.t_lu['t12'] = t12; state.r_lu['r13'] = r13; state.t_lu['t13'] = t13;
          state.r_lu['r14'] = r14; state.t_lu['t14'] = t14; state.r_lu['r15'] = r15; state.t_lu['t15'] = t15;
          state.r_lu['r16'] = r16; state.t_lu['t16'] = t16; state.r_lu['r17'] = r17; state.t_lu['t17'] = t17;
          state.r_lu['r18'] = r18; state.t_lu['t18'] = t18; state.r_lu['r19'] = r19; state.t_lu['t19'] = t19;
          state.r_lu['r20'] = r20; state.t_lu['t20'] = t20; state.r_lu['r21'] = r21; state.t_lu['t21'] = t21;
          state.r_lu['r22'] = r22; state.t_lu['t22'] = t22; state.r_lu['r23'] = r23; state.t_lu['t23'] = t23;
          state.r_lu['r24'] = r24; state.t_lu['t24'] = t24; state.r_lu['r25'] = r25; state.t_lu['t25'] = t25;
          state.r_lu['r26'] = r26; state.t_lu['t26'] = t26; state.r_lu['r27'] = r27; state.t_lu['t27'] = t27;
          state.r_lu['r28'] = r28; state.t_lu['t28'] = t28; state.r_lu['r29'] = r29; state.t_lu['t29'] = t29;
        ''',
        'myonmousedown': '''
          remove_happened = false;
          for (const key of state.rects.keys()) {
              r_ptr = state.r_lu[key];
              x_r   = parseInt(r_ptr.getAttribute('x'));      y_r   = parseInt(r_ptr.getAttribute('y'));
              w_r   = parseInt(r_ptr.getAttribute('width'));  h_r   = parseInt(r_ptr.getAttribute('height'));
              contains_flag = (event.offsetX >= x_r) && (event.offsetX <= (x_r + w_r)) &&
                              (event.offsetY >= y_r) && (event.offsetY <= (y_r + h_r));
              if (contains_flag) {
                  r_ptr.setAttribute('x',      -10); r_ptr.setAttribute('y',      -10);
                  r_ptr.setAttribute('width',    5); r_ptr.setAttribute('height',   5);
                  t_ptr = state.t_lu['t'+key.substring(1)];
                  t_ptr.setAttribute('x',      -10); t_ptr.setAttribute('y',      -10);
                  remove_happened = true;
                  state.rects.delete(key)
                  self.updateExportString()
            }
          }
          if (remove_happened == false) {
              state.drag_op = true; state.x0 = state.x1 = event.offsetX; 
                                    state.y0 = state.y1 = event.offsetY; 
              self.drawDragOp();
          }
        ''',
        'myonmouseup':'''
          if (state.drag_op) {
            state.x1 = event.offsetX; state.y1 = event.offsetY; state.drag_op = false; self.resetDragOp();

            el_str = t_str = null;
            for (i=0;i<30;i++) {
              el_str = 'r' + i; t_str = 't' + i
              if (state.rects.has(el_str) == false) break;
            }
            
            if (el_str != null) {
              xa_i = Math.floor(state.xa/25.0); ya_i = Math.floor(state.ya/25.0);
              xb_i = Math.ceil (state.xb/25.0); yb_i = Math.ceil (state.yb/25.0);
              xa = Math.floor(25*(Math.floor(state.xa/25.0))); ya = Math.floor(25*(Math.floor(state.ya/25.0)));
              xb = Math.ceil (25*(Math.ceil (state.xb/25.0))); yb = Math.ceil (25*(Math.ceil (state.yb/25.0)));

              el_up = state.r_lu[el_str];
              if (el_up != null) {
                el_up.setAttribute('x',      xa);         el_up.setAttribute('y',      ya);
                el_up.setAttribute('width',  (xb - xa));  el_up.setAttribute('height', (yb - ya));
                el_up = state.t_lu[t_str];
                el_up.setAttribute('x',      xa+5);       el_up.setAttribute('y',      ya+20);
                state.rects.add(el_str)
                self.updateExportString()
              }
            }
          }
        ''',
        'myonmousemove':'''
          if (state.drag_op) { state.x1 = event.offsetX; state.y1 = event.offsetY; self.drawDragOp(); }
        ''',
        'drawDragOp':'''
          if (state.x0 < state.x1) { state.xa = state.x0; state.xb = state.x1; } else { state.xa = state.x1; state.xb = state.x0; }
          if (state.y0 < state.y1) { state.ya = state.y0; state.yb = state.y1; } else { state.ya = state.y1; state.yb = state.y0; }
          state.xa = Math.floor(25*(Math.floor(state.xa/25.0))); state.ya = Math.floor(25*(Math.floor(state.ya/25.0)));
          state.xb = Math.ceil (25*(Math.ceil (state.xb/25.0))); state.yb = Math.ceil (25*(Math.ceil (state.yb/25.0)));
          drag.setAttribute('x',      state.xa);               drag.setAttribute('y',      state.ya);
          drag.setAttribute('width',  (state.xb - state.xa));  drag.setAttribute('height', (state.yb - state.ya));
        ''',
        'resetDragOp':'''
          drag.setAttribute('x',      -10); drag.setAttribute('y',      -10);
          drag.setAttribute('width',    5); drag.setAttribute('height',   5);
        ''',
        'updateExportString':'''
            x0 = 1000; y0 = 1000;
            for (i=0;i<30;i++) {
              el_str = 'r' + i; t_str = 't' + i
              if (state.rects.has(el_str)) {
                x = parseInt(state.r_lu[el_str].getAttribute('x'));
                y = parseInt(state.r_lu[el_str].getAttribute('y'));
                if (x < x0) { x0 = x; } if (y < y0) { y0 = y; }
              }
            }
            s = '';
            for (i=0;i<30;i++) {
              el_str = 'r' + i; t_str = 't' + i;
              if (state.rects.has(el_str)) {
                x = parseInt(state.r_lu[el_str].getAttribute('x'));
                y = parseInt(state.r_lu[el_str].getAttribute('y'));
                w = parseInt(state.r_lu[el_str].getAttribute('width'));
                h = parseInt(state.r_lu[el_str].getAttribute('height'));
                s += '(' + ((x-x0)/25) + ',' + ((y-y0)/25) + ',' + (w/25) + ',' + (h/25) + ')';
                s += ':' + '("' + el_str + '", {}),|';
              }
            }
            data.export_string = s;
        '''
    }

