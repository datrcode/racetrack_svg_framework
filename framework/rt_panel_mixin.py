# Copyright 2023 David Trimm
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
from panel.reactive import ReactiveHTML
import param

from rt_layouts_mixin import RTLayout

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
        pn.extension(inline=False)

    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------

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
    # Create an interactive panel
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
        return RTReactiveHTML(df, self, spec, w, h, rt_params, h_gap, v_gap, widget_h_gap, widget_v_gap, **kwargs)

#
# ReactiveHTML Class for Panel Implementation
#
class RTReactiveHTML(ReactiveHTML):
    #
    # Inner Modification for RT SVG Render
    #
    # Initial Picture Is A Computer Mouse:  Source & License:
    #
    # https://www.svgrepo.com/svg/24318/computer-mouse
    #
    # https://www.svgrepo.com/page/licensing/#CC0
    #
    mod_inner = param.String(default="""
<svg fill="#000000" version="1.1" id="Capa_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
	 width="800px" height="800px" viewBox="0 0 800 800" xml:space="preserve">
  <rect x="0" y="0" width="800" height="800" fill="#ffffff"/> <g> <g>
		<path d="M25.555,11.909c-1.216,0-2.207,1.963-2.207,4.396c0,2.423,0.991,4.395,2.207,4.395c1.208,0,2.197-1.972,2.197-4.395
			C27.751,13.872,26.762,11.909,25.555,11.909z"/>
		<path d="M18.22,5.842c4.432,0,6.227,0.335,6.227,3.653h2.207c0-5.851-4.875-5.851-8.433-5.851c-4.422,0-6.227-0.326-6.227-3.644
			H9.795C9.795,5.842,14.671,5.842,18.22,5.842z"/>
		<path d="M29.62,9.495c0.209,0.632,0.331,1.315,0.331,2.031v9.548c0,2.681-1.562,4.91-3.608,5.387
			c0.004,0.031,0.021,0.059,0.021,0.1v7.67c0,0.445-0.363,0.81-0.817,0.81c-0.445,0-0.809-0.365-0.809-0.81v-7.67
			c0-0.041,0.019-0.068,0.022-0.1c-2.046-0.477-3.609-2.706-3.609-5.387v-9.548c0-0.715,0.121-1.399,0.331-2.031
			c-6.057,1.596-10.586,7.089-10.586,13.632v12.716c-0.001,7.787,6.37,14.158,14.155,14.158h0.999
			c7.786,0,14.156-6.371,14.156-14.158V23.127C40.206,16.584,35.676,11.091,29.62,9.495z"/>
	</g> </g> </svg>
    """)

    #
    # Panel Template
    # - The following is re-written in the constructor
    #
    _template = """
        <svg id="parent" width="1280" height="256">
            <svg id="mod" width="1280" height="256">
                ${mod_inner}
            </svg>
            <rect id="drag" x="-10" y="-10" width="5" height="5" fill="#ffffff" opacity="0.6" />
            <rect id="screen" x="0" y="0" width="100" height="100" opacity="0.05" 
              onmousedown="${script('myonmousedown')}"
              onmousemove="${script('myonmousemove')}"
              onmouseup="${script('myonmouseup')}"
            />
        </svg>
    """
        
    #
    # Constructor
    #
    def __init__(self,
                 df,
                 rt_self,
                 spec,                # Layout specification
                 w,                   # Width of the panel
                 h,                   # Heght of the panel
                 rt_params      = {}, # Racetrack params -- dictionary of param=value
                 # ------------------ #
                 h_gap          = 0,  # Horizontal left/right gap
                 v_gap          = 0,  # Verticate top/bottom gap
                 widget_h_gap   = 1,  # Horizontal gap between widgets
                 widget_v_gap   = 1,  # Vertical gap between widgets
                 **kwargs):
        # Setup specific instance information
        # - Copy the member variables
        self.rt_self      = rt_self
        self.spec         = spec
        self.w            = w
        self.h            = h
        self.rt_params    = rt_params
        self.h_gap        = h_gap
        self.v_gap        = v_gap
        self.widget_h_gap = widget_h_gap
        self.widget_v_gap = widget_v_gap
        self.kwargs       = kwargs
        if   self.rt_self.isPandas(df):                     
            df = df.copy()
        elif self.rt_self.isPolars(df):
            df = df.clone()
        self.dfs        = [df]            
                     
        # - Create the template ... copy of the above with variables filled in...
        self._template = f'<svg id="parent" width="{w}" height="{h}">'                               + \
                            f'<svg id="mod" width="{w}" height="{h}">'                               + \
                                """\n${mod_inner}\n"""                                               + \
                            '</svg>'                                                                 + \
                            '<rect id="drag" x="-10" y="-10" width="5" height="5" stroke="#000000" ' + \
                                  'fill="#ffffff" opacity="0.6" />'                                  + \
                            f'<rect id="screen" x="0" y="0" width="{w}" height="{h}" opacity="0.05"' + \
                            """ onmousedown="${script('myonmousedown')}"   """                       + \
                            """ onmousemove="${script('myonmousemove')}"   """                       + \
                            """ onmouseup="${script('myonmouseup')}"       """                       + \
                            '/>'                                                                     + \
                         '</svg>'
        self.dfs_layout = [self.rt_self.layout(self.spec, df, w=self.w, h=self.h,
                                               h_gap=self.h_gap,v_gap=self.v_gap,
                                               widget_h_gap=self.widget_h_gap,widget_v_gap=self.widget_v_gap,
                                               track_state=True, rt_reactive_html=self,
                                               **self.rt_params)]
        self.mod_inner = self.dfs_layout[0]._repr_svg_()

        # - Create a lock for threading
        self.lock = threading.Lock()

        # Execute the super initialization
        super().__init__(**kwargs)

        # Watch for callbacks
        self.param.watch(self.applyDragOp, 'drag_op_finished')

        # Viz companions for sync
        self.companions = []
    
    #
    # Return the visible dataframe.
    #
    def visibleDataFrame(self):
        return self.dfs[-1]
    
    def register_companion_viz(self, viz):
        self.companions.append(viz)
    
    def unregister_companion_viz(self, viz):
        if viz in self.companions:
            self.companions.remove(viz)

    #
    # Drag operation state & method
    #
    drag_op_finished = param.Boolean(default=False)
    drag_x0          = param.Integer(default=0)
    drag_y0          = param.Integer(default=0)
    drag_x1          = param.Integer(default=10)
    drag_y1          = param.Integer(default=10)
    drag_shiftkey    = param.Boolean(default=False)
    async def applyDragOp(self,event):
        self.lock.acquire()
        try:
            if self.drag_op_finished:
                _x0,_y0,_x1,_y1 = self.drag_x0, self.drag_y0, self.drag_x1, self.drag_y1
                if _x0 == _x1:
                    _x1 += 1
                if _y0 == _y1:
                    _y1 += 1
                _df = self.dfs_layout[-1].overlappingDataFrames((_x0,_y0,_x1,_y1))
                # Go back up the stack...
                if _df is None or len(_df) == 0:
                    if len(self.dfs) > 1:
                        self.dfs        = self.dfs    [:-1]
                        self.dfs_layout = self.dfs_layout[:-1]
                        self.mod_inner  = self.dfs_layout[-1]._repr_svg_()

                        # ascend stack for all registered companion vizs
                        for c in self.companions:
                            if isinstance(c, RTReactiveHTML):
                                if len(c.dfs) > 1:
                                    c.dfs = c.dfs[:-1]
                                    c.dfs_layout = c.dfs_layout[:-1]
                                    c.mod_inner  = c.dfs_layout[-1]._repr_svg_()

                # Filter and go down the stack
                else:
                    # Remove data option...
                    if len(self.dfs) > 0 and self.drag_shiftkey:
                        _df = self.dfs[-1].query('index not in @_df.index')

                    # Make sure we still have data...
                    if len(_df) > 0:
                        # Re-layout w/ new dataframe
                        _layout = self.rt_self.layout(self.spec,                      _df,
                                                      w=self.w,                       h=self.h, 
                                                      h_gap=self.h_gap,               v_gap=self.v_gap,
                                                      widget_h_gap=self.widget_h_gap, widget_v_gap=self.widget_v_gap,
                                                      track_state=True, rt_reactive_html=self,
                                                      **self.rt_params)
                        # Update the stack
                        self.dfs.       append(_df)
                        self.dfs_layout.append(_layout)
                        self.mod_inner = _layout._repr_svg_()

                        # adjust layout for all registered companion vizs
                        for c in self.companions:
                            if isinstance(c, RTReactiveHTML):
                                _clayout = c.rt_self.layout(c.spec,
                                                            _df,
                                                            w=c.w,
                                                            h=c.h,
                                                            h_gap=c.h_gap,
                                                            v_gap=c.v_gap,
                                                            widget_h_gap=c.widget_h_gap,
                                                            widget_v_gap=c.widget_v_gap,
                                                            track_state=True,
                                                            rt_reactive_html=self,
                                                            **c.rt_params)
                                c.dfs.append(_df)
                                c.dfs_layout.append(_clayout)
                                c.mod_inner = _clayout._repr_svg_()

                # Mark operation as finished
                self.drag_op_finished = False
        finally:
            self.lock.release()

    #
    # Panel Javascript Definitions
    #
    _scripts = {
        'render':"""
            mod.innerHTML = data.mod_inner;
            state.x0_drag  = state.y0_drag = -10;
            state.x1_drag  = state.y1_drag =  -5;
            state.shiftkey = false;
            state.drag_op  = false;            
        """,
        'myonmousemove':"""
            if (state.drag_op) {
                state.x1_drag  = event.offsetX;
                state.y1_drag  = event.offsetY;
                state.shiftkey = event.shiftKey;
                self.myUpdateDragRect();
            }
        """,
        'myonmousedown':"""
            state.x0_drag  = event.offsetX;
            state.y0_drag  = event.offsetY;
            state.x1_drag  = event.offsetX+1;
            state.y1_drag  = event.offsetY+1;
            state.drag_op  = true;
            state.shiftkey = event.shiftKey;
            self.myUpdateDragRect();
        """,
        'myonmouseup':"""
            if (state.drag_op) {
                state.x1_drag  = event.offsetX;
                state.y1_drag  = event.offsetY;
                state.shiftkey = event.shiftKey;
                state.drag_op  = false;
                self.myUpdateDragRect();
                data.drag_x0          = state.x0_drag;
                data.drag_y0          = state.y0_drag;
                data.drag_x1          = state.x1_drag;
                data.drag_y1          = state.y1_drag;
                data.drag_shiftkey    = state.shiftkey
                data.drag_op_finished = true;
            }
        """,
        'mod_inner':"""
            mod.innerHTML = data.mod_inner;
        """,
        'myUpdateDragRect':"""
            if (state.drag_op) {
                x = state.x0_drag; 
                if (state.x1_drag < x) { x = state.x1_drag; }
                y = state.y0_drag; 
                if (state.y1_drag < y) { y = state.y1_drag; }
                w = Math.abs(state.x1_drag - state.x0_drag)
                h = Math.abs(state.y1_drag - state.y0_drag)
                drag.setAttribute('x',x);     drag.setAttribute('y',y);
                drag.setAttribute('width',w); drag.setAttribute('height',h);
                if (state.shiftkey) { drag.setAttribute('stroke','#ff0000'); }
                else                { drag.setAttribute('stroke','#000000'); }
            } else {
                drag.setAttribute('x',-10);   drag.setAttribute('y',-10);
                drag.setAttribute('width',5); drag.setAttribute('height',5);
            }
        """
    }

