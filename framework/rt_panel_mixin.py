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

import panel as pn
from panel.reactive import ReactiveHTML
import param

__name__ = 'rt_panel_mixin'

#
# Panel Mixin
#
class RTPanelMixin(object):
    #
    # Constructor
    #
    def __panel_mixin_init__(self):
        pn.extension(inline=False)

    #
    # Create an interactive panel
    #
    def interactivePanel(self,
                         spec,                # Layout specification
                         df,                  # Root dataframe
                         w,                   # Width of the panel
                         h,                   # Heght of the panel
                         # ------------------ #
                         h_gap          = 0,  # Horizontal left/right gap
                         v_gap          = 0,  # Verticate top/bottom gap
                         widget_h_gap   = 1,  # Horizontal gap between widgets
                         widget_v_gap   = 1,  # Vertical gap between widgets
                         **kwargs):           # Other arguments to pass to the layout instance
        return RTReactiveHTML(self, spec, df, w, h, h_gap, v_gap, widget_h_gap, widget_v_gap, **kwargs)

#
# ReactiveHTML Class for Panel Implementation
#
class RTReactiveHTML(ReactiveHTML):
    #
    # Inner Modification for RT SVG Render
    # - The following is re-written in the constructor
    #
    mod_inner = param.String(default="""
        <rect x="0" y="0" width="100" height="100" fill="#808080" />
        <circle cx="50" cy="50" r="40" fill="#000000" />
    """)

    #
    # Panel Template
    # - The following is re-written in the constructor
    #
    _template = """
        <svg id="parent" width="100" height="100">
            <svg id="mod" width="100" height="100">
                ${mod_inner}
            </svg>
            <rect id="drag" x="-10" y="-10" width="5" height="5" fill="#ffffff" opacity="0.6" />
            <rect id="screen" x="0" y="0" width="100" height="100" opacity="0.05" 
              onmousedown="${script('_onmousedown_')}"
              onmousemove="${script('_onmousemove_')}"
              onmouseup="${script('_onmouseup_')}"              
            />
        </svg>
    """
        
    #
    # Constructor
    #
    def __init__(self,
                 rt_self,
                 spec,                # Layout specification
                 df,                  # Root dataframe
                 w,                   # Width of the panel
                 h,                   # Heght of the panel
                 # ------------------ #
                 h_gap          = 0,  # Horizontal left/right gap
                 v_gap          = 0,  # Verticate top/bottom gap
                 widget_h_gap   = 1,  # Horizontal gap between widgets
                 widget_v_gap   = 1,  # Vertical gap between widgets
                 **kwargs):
        # Setup specific instance information
        # - Copy the member variables
        self.spec         = spec
        self.df_root      = df
        self.w            = w
        self.h            = h
        self.h_gap        = h_gap
        self.v_gap        = v_gap
        self.widget_h_gap = widget_h_gap
        self.widget_v_gap = widget_v_gap
        self.kwargs       = kwargs 
        # - Create the template ... copy of the above with variables filled in...
        self._template = f'<svg id="parent" width="{w}" height="{h}">'                                              + \
                            f'<svg id="mod" width="{w}" height="{h}">'                                              + \
                                """\n${mod_inner}\n"""                                                                  + \
                            '</svg>'                                                                                + \
                            '<rect id="drag" x="-10" y="-10" width="5" height="5" fill="#ffffff" opacity="0.6" />'  + \
                            f'<rect id="screen" x="0" y="0" width="{w}" height="{h}" opacity="0.05"'                + \
                            """ onmousedown="${script('_onmousedown_')}" """                                        + \
                            """ onmousemove="${script('_onmousemove_')}" """                                        + \
                            """ onmouseup="${script('_onmouseup_')}"     """                                        + \
                            '/>'                                                                                    + \
                         '</svg>' 
        # - Create the base SVG
        self.mod_inner = rt_self.layout(spec,df,w=w,h=h,h_gap=h_gap,v_gap=v_gap,
                                        widget_h_gap=widget_h_gap,widget_v_gap=widget_v_gap,
                                        **kwargs)
        # Execute the super initialization
        super().__init__(**kwargs)
        # Watch for callbacks
        self.param.watch(self.applyDragOp,'drag_op_finished')
        self.param.trigger('drag_op_finished')
    
    drag_op_finished = param.Boolean(default=False)    
    async def applyDragOp(self,event):
        if self.drag_op_finished:
            self.drag_op_finished = False
        
    _scripts = {
        'render':"""
            mod.innerHTML = data.mod_inner;
            state.x0_drag = state.y0_drag = -10;
            state.x1_drag = state.y1_drag =  -5;
            state.drag_op = false;            
        """,
        '_onmousemove_':"""
            if (state.drag_op) {
                state.x1_drag = event.offsetX;
                state.y1_drag = event.offsetY;
                self._updateDragRect_();
            }
        """,
        '_onmousedown_':"""
            state.x0_drag = event.offsetX;
            state.y0_drag = event.offsetY;
            state.x1_drag = event.offsetX+1;
            state.y1_drag = event.offsetY+1;
            state.drag_op = true;
            self._updateDragRect_();
        """,
        '_onmouseup_':"""
            if (state.drag_op) {
                state.x1_drag = event.offsetX;
                state.y1_drag = event.offsetY;            
                state.drag_op = false;
                self._updateDragRect_();
                data.drag_op_finished = true;
            }
        """,
        'mod_inner':"""
            mod.innerHTML = data.mod_inner;
        """,
        '_updateDragRect_':"""
            if (state.drag_op) {
                x = state.x0_drag; 
                if (state.x1_drag < x) { x = state.x1_drag; }
                y = state.y0_drag; 
                if (state.y1_drag < y) { y = state.y1_drag; }
                w = Math.abs(state.x1_drag - state.x0_drag)
                h = Math.abs(state.y1_drag - state.y0_drag)
                drag.setAttribute('x',x);     drag.setAttribute('y',y);
                drag.setAttribute('width',w); drag.setAttribute('height',h);
            } else {
                drag.setAttribute('x',-10);   drag.setAttribute('y',-10);
                drag.setAttribute('width',5); drag.setAttribute('height',5);
            }
        """
    }

