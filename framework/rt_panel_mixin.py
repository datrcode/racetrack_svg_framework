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
    #
    def __panel_mixin_init__(self):
        pn.extension(inline=False)

    #
    # Create an interactive panel
    #
    def interactivePanel(self,
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
        return RTReactiveHTML(self, spec, w, h, rt_params, h_gap, v_gap, widget_h_gap, widget_v_gap, **kwargs)

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
                 w,                   # Width of the panel
                 h,                   # Heght of the panel
                 rt_params,           # Racetrack params -- dictionary of param=value
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
        # - Create the template ... copy of the above with variables filled in...
        self._template = f'<svg id="parent" width="{w}" height="{h}">'                               + \
                            f'<svg id="mod" width="{w}" height="{h}">'                               + \
                                """\n${mod_inner}\n"""                                               + \
                            '</svg>'                                                                 + \
                            '<rect id="drag" x="-10" y="-10" width="5" height="5" stroke="#000000" ' + \
                                  'fill="#ffffff" opacity="0.6" />'                                  + \
                            f'<rect id="screen" x="0" y="0" width="{w}" height="{h}" opacity="0.05"' + \
                            """ onmousedown="${script('_onmousedown_')}"   """                       + \
                            """ onmousemove="${script('_onmousemove_')}"   """                       + \
                            """ onmouseup="${script('_onmouseup_')}"       """                       + \
                            '/>'                                                                     + \
                         '</svg>' 
        # - Assign place holders
        self.dfs        = [pd.DataFrame()]
        self.dfs_layout = [RTLayout(rt_self,{},str(self.mod_inner))]
        self.mod_inner  = self.dfs_layout[0]._repr_svg_()

        # - Create a lock for threading
        self.lock = threading.Lock()
        
        # Execute the super initialization
        super().__init__(**kwargs)
        # Watch for callbacks
        self.param.watch(self.applyDragOp,'drag_op_finished')
        self.param.trigger('drag_op_finished')
    
    #
    # Set the root dataframe... because I can't figure out how to do this in the constructor
    #
    def setRoot(self, df):
        self.dfs        = [df.copy()]
        self.dfs_layout = [self.rt_self.layout(self.spec, df, w=self.w, h=self.h,
                                            h_gap=self.h_gap,v_gap=self.v_gap,
                                            widget_h_gap=self.widget_h_gap,widget_v_gap=self.widget_v_gap,
                                            track_state=True,
                                            **self.rt_params)]
        self.mod_inner = self.dfs_layout[0]._repr_svg_()

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
                                                      track_state=True,
                                                      **self.rt_params)
                        # Update the stack
                        self.dfs.       append(_df)
                        self.dfs_layout.append(_layout)
                        self.mod_inner = _layout._repr_svg_()

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
        '_onmousemove_':"""
            if (state.drag_op) {
                state.x1_drag  = event.offsetX;
                state.y1_drag  = event.offsetY;
                state.shiftkey = event.shiftKey;
                self._updateDragRect_();
            }
        """,
        '_onmousedown_':"""
            state.x0_drag  = event.offsetX;
            state.y0_drag  = event.offsetY;
            state.x1_drag  = event.offsetX+1;
            state.y1_drag  = event.offsetY+1;
            state.drag_op  = true;
            state.shiftkey = event.shiftKey;
            self._updateDragRect_();
        """,
        '_onmouseup_':"""
            if (state.drag_op) {
                state.x1_drag  = event.offsetX;
                state.y1_drag  = event.offsetY;
                state.shiftkey = event.shiftKey;
                state.drag_op  = false;
                self._updateDragRect_();
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
                if (state.shiftkey) { drag.setAttribute('stroke','#ff0000'); }
                else                { drag.setAttribute('stroke','#000000'); }
            } else {
                drag.setAttribute('x',-10);   drag.setAttribute('y',-10);
                drag.setAttribute('width',5); drag.setAttribute('height',5);
            }
        """
    }

