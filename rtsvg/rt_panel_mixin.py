# Copyright 2024 David Trimm
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

from .rt_layouts_mixin import RTComponentsLayout

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
    # interactiveGraphLayout()
    #
    def interactiveGraphLayout(self, df, ln_params, w=600, h=400, **kwargs):
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

        Use saveLayout() and loadLayout() to store and retrieve layouts in parquet format
        '''
        return RTGraphInteractiveLayout(self, df, ln_params, w, h, **kwargs)

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
        h_gap, v_gap, widget_h_gap, widget_v_gap : int
            Horizontal, vertical, and widget gap between views
        '''
        return RTReactiveHTML(df, self, spec, w, h, rt_params, h_gap, v_gap, widget_h_gap, widget_v_gap, **kwargs)

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
              onmousewheel="${script('myonmousewheel')}"
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
        self.df           = self.rt_self.copyDataFrame(df)
        self.df_level     = 0
        self.dfs          = [df]

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
                            """ onmousewheel="${script('myonmousewheel')}" """                       + \
                            '/>'                                                                     + \
                         '</svg>'
        self.dfs_layout = []
        self.dfs_layout.append(self.__createLayout__(df))
        self.mod_inner = self.dfs_layout[0]._repr_svg_()

        # - Create a lock for threading
        self.lock = threading.Lock()

        # Execute the super initialization
        super().__init__(**kwargs)

        # Watch for callbacks
        self.param.watch(self.applyDragOp,   'drag_op_finished')
        self.param.watch(self.applyWheelOp,  'wheel_op_finished')
        self.param.watch(self.applyMiddleOp, 'middle_op_finished')

        # Viz companions for sync
        self.companions = []
    
    #
    # __createLayout__() - create the layout for the specified dataframe
    #
    def __createLayout__(self, __df__):
        _layout_ = self.rt_self.layout(self.spec, __df__, w=self.w, h=self.h, h_gap=self.h_gap, v_gap=self.v_gap,
                                       widget_h_gap=self.widget_h_gap, widget_v_gap=self.widget_v_gap,
                                       track_state=True, rt_reactive_html=self, **self.rt_params)
        if len(self.dfs_layout) > 0: # Doesn't exist at the very first layout level
            _layout_.applyViewConfigurations(self.dfs_layout[0]) # Apply any adjustments to the views that have occurred
        return _layout_
    #
    # Return the visible dataframe.
    #
    def visibleDataFrame(self):
        return self.dfs[self.df_level]
    
    def register_companion_viz(self, viz):
        self.companions.append(viz)
    
    def unregister_companion_viz(self, viz):
        if viz in self.companions:
            self.companions.remove(viz)

    #
    # Middle button state & method
    #
    x0_middle          = param.Integer(default=0)
    y0_middle          = param.Integer(default=0)
    x1_middle          = param.Integer(default=0)
    y1_middle          = param.Integer(default=0)
    middle_op_finished = param.Boolean(default=False)
    async def applyMiddleOp(self,event):
        self.lock.acquire()
        try:
            if self.middle_op_finished:
                x0, y0, x1, y1 = self.x0_middle, self.y0_middle, self.x1_middle, self.y1_middle
                dx, dy         = x1 - x0, y1 - y0
                _comp_ , _key_ , _adj_coordinate_ = self.dfs_layout[self.df_level].identifyComponent((x0,y0))
                if _comp_ is not None:
                    if (abs(self.x0_middle - self.x1_middle) <= 1) and (abs(self.y0_middle - self.y1_middle) <= 1):
                        if _comp_.applyMiddleClick(_adj_coordinate_):
                            self.mod_inner  = self.dfs_layout[self.df_level]._repr_svg_() # Re-render current
                    else:
                        if _comp_.applyMiddleDrag(_adj_coordinate_, (dx,dy)):
                            self.mod_inner  = self.dfs_layout[self.df_level]._repr_svg_() # Re-render current
        finally:
            self.middle_op_finished = False
            self.lock.release()

    #
    # Wheel operation state & method
    #
    wheel_x           = param.Integer(default=0)
    wheel_y           = param.Integer(default=0)
    wheel_rots        = param.Integer(default=0) # Mult by 10 and rounded...
    wheel_op_finished = param.Boolean(default=False)
    async def applyWheelOp(self,event):
        self.lock.acquire()
        try:
            if self.wheel_op_finished:
                x, y, rots = self.wheel_x, self.wheel_y, self.wheel_rots
                if rots != 0:
                    # Find the compnent where the scroll event occurred
                    _comp_ , _key_ , _adj_coordinate_ = self.dfs_layout[self.df_level].identifyComponent((x,y))
                    if _comp_ is not None:
                        if _comp_.applyScrollEvent(rots, _adj_coordinate_):
                            # Re-render current
                            self.mod_inner  = self.dfs_layout[self.df_level]._repr_svg_()
                            # Propagate the view configuration to the same component across the dataframe stack
                            for i in range(len(self.dfs_layout)):
                                if i != self.df_level:
                                    _comp_stack_ = self.dfs_layout[i].componentInstance(_key_)
                                    if _comp_stack_ is not None:
                                        _comp_stack_.applyViewConfiguration(_comp_)

        finally:
            self.wheel_op_finished = False
            self.wheel_rots        = 0            
            self.lock.release()

    #
    # popStack() - as long as there are items on the stack, go up the stack
    #
    def popStack(self, callers=None):
        if self.df_level == 0: return
        if callers is not None and self in callers: return
        if callers is None: callers = set([self])
        else:               callers.add(self)
        # ascend this panel's stack
        self.df_level   = self.df_level - 1
        self.mod_inner  = self.dfs_layout[self.df_level]._repr_svg_()
        # ascend stack for all registered companion vizs
        for c in self.companions:
            if isinstance(c, RTReactiveHTML) or isinstance(c, RTGraphInteractiveLayout): c.popStack(callers=callers)

    #
    # setStackPosition() - set to a specific position
    #
    def setStackPosition(self, i_found, callers=None):
        if i_found < 0 or i_found >= len(self.dfs_layout): return
        if callers is not None and self in callers: return
        if callers is None: callers = set([self])
        else:               callers.add(self)
        # set the specific stack level
        self.df_level   = i_found
        self.mod_inner  = self.dfs_layout[self.df_level]._repr_svg_()
        # do the same for the companions
        for c in self.companions:
            if isinstance(c, RTReactiveHTML) or isinstance(c, RTGraphInteractiveLayout): c.setStackPostion(i_found, callers=callers)

    #
    # pushStack() - push a dataframe onto the stack
    #
    def pushStack(self, df, callers=None):
        if callers is not None and self in callers: return
        if callers is None: callers = set([self])
        else:               callers.add(self)

        # Re-layout w/ new dataframe
        self.dfs         = self.dfs       [:(self.df_level+1)]
        self.dfs_layout  = self.dfs_layout[:(self.df_level+1)]
        self.df_level   += 1
        _layout = self.__createLayout__(df)
        # Update the stack
        self.dfs       .append(df)
        self.dfs_layout.append(_layout)
        self.mod_inner = _layout._repr_svg_()

        # adjust layout for all registered companion vizs
        for c in self.companions:
            if isinstance(c, RTReactiveHTML) or isinstance(c, RTGraphInteractiveLayout): c.pushStack(df, callers=callers)

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
                _df = self.dfs_layout[self.df_level].overlappingDataFrames((_x0,_y0,_x1,_y1))
                # Go back up the stack...
                if _df is None or len(_df) == 0:
                    if self.df_level > 0: self.popStack()
                # Filter and go down the stack
                else:
                    # Align the dataframes if necessary
                    if   self.rt_self.isPandas(_df):
                        if self.df.columns.equals(_df.columns) == False:
                            _df = _df.drop(set(_df.columns) - set(self.df.columns), axis=1)
                    elif self.rt_self.isPolars(_df):
                        if set(_df.columns) != set(self.df.columns):
                            _df = _df.drop(set(_df.columns) - set(self.df.columns))

                    # Remove data option...
                    if self.df_level > 0 and self.drag_shiftkey:
                        if   self.rt_self.isPandas(self.df):
                            _df = self.dfs[self.df_level].query('index not in @_df.index')
                        elif self.rt_self.isPolars(self.df):
                            _df = self.dfs[self.df_level].join(_df, on=_df.columns, how='anti') # May not correctly consider non-unique rows...
                        else:
                            raise Exception('RTPanel.applyDragOp() - only pandas and polars supported')

                    # Make sure we still have data...
                    if len(_df) > 0:
                        # See if the dataframe is already in the stack
                        i_found = None
                        if   self.rt_self.isPandas(self.df):
                            for i in range(len(self.dfs)):
                                if len(self.dfs[i]) == len(_df) and self.dfs[i].equals(_df):
                                    i_found = i
                                    break
                        elif self.rt_self.isPolars(self.df):
                            for i in range(len(self.dfs)):
                                if len(self.dfs[i]) == len(_df) and self.dfs[i].frame_equal(_df):
                                    i_found = i
                                    break

                        # Dataframe already in the stack...  go to that stack position
                        if i_found is not None: 
                            self.setStackPosition(i_found)
                        # Push a new dataframe onto the stack
                        else:
                            self.pushStack(_df)

                # Mark operation as finished
                self.drag_op_finished = False
        finally:
            self.lock.release()

    #
    # Panel Javascript Definitions
    #
    _scripts = {
        'render':"""
            mod.innerHTML  = data.mod_inner;
            state.x0_drag  = state.y0_drag = -10;
            state.x1_drag  = state.y1_drag =  -5;
            state.shiftkey = false;
            state.drag_op  = false;
            data.middle_op_finished = false;
        """,
        'myonmousemove':"""
            event.preventDefault();
            if (state.drag_op) {
                state.x1_drag  = event.offsetX;
                state.y1_drag  = event.offsetY;
                state.shiftkey = event.shiftKey;
                self.myUpdateDragRect();
            }
        """,
        'myonmousedown':"""
            event.preventDefault();
            if (event.button == 0) {
                state.x0_drag  = event.offsetX;
                state.y0_drag  = event.offsetY;
                state.x1_drag  = event.offsetX+1;
                state.y1_drag  = event.offsetY+1;
                state.drag_op  = true;
                state.shiftkey = event.shiftKey;
                self.myUpdateDragRect();
            } else if (event.button == 1) {
                data.x0_middle = data.x1_middle = event.offsetX;
                data.y0_middle = data.y1_middle = event.offsetY;
            }
        """,
        'myonmouseup':"""
            event.preventDefault();
            if (state.drag_op && event.button == 0) {
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
            } else if (event.button == 1) {
                data.x1_middle          = event.offsetX;
                data.y1_middle          = event.offsetY;
                data.middle_op_finished = true;                
            }
        """,
        'myonmousewheel':"""
            event.preventDefault();
            data.wheel_x           = event.offsetX;
            data.wheel_y           = event.offsetY;
            data.wheel_rots        = Math.round(10*event.deltaY);
            data.wheel_op_finished = true;
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

#
# ReactiveHTML Class for Panel Implementation
#
class RTGraphInteractiveLayout(ReactiveHTML):
    #
    # Print Representation
    #
    def __str__(self): return """
-------------------------------------------------
Interactivity Key Commands
----+--------------------------------------------
c   | reset view or focus view on selected
C   | focus view on selected + neighbors
e   | expand selection
E   | expand selection (directed graph)
g   | next mouse press is layout
G   | cycle through layout modes
n   | cycle through node selection modes
p   | keep only selected nodes (push stack)
P   | pop stack
q   | invert selection
Q   | common neighbors
s   | set sticky labels
S   | remove sticky labels from selected
t   | consolidate all nodes at the mouse location
w   | add selected nodes to sticky labels
W   | cycle label visibility (all | sticky | none)
y   | vertical layout
Y   | horizontal layout
1-6 | select numbered degree
7   | select degree 10 -> 20
8   | select degree 20 -> 50
9   | select degree 50 -> 100
0   | select degree 100 -> inf    
    """
    #
    # Inner Modification for RT SVG Render
    #
    mod_inner = param.String(default="""<circle cx="300" cy="200" r="10" fill="red" />""")

    #
    # All Entities Path
    #
    allentitiespath = param.String(default="M -100 -100 l 10 0 l 0 10 l -10 0 l 0 -10 Z")

    #
    # Selection Path
    #
    selectionpath = param.String(default="M -100 -100 l 10 0 l 0 10 l -10 0 l 0 -10 Z")

    #
    # Information String
    #
    info_str              = param.String(default=" | | grid")

    #
    # Operation String
    #
    op_str                = param.String(default="Select")

    #
    # Layout Mode String
    #
    layout_mode           = param.String(default="grid")

    #
    # Panel Template
    # - rewritten in constructor with width and height filled in
    #
    _template = """
<svg id="svgparent" width="600" height="400" tabindex="0" 
     onkeypress="${script('keyPress')}" onkeydown="${script('keyDown')}" onkeyup="${script('keyUp')}">
    <svg id="mod" width="600" height="400"> ${mod_inner} </svg>
    <rect id="drag" x="-10" y="-10" width="5" height="5" stroke="#000000" stroke-width="2" fill="none" />
    <line   id="layoutline"      x1="-10" y1="-10" x2="-10"    y2="-10"    stroke="#000000" stroke-width="2" />
    <rect   id="layoutrect"      x="-10"  y="-10"  width="10"  height="10" stroke="#000000" stroke-width="2" />
    <circle id="layoutcircle"    cx="-10" cy="-10" r="5"       fill="none" stroke="#000000" stroke-width="6" />
    <circle id="layoutsunflower" cx="-10" cy="-10" r="5"                   stroke="#000000" stroke-width="2" />
    <rect id="screen" x="0" y="0" width="600" height="400" opacity="0.05"
          onmousedown="${script('downSelect')}"          onmousemove="${script('moveEverything')}"
          onmouseup="${script('upEverything')}"          onmousewheel="${script('mouseWheel')}" />
    <text id="opstr"   x="598" y="12"  fill="#000000" font-size="10px" text-anchor="end"> ${op_str} </text>
    <text id="infostr" x="5"   y="398" fill="#000000" font-size="10px"> ${info_str} </text>
    <path id="allentitieslayer" d="${allentitiespath}" fill="#000000" fill-opacity="0.01" stroke="none"
          onmousedown="${script('downAllEntities')}" onmousemove="${script('moveEverything')}" 
          onmouseup="${script('upEverything')}"      onmousewheel="${script('mouseWheel')}" />
    <path id="selectionlayer" d="${selectionpath}" fill="#ff0000" transform="" stroke="none"
          onmousedown="${script('downMove')}"        onmousemove="${script('moveEverything')}"
          onmouseup="${script('upEverything')}"      onmousewheel="${script('mouseWheel')}" />
</svg>
"""

    #
    # Constructor
    #
    def __init__(self,
                 rt_self,           # RACETrack instance
                 df,                # data frame
                 ln_params,         # linknode params
                 w           =600,  # width
                 h           =400,  # height
                 **kwargs):
        # Setup specific instance information
        # - Copy the member variables
        self.rt_self           = rt_self
        self.ln_params         = ln_params
        if 'pos' not in ln_params.keys(): ln_params['pos'] = {}
        self.pos               = ln_params['pos']
        self.w                 = w
        self.h                 = h
        self.kwargs            = kwargs
        self.df                = self.rt_self.copyDataFrame(df)
        self.df_level          = 0
        self.dfs               = [self.df]
        self.dfs_layout        = [self.__renderView__(self.df)]
        self.graphs            = [self.rt_self.createNetworkXGraph(self.df, ln_params['relationships'])]
        self.mod_inner         = self.dfs_layout[self.df_level]._repr_svg_()
        self.allentitiespath   = self.dfs_layout[self.df_level].__createPathDescriptionForAllEntities__()
        if 'draw_labels' in ln_params and ln_params['draw_labels']: self.label_mode    = 'all labels'
        else:                                                       self.label_mode    = 'no labels'
        if 'label_only' in ln_params:                               self.sticky_labels = set(ln_params['label_only'])
        else:                                                       self.sticky_labels = set()
        self.selected_entities = set(self.sticky_labels) # if there are set labels, select them by default

        # Recast the template with the width's and height's
        self._template = '''<svg id="svgparent" width="''' + str(self.w) + '''" height="''' + str(self.h) + '''" tabindex="0" ''' + \
                         '''     onkeypress="${script('keyPress')}" onkeydown="${script('keyDown')}" onkeyup="${script('keyUp')}"> ''' + \
                         '''    <svg id="mod" width="''' + str(self.w) + '''" height="''' + str(self.h) + '''"> ${mod_inner} </svg> ''' + \
                         '''    <rect id="drag" x="-10" y="-10" width="5" height="5" stroke="#000000" stroke-width="2" fill="none" /> ''' + \
                         '''    <line   id="layoutline"      x1="-10" y1="-10" x2="-10"    y2="-10"    stroke="#000000" stroke-width="2" /> ''' + \
                         '''    <rect   id="layoutrect"      x="-10"  y="-10"  width="10"  height="10" stroke="#000000" stroke-width="2" /> ''' + \
                         '''    <circle id="layoutcircle"    cx="-10" cy="-10" r="5"       fill="none" stroke="#000000" stroke-width="6" /> ''' + \
                         '''    <circle id="layoutsunflower" cx="-10" cy="-10" r="5"                   stroke="#000000" stroke-width="2" /> ''' + \
                         '''    <rect id="screen" x="0" y="0" width="''' + str(self.w) + '''" height="''' + str(self.h) + '''" opacity="0.05" ''' + \
                         '''          onmousedown="${script('downSelect')}"          onmousemove="${script('moveEverything')}" ''' + \
                         '''          onmouseup="${script('upEverything')}"          onmousewheel="${script('mouseWheel')}" /> ''' + \
                         '''    <text id="opstr"   x="''' + str(self.w-2) + '''" y="12"  fill="#000000" font-size="10px" text-anchor="end"> ${op_str} </text> ''' + \
                         '''    <text id="infostr" x="5"   y="''' + str(self.h-4) + '''" fill="#000000" font-size="10px"> ${info_str} </text> ''' + \
                         '''    <path id="allentitieslayer" d="${allentitiespath}" fill="#000000" fill-opacity="0.01" stroke="none" ''' + \
                         '''          onmousedown="${script('downAllEntities')}" onmousemove="${script('moveEverything')}"  ''' + \
                         '''          onmouseup="${script('upEverything')}"      onmousewheel="${script('mouseWheel')}" /> ''' + \
                         '''    <path id="selectionlayer" d="${selectionpath}" fill="#ff0000" transform="" stroke="none" ''' + \
                         '''          onmousedown="${script('downMove')}"        onmousemove="${script('moveEverything')}" ''' + \
                         '''          onmouseup="${script('upEverything')}"      onmousewheel="${script('mouseWheel')}" /> ''' + \
                         '''</svg>'''

        # - Create a lock for threading
        self.lock = threading.Lock()

        # Execute the super initialization
        super().__init__(**kwargs)

        # Watch for callbacks
        self.param.watch(self.applyDragOp,      'drag_op_finished')
        self.param.watch(self.applyMoveOp,      'move_op_finished')
        self.param.watch(self.applyWheelOp,     'wheel_op_finished')
        self.param.watch(self.applyMiddleOp,    'middle_op_finished')
        self.param.watch(self.applyKeyOp,       'key_op_finished')
        self.param.watch(self.applyLayoutOp,    'layout_shape')
        self.param.watch(self.unselectedMoveOp, 'unselected_move_op_finished')

        # For companion visualizations
        self.companions = []


    # register companion visualizations
    def register_companion_viz(self, viz):
        self.companions.append(viz)
    
    # unregister companion visualizations
    def unregister_companion_viz(self, viz):
        if viz in self.companions:
            self.companions.remove(viz)

    #
    # saveLayout() - save the current layout
    #
    def saveLayout(self, filename):
        _lu_ = {'node':[], 'x':[], 'y':[]}
        for _node_ in self.pos:
            _lu_['node'].append(_node_)
            _lu_['x'].append(self.pos[_node_][0])
            _lu_['y'].append(self.pos[_node_][1])
        pd.DataFrame(_lu_).to_parquet(filename)

    #
    # loadLayout() - load a layout
    #
    def loadLayout(self, filename):
        if filename.lower().endswith('.csv'): _df_ = pd.read_csv(filename)
        else:                                 _df_ = pd.read_parquet(filename)
        for row_i, row in _df_.iterrows(): self.pos[row['node']] = (float(row['x']), float(row['y']))
        self.__refreshView__(info=False)

    #
    # selectEntities() - set the selected entities
    #
    def selectEntities(self, 
                       selection,                # string or set
                       set_op       = 'replace', # "replace", "add", "subtract", "intersect"
                       method       = 'exact',   # "exact", "substring", "regex"
                       ignore_case  = True):     # ignore the case when performing the match
        # Get all nodes in the current graph // these are the non-labeled variants
        all_nodes = set(self.graphs[self.df_level].nodes())

        # Perform either substring or regex matching if selected
        if   method == 'substring': # SUBSTRING MATCHES
            if type(selection) == str: _substrings_ = set([selection])
            else:                      _substrings_ = set(selection)
            _set_ = set()
            for _substring_ in _substrings_:
                if ignore_case: _substring_ = _substring_.lower()
                if 'node_labels' in self.ln_params:
                    for _node_ in self.ln_params['node_labels'].keys():
                        if _node_ in all_nodes: # only match nodes in the graph
                            if   ignore_case:
                                if _substring_ in str(self.ln_params['node_labels'][_node_]).lower(): _set_.add(_node_)
                            elif _substring_ in str(self.ln_params['node_labels'][_node_]): _set_.add(_node_)
                for _node_ in all_nodes:
                    if   ignore_case:
                        if _substring_ in str(_node_).lower(): _set_.add(_node_)
                    elif _substring_ in str(_node_): _set_.add(_node_)
        elif method == 'regex':     # REGEX MATCHES
            _set_ = set() # Not Implemented Yet
        else:                       # EXACT MATCHES
            # Fix up the selection so that it's definitely a set...
            if    selection is None:                                 selection_as_set = set()
            elif  type(selection) == list or type(selection) == set: selection_as_set = set(selection)
            elif  type(selection) == dict:                           selection_as_set = set(selection.keys())
            else:                                                    selection_as_set = set([selection])

            # Fix the case...
            if ignore_case: selection_as_set = {x.lower() for x in selection_as_set}

            # Iterate through the nodes...
            if 'node_labels' in self.ln_params: # node labels handled a little differently
                _set_ = set()
                for _node_ in self.ln_params['node_labels'].keys():
                    _label_ = self.ln_params['node_labels'][_node_]

                    if ignore_case: _label_, _node_cased_ = _label_.lower(), _node_.lower()
                    else:           _label_, _node_cased_ = _label_, _node_

                    if _node_ in all_nodes and (_node_cased_ in selection_as_set or _label_ in selection_as_set): _set_.add(_node_)
                for _node_ in all_nodes:
                    _node_cased_ = str(_node_).lower() if ignore_case else _node_
                    if _node_cased_ in selection_as_set: _set_.add(_node_)
                self.selected_entities = _set_
            else: # just use the selection
                if ignore_case:
                    _set_ = set()
                    for _node_ in all_nodes:
                        _node_cased_ = str(_node_).lower()
                        if _node_cased_ in selection_as_set: _set_.add(_node_)
                else:
                    _set_ = selection_as_set & all_nodes

        if   set_op == 'replace':   self.selected_entities  = _set_
        elif set_op == 'add':       self.selected_entities |= _set_
        elif set_op == 'subtract':  self.selected_entities -= _set_
        elif set_op == 'intersect': self.selected_entities &= _set_

        self.__refreshView__(comp=False)


    #
    # selectedEntities() - return the selected entities
    #
    def selectedEntities(self):
        _set_ = set()
        if 'node_labels' in self.ln_params:
            for _node_ in self.selected_entities:
                if _node_ in self.ln_params['node_labels']: _set_.add(self.ln_params['node_labels'][_node_])
                else:                                       _set_.add(_node_)
        else:
            _set_ = self.selected_entities
        return _set_

    #
    # selectedNodes() - return the selected nodes
    # - distinction is that the node is the representation within the dataframe
    # - versus the entity may be the lookup label if the node_labels is set
    # - if there are no node_labels, this should return the same as selectedEntities()
    #
    def selectedNodes(self):
        if 'node_labels' in self.ln_params:
            _set_, covered = set(), set()
            for _node_ in self.ln_params['node_labels'].keys():
                if _node_                                in self.selectedEntities() or \
                   self.ln_params['node_labels'][_node_] in self.selectedEntities(): _set_.add(_node_)
                covered.add(_node_), covered.add(self.ln_params['node_labels'][_node_])
            for _node_ in self.selectedEntities():
                if _node_ not in covered: _set_.add(_node_)
            return _set_
        else:
            return set(self.selected_entities) # no node labels, it's the same... return a copy

    #
    # __renderView__() - render the view
    #
    def __renderView__(self, __df__):
        # _ln_ = self.rt_self.linkNode(__df__, w=self.w, h=self.h, **self.ln_params)
        _ln_ = self.rt_self.link(__df__, w=self.w, h=self.h, **self.ln_params)
        return _ln_

    #
    # applyLayoutOp() - apply layout operation to the selected entities.
    #
    def applyLayoutOp(self, event):
        #self.lock.acquire()
        try:
            x0, y0, x1, y1 = self.drag_x0, self.drag_y0, self.drag_x1, self.drag_y1
            as_list     = list(self.selected_entities)
            nodes_moved = False
            _ln_        = self.dfs_layout[self.df_level]
            if len(as_list) > 1:
                if   self.layout_shape == "grid":
                    pos_adj = self.rt_self.rectangularArrangement(self.graphs[self.df_level], as_list, bounds=(x0,y0,x1,y1))
                    for _node_ in pos_adj:
                        _ln_.pos[_node_] = (float(_ln_.xT_inv(pos_adj[_node_][0])),float(_ln_.yT_inv(pos_adj[_node_][1])))
                    nodes_moved = True
                elif self.layout_shape == "circle":
                    wx0, wy0 = _ln_.xT_inv(x0), _ln_.yT_inv(y0)
                    wx1, wy1 = _ln_.xT_inv(x1), _ln_.yT_inv(y1)
                    r = sqrt((wx0 - wx1)**2 + (wy0 - wy1)**2)
                    if r < 0.001: r = 0.001
                    pos_adj = self.rt_self.circularOptimizedArrangement(self.graphs[self.df_level], as_list, _ln_.pos, xy=(wx0,wy0), r=r)
                    for _node_ in pos_adj: _ln_.pos[_node_] = (pos_adj[_node_][0],pos_adj[_node_][1])
                    nodes_moved = True
                elif self.layout_shape == "sunflower":
                    r = sqrt((x0 - x1)**2 + (y0 - y1)**2)
                    pos_adj = self.rt_self.sunflowerSeedArrangement(self.graphs[self.df_level], as_list, xy=(x0,y0), r_max=r)
                    for _node_ in pos_adj:
                        _ln_.pos[_node_] = (float(_ln_.xT_inv(pos_adj[_node_][0])),float(_ln_.yT_inv(pos_adj[_node_][1])))
                    nodes_moved = True
                elif self.layout_shape == "line" or self.layout_shape == "v-line" or self.layout_shape == "h-line":
                    if   self.layout_shape == "v-line": x0, x1, dx = x1, x1, 0
                    elif self.layout_shape == "h-line": y0, y1, dy = y1, y1, 0
                    wx0, wy0 = _ln_.xT_inv(x0), _ln_.yT_inv(y0)
                    wx1, wy1 = _ln_.xT_inv(x1), _ln_.yT_inv(y1)
                    pos_adj = self.rt_self.linearOptimizedArrangement(self.graphs[self.df_level], as_list, _ln_.pos, ((wx0,wy0),(wx1,wy1)))
                    for _node_ in pos_adj: _ln_.pos[_node_] = (pos_adj[_node_][0],pos_adj[_node_][1])
                    nodes_moved = True
            elif len(as_list) == 1:
                _ln_.pos[as_list[0]] = (float(_ln_.xT_inv((x0+x1)/2)), float(_ln_.yT_inv((y0+y1)/2)))
                nodes_moved = True

            # Reposition if the nodes moved
            if nodes_moved:
                for i in range(len(self.dfs_layout)): self.dfs_layout[i].invalidateRender()
                self.__refreshView__(info=False)
        finally:
            self.layout_shape = ""
            #self.lock.release()

    #
    # Middle button state & method
    #
    x0_middle          = param.Integer(default=0)
    y0_middle          = param.Integer(default=0)
    x1_middle          = param.Integer(default=0)
    y1_middle          = param.Integer(default=0)
    middle_op_finished = param.Boolean(default=False)

    #
    # applyMiddleOp() - apply middle operation -- either pan view or reset view
    #
    async def applyMiddleOp(self,event):
        self.lock.acquire()
        try:
            if self.middle_op_finished:
                x0, y0, x1, y1 = self.x0_middle, self.y0_middle, self.x1_middle, self.y1_middle
                dx, dy         = x1 - x0, y1 - y0
                _comp_ , _adj_coordinate_ = self.dfs_layout[self.df_level], (x0,y0)
                if _comp_ is not None:
                    if (abs(self.x0_middle - self.x1_middle) <= 1) and (abs(self.y0_middle - self.y1_middle) <= 1):
                        if _comp_.applyMiddleClick(_adj_coordinate_):
                            self.__refreshView__(info=False)
                            for i in range(len(self.dfs_layout)):
                                if i != self.df_level:
                                    self.dfs_layout[i].invalidateRender()
                                    self.dfs_layout[i].applyViewConfiguration(self.dfs_layout[self.df_level])
                    else:
                        if _comp_.applyMiddleDrag(_adj_coordinate_, (dx,dy)):
                            self.__refreshView__(info=False)
                            for i in range(len(self.dfs_layout)): 
                                if i != self.df_level:
                                    self.dfs_layout[i].invalidateRender()
                                    self.dfs_layout[i].applyViewConfiguration(self.dfs_layout[self.df_level])
        finally:
            self.middle_op_finished = False
            self.lock.release()

    #
    # Wheel operation state & method
    #
    wheel_x           = param.Integer(default=0)
    wheel_y           = param.Integer(default=0)
    wheel_rots        = param.Integer(default=0) # Mult by 10 and rounded...
    wheel_op_finished = param.Boolean(default=False)

    #
    # applyWheelOp() - apply mouse wheel operation (zoom in & out)
    #
    async def applyWheelOp(self,event):
        self.lock.acquire()
        try:
            if self.wheel_op_finished:
                x, y, rots = self.wheel_x, self.wheel_y, self.wheel_rots
                if rots != 0:
                    # Find the compnent where the scroll event occurred
                    _comp_ , _adj_coordinate_ = self.dfs_layout[self.df_level], (x,y)
                    if _comp_ is not None:
                        if _comp_.applyScrollEvent(rots, _adj_coordinate_):
                            # Re-render current
                            self.__refreshView__(info=False)
                            # Propagate the view configuration to the same component across the dataframe stack
                            for i in range(len(self.dfs_layout)):
                                if i != self.df_level:
                                    self.dfs_layout[i].applyViewConfiguration(_comp_)
        finally:
            self.wheel_op_finished = False
            self.wheel_rots        = 0            
            self.lock.release()


    #
    # __refreshView__() - refresh the view
    #
    def __refreshView__(self, comp=True, info=True, all_ents=True, sel_ents=True):
        if (comp):     self.mod_inner        = self.dfs_layout[self.df_level].renderSVG()
        if (info):     self.info_str         = f'{len(self.selected_entities)} Selected | {self.label_mode} | {self.layout_mode}'
        if (all_ents): self.allentitiespath  = self.dfs_layout[self.df_level].__createPathDescriptionForAllEntities__()
        if (sel_ents): self.selectionpath    = self.dfs_layout[self.df_level].__createPathDescriptionOfSelectedEntities__(my_selection=self.selected_entities)

    #
    # popStack() - as long as there are items on the stack, go up the stack
    #
    def popStack(self, callers=None):
        if self.df_level == 0: return
        if callers is not None and self in callers: return
        if callers is None: callers = set([self])
        else:               callers.add(self)

        self.df_level -= 1

        self.__refreshView__()
        for c in self.companions:
            if isinstance(c, RTReactiveHTML) or isinstance(c, RTGraphInteractiveLayout): c.popStack(callers=callers)

    #
    # setStackPosition() - set to a specific position
    #
    def setStackPostion(self, i_found, callers=None):
        if i_found < 0 or i_found >= len(self.dfs_layout): return
        if callers is not None and self in callers: return
        if callers is None: callers = set([self])
        else:               callers.add(self)

        if i_found < 0 or i_found >= len(self.dfs_layout): return

        self.df_level = i_found

        self.__refreshView__()
        for c in self.companions:
            if isinstance(c, RTReactiveHTML) or isinstance(c, RTGraphInteractiveLayout): c.setStackPosition(i_found, callers=callers)

    #
    # pushStack() - push a dataframe onto the stack
    #
    def pushStack(self, df, g=None, callers=None):
        if callers is not None and self in callers: return
        if callers is None: callers = set([self])
        else:               callers.add(self)

        if g is None: g = self.rt_self.createNetworkXGraph(df, self.ln_params['relationships'])

        _ln_ = self.__renderView__(df)
        _ln_.applyViewConfiguration(self.dfs_layout[self.df_level])
        if len(self.dfs_layout) > (self.df_level+1):
            new_dfs, new_dfs_layout, new_graphs = [], [], []
            for i in range(self.df_level+1):
                new_dfs.append(self.dfs[i]), new_dfs_layout.append(self.dfs_layout[i]), new_graphs.append(self.graphs[i])
            self.dfs, self.dfs_layout, self.graphs = new_dfs, new_dfs_layout, new_graphs
        self.dfs        .append(df)
        self.dfs_layout .append(_ln_)
        self.graphs     .append(g)
        self.df_level += 1
        self.selected_entities = set()

        self.__refreshView__()

        for c in self.companions:
            if isinstance(c, RTReactiveHTML) or isinstance(c, RTGraphInteractiveLayout): c.pushStack(df, callers=callers)

    #
    # applyKeyOp() - apply specified key operation
    #
    async def applyKeyOp(self,event):
        self.lock.acquire()
        try:
            _ln_ = self.dfs_layout[self.df_level]
            #
            # "E" - Expand / Expand w/ Directed
            #
            if self.key_op_finished == 'e' or self.key_op_finished == 'E':
                if self.key_op_finished == 'E':
                    _digraph_ = self.rt_self.createNetworkXGraph(self.dfs[self.df_level], self.ln_params['relationships'], use_digraph=True)
                    _new_set_ = set(self.selected_entities)
                    for _node_ in self.selected_entities:
                        for _nbor_ in _digraph_.neighbors(_node_):
                            _new_set_.add(_nbor_)
                    self.selected_entities = _new_set_
                else:
                    _new_set_ = set(self.selected_entities)
                    for _node_ in self.selected_entities:
                        for _nbor_ in self.graphs[self.df_level].neighbors(_node_):
                            _new_set_.add(_nbor_)
                    self.selected_entities = _new_set_

                self.__refreshView__(comp=False, all_ents=False)

            #
            # "Q" - Invert Selection / Common Neighbors
            #            
            elif self.key_op_finished == 'q' or self.key_op_finished == 'Q':
                if   self.key_op_finished == 'Q': # common neighbors
                    inter_set = None
                    for _node_ in self.selected_entities:
                        nbor_set = set()
                        for _nbor_ in self.graphs[self.df_level].neighbors(_node_):
                            nbor_set.add(_nbor_)
                        if inter_set is None: inter_set = nbor_set             # first time, it gets the nbors
                        else:                 inter_set = inter_set & nbor_set # all other times it's and'ed
                    if inter_set is not None: self.selected_entities = inter_set
                else:                   # invert selection
                    _new_set_ = set()
                    for _node_ in self.graphs[self.df_level]:
                        if _node_ not in self.selected_entities:
                            _new_set_.add(_node_)
                    self.selected_entities = _new_set_

                self.__refreshView__(comp=False, all_ents=False)

            #
            # "S" - Set Sticky Labels & Remove Sticky Labels
            #
            elif self.key_op_finished == 's' or self.key_op_finished == 'S':
                if self.key_op_finished == 'S': self.sticky_labels = self.sticky_labels - self.selected_entities
                else:                           self.sticky_labels = set(self.selected_entities) # make a new set object
                if self.label_mode == 'sticky labels': _ln_.labelOnly(self.sticky_labels)
                self.ln_params['label_only']  = self.sticky_labels

                self.__refreshView__(info=False, all_ents=False, sel_ents=False)

            #
            # "T" - Collapse
            #
            elif len(self.selected_entities) > 0 and self.key_op_finished == 't':
                for _entity_ in self.selected_entities:
                    xy = _ln_.pos[_entity_]
                    _ln_.pos[_entity_] = (_ln_.xT_inv(self.x_mouse), _ln_.yT_inv(self.y_mouse))
                for i in range(len(self.dfs_layout)): self.dfs_layout[i].invalidateRender()

                self.__refreshView__(info=False)

            #
            # "Y" - Organize Selected into a Vertical or Horizontal Line
            #
            elif self.key_op_finished == 'y' or self.key_op_finished == 'Y':
                if self.key_op_finished == 'Y':
                    for _entity_ in self.selected_entities:
                        xy = _ln_.pos[_entity_]
                        _ln_.pos[_entity_] = (xy[0], _ln_.yT_inv(self.y_mouse))
                else:
                    for _entity_ in self.selected_entities:
                        xy = _ln_.pos[_entity_]
                        _ln_.pos[_entity_] = (_ln_.xT_inv(self.x_mouse), xy[1])
                for i in range(len(self.dfs_layout)): self.dfs_layout[i].invalidateRender()

                self.__refreshView__(info=False)

            #
            # "W" - Remove Sticky Labels & Label Toggles
            #
            elif self.key_op_finished == 'w' or self.key_op_finished == 'W':
                if self.key_op_finished == 'W':
                    if   self.label_mode == 'all labels':    
                        self.label_mode = 'sticky labels'
                        _ln_.labelOnly(self.sticky_labels)
                        _ln_.drawLabels(True)
                        self.ln_params['draw_labels'] = True
                    elif self.label_mode == 'sticky labels':
                        self.label_mode = 'no labels'
                        _ln_.drawLabels(False)
                        self.ln_params['draw_labels'] = False                        
                    else:                                    
                        self.label_mode = 'all labels'
                        _ln_.drawLabels(True)
                        self.ln_params['draw_labels'] = True
                        _ln_.labelOnly(set())
                else:
                    self.sticky_labels = self.sticky_labels | self.selected_entities
                    if self.label_mode == 'sticky labels': _ln_.labelOnly(self.sticky_labels)
                    self.ln_params['label_only'] = self.sticky_labels

                self.__refreshView__(all_ents=False, sel_ents=False)

            #
            # 'C' - Center on Selected (if selected) or Reset View (if not selected) / Selected + Neighbors
            #
            elif self.key_op_finished == 'c' or self.key_op_finished == 'C':
                _rerender_ = False
                if self.key_op_finished == 'C':
                    if len(self.selected_entities) > 0:
                        _new_set_ = set(self.selected_entities)
                        for _node_ in self.selected_entities:
                            for _nbor_ in self.graphs[self.df_level].neighbors(_node_):
                                _new_set_.add(_nbor_)
                                _view_ = _ln_.__calculateGeometry__(for_entities=_new_set_)
                                _ln_.setViewWindow(_view_)
                                _rerender_ = True
                else:
                    if len(self.selected_entities) > 0: # Zoom to selected entities
                        _view_ = _ln_.__calculateGeometry__(for_entities=self.selected_entities)
                        _ln_.setViewWindow(_view_)
                        _rerender_ = True
                    else:                               # Recenter complete view
                        _view_ = _ln_.__calculateGeometry__()
                        _ln_.setViewWindow(_view_)
                        _rerender_ = True
                
                if _rerender_:
                    for i in range(len(self.dfs_layout)): self.dfs_layout[i].invalidateRender()

                    self.__refreshView__(info=False)

                    for i in range(len(self.dfs_layout)):
                        if i != self.df_level: self.dfs_layout[i].applyViewConfiguration(_ln_)
            #
            # 'p' - filter nodes in / out of view
            #
            elif self.key_op_finished == 'p' or self.key_op_finished == 'P':
                if self.key_op_finished == 'P' and self.df_level > 0: # pop the stack
                    self.popStack()

                elif len(self.selected_entities) > 0: # push the stack
                    _g_ = copy.deepcopy(self.graphs[self.df_level])
                    for _entity_ in self.selected_entities: _g_.remove_node(_entity_)
                    _df_ = self.rt_self.filterDataFrameByGraph(self.dfs[self.df_level], self.ln_params['relationships'], _g_)
                    if len(_df_) > 0: self.pushStack(_df_, _g_)

            #
            # Degree Related Operations
            #
            elif len(self.key_op_finished) == 1 and self.key_op_finished in '0123456789':
                _match_ = set()
                c       = self.key_op_finished
                min_degree = 10 if c == '7' else 20 if c == '8' else 50  if c == '9' else 100 if c == '0' else None
                max_degree = 20 if c == '7' else 50 if c == '8' else 100 if c == '9' else 1e9 if c == '0' else None

                if min_degree is not None:
                    for _node_ in self.graphs[self.df_level]:
                        if self.graphs[self.df_level].degree(_node_) >= min_degree and self.graphs[self.df_level].degree(_node_) < max_degree: _match_.add(_node_)
                else:
                    _degree_ = int(self.key_op_finished)
                    for _node_ in self.graphs[self.df_level]:
                        if self.graphs[self.df_level].degree(_node_) == _degree_: _match_.add(_node_)

                if   self.shiftkey:               self.selected_entities = self.selected_entities - _match_
                elif len(self.selected_entities): self.selected_entities = self.selected_entities & _match_
                else:                             self.selected_entities = _match_

                self.__refreshView__(comp=False, all_ents=False)
            
            #
            # Next Selection Op Settings...
            # ... this replaces the shift and ctrl and ctrl-shift operators -- those aren't reliabel across IDE's...
            #
            elif self.key_op_finished == 'n':
                if   self.op_str == 'Select':   self.op_str = 'Add'
                elif self.op_str == 'Add':      self.op_str = 'Subtract'
                elif self.op_str == 'Subtract': self.op_str = 'Intersect'
                else:                           self.op_str = 'Select'
            
            #
            # Next Layout Option
            #
            elif self.key_op_finished == 'G':
                if   self.layout_mode == 'grid':      self.layout_mode = 'circle'
                elif self.layout_mode == 'circle':    self.layout_mode = 'sunflower'
                elif self.layout_mode == 'sunflower': self.layout_mode = 'line'
                elif self.layout_mode == 'line':      self.layout_mode = 'h-line'
                elif self.layout_mode == 'h-line':    self.layout_mode = 'v-line'
                else:                                 self.layout_mode = 'grid'

                self.__refreshView__(comp=False, all_ents=False, sel_ents=False)

        finally:
            self.key_op_finished = ''
            self.lock.release()

    #
    # Drag operation state
    #
    drag_op_finished  = param.Boolean(default=False)
    drag_x0           = param.Integer(default=0)
    drag_y0           = param.Integer(default=0)
    drag_x1           = param.Integer(default=10)
    drag_y1           = param.Integer(default=10)

    #
    # Unselected move operation state
    #
    allentities_x0              = param.Integer(default=10)
    allentities_y0              = param.Integer(default=10)
    unselected_move_op_finished = param.Boolean(default=False)

    #
    # Move operation state
    #
    move_op_finished = param.Boolean(default=False)

    #
    # Shape operation state
    #
    layout_shape     = param.String(default="")

    # Key States
    shiftkey         = param.Boolean(default=False)
    ctrlkey          = param.Boolean(default=False)
    last_key         = param.String(default='')
    key_op_finished  = param.String(default='')

    # Mouse States
    x_mouse          = param.Integer(default=0)
    y_mouse          = param.Integer(default=0)

    #
    # applyDragOp() - select the nodes within the drag operations bounding box.
    #
    async def applyDragOp(self,event):
        self.lock.acquire()
        try:
            if self.drag_op_finished:
                _x0,_y0,_x1,_y1 = min(self.drag_x0, self.drag_x1), min(self.drag_y0, self.drag_y1), max(self.drag_x1, self.drag_x0), max(self.drag_y1, self.drag_y0)
                if _x0 == _x1: _x1 += 1
                if _y0 == _y1: _y1 += 1
                _rect_ = Polygon([(_x0,_y0), (_x0,_y1), (_x1,_y1), (_x1,_y0)])
                _overlapping_entities_  = set(self.dfs_layout[self.df_level].overlappingEntities(_rect_))
                if _overlapping_entities_ is None: _overlapping_entities_ = set()

                if   self.op_str == 'Intersect': self.selected_entities = set(self.selected_entities) & set(_overlapping_entities_)
                elif self.op_str == 'Subtract':  self.selected_entities = set(self.selected_entities) - set(_overlapping_entities_)
                elif self.op_str == 'Add':       self.selected_entities = set(self.selected_entities) | set(_overlapping_entities_)
                else:                            self.selected_entities = _overlapping_entities_
                
                self.__refreshView__(comp=False, all_ents=False)

        finally:
            self.drag_op_finished = False
            self.lock.release()

    #
    # applyMoveOp() - apply a move operation to the selected node(s)
    # - may also be used to de-select a selected node when the op string is "Subtract" and no drag occurs
    #
    async def applyMoveOp(self,event):
        self.lock.acquire()
        try:
            if self.move_op_finished:
                if self.drag_x0 == self.drag_x1 and self.drag_y0 == self.drag_y1 and self.op_str == 'Subtract':
                    _point_entities_  = self.dfs_layout[self.df_level].entitiesAtPoint((self.drag_x0,self.drag_y0))
                    self.__refreshView__(comp=False, all_ents=False)
                else:
                    self.dfs_layout[self.df_level].__moveSelectedEntities__((self.drag_x1 - self.drag_x0, self.drag_y1 - self.drag_y0), my_selection=self.selected_entities)
                    self.__refreshView__()
                    for i in range(len(self.dfs_layout)):
                        if i != self.df_level:  self.dfs_layout[i].invalidateRender()
        finally:
            self.move_op_finished = False
            self.lock.release()

    #
    # unselectedMoveOp() - occurs when user clicks directly on an unselected node.
    #
    async def unselectedMoveOp(self, event):
        self.lock.acquire()
        try:
            if self.unselected_move_op_finished:
                _x_,_y_ = self.allentities_x0, self.allentities_y0
                _overlapping_entities_  = self.dfs_layout[self.df_level].entitiesAtPoint((_x_,_y_))
                if _overlapping_entities_ is None: _overlapping_entities_ = set()

                if   self.op_str == 'Add':      self.selected_entities = (set(self.selected_entities) | set(_overlapping_entities_))
                elif self.op_str == 'Subtract': self.selected_entities = (set(self.selected_entities) - set(_overlapping_entities_))
                else:                           self.selected_entities = set(_overlapping_entities_)

                if self.drag_x0 == self.drag_x1 and self.drag_y0 == self.drag_y1:
                    pass # just do the selection operation
                else: # and do a move operation
                    self.dfs_layout[self.df_level].__moveSelectedEntities__((self.drag_x1 - self.drag_x0, self.drag_y1 - self.drag_y0), my_selection=self.selected_entities)
                    for i in range(len(self.dfs_layout)): self.dfs_layout[i].invalidateRender()
                    self.mod_inner       = self.dfs_layout[self.df_level]._repr_svg_()
                    self.allentitiespath = self.dfs_layout[self.df_level].__createPathDescriptionForAllEntities__()

                self.__refreshView__(comp=False, all_ents=False)

        finally:
            self.unselected_move_op_finished = False
            self.lock.release()

    #
    # Panel Javascript Definitions
    #
    _scripts = {
        'render':"""
            mod.innerHTML            = data.mod_inner;
            opstr.innerHTML          = data.op_str;
            infostr.innerHTML        = data.info_str;
            state.x0_drag            = state.y0_drag = -10;
            state.x1_drag            = state.y1_drag =  -5;
            data.shiftkey            = false;
            data.ctrlkey             = false;
            state.drag_op            = false;
            state.move_op            = false;
            state.unselected_move_op = false;
            state.layout_op          = false; // true if next mouse button 1 press is the begin of a layout
            state.layout_op_shape    = "";    // trigger field for python to peform the layout operation
            data.middle_op_finished  = false;
            data.move_op_finished    = false;
            svgparent.focus(); // else it loses focus on every render...
        """,
        'keyPress':"""
            svgparent.focus(); // else it loses focus on every render...
        """,
        'keyDown':"""
            data.ctrlkey  = event.ctrlKey;
            data.shiftkey = event.shiftKey;

            if      (event.key == "c") { data.key_op_finished = 'c';  } // (if selected) zoom to selected, else zoom to entire view
            else if (event.key == "C") { data.key_op_finished = 'C';  } // Zoom to selected + neighbors
            else if (event.key == "e") { data.key_op_finished = 'e';  } // Expand
            else if (event.key == "E") { data.key_op_finished = 'E';  } // Expand (w/ digraph)
            else if (event.key == "g") { state.layout_op      = true; } // Mouse press is layout shape
            else if (event.key == "G") { data.key_op_finished = 'G';  } // Iterate through layout shapes
            else if (event.key == "n") { data.key_op_finished = 'n';  } // Iterate through selection methdology
            else if (event.key == "q") { data.key_op_finished = 'q';  } // Invert selection
            else if (event.key == "Q") { data.key_op_finished = 'Q';  } // Select common neighbors to selected nodes
            else if (event.key == "s") { data.key_op_finished = 's';  } // Set sticky labels
            else if (event.key == "S") { data.key_op_finished = 'S';  } // Subtract selected from sticky labels
            else if (event.key == "t") { data.key_op_finished = 't';  } // Collapse selected to a single point
            else if (event.key == "w") { data.key_op_finished = 'w';  } // Add to sticky labels (it's right above 's')
            else if (event.key == "W") { data.key_op_finished = 'W';  } // Iterate through label settings
            else if (event.key == "y") { data.key_op_finished = 'y';  } // Arrange selected into a verticle line at mouse
            else if (event.key == "Y") { data.key_op_finished = 'Y';  } // Arrange selected into a horizontal line at mouse
            else if (event.key == "p") { data.key_op_finished = 'p';  } // push the stack (remove the selected from the current graph)
            else if (event.key == "P") { data.key_op_finished = 'P';  } // pop the stack (add removed nodes back in)
            else if (event.key == "1" || event.key == "!") { data.key_op_finished = '1';  }
            else if (event.key == "2" || event.key == "@") { data.key_op_finished = '2';  }
            else if (event.key == "3" || event.key == "#") { data.key_op_finished = '3';  }
            else if (event.key == "4" || event.key == "$") { data.key_op_finished = '4';  }
            else if (event.key == "5" || event.key == "%") { data.key_op_finished = '5';  }
            else if (event.key == "6" || event.key == "^") { data.key_op_finished = '6';  }
            else if (event.key == "7" || event.key == "&") { data.key_op_finished = '7';  }
            else if (event.key == "8" || event.key == "*") { data.key_op_finished = '8';  }
            else if (event.key == "9" || event.key == "(") { data.key_op_finished = '9';  }
            else if (event.key == "0" || event.key == ")") { data.key_op_finished = '0';  }

            data.last_key = event.key;
            svgparent.focus(); // else it loses focus on every render...
        """,
        'keyUp':"""
            data.ctrlkey  = event.ctrlKey;
            data.shiftkey = event.shiftKey;
            if (event.key == "g") { state.layout_op = false; }
            svgparent.focus(); // else it loses focus on every render...
        """,
        'moveEverything':"""
            data.ctrlkey   = event.ctrlKey;
            data.shiftkey  = event.shiftKey;
            data.x_mouse   = event.offsetX; 
            data.y_mouse   = event.offsetY;
            state.x1_drag  = event.offsetX; 
            state.y1_drag  = event.offsetY; 
            if (state.drag_op)               { self.myUpdateDragRect(); }
            if (state.move_op)               { selectionlayer.setAttribute("transform", "translate(" + (state.x1_drag - state.x0_drag) + "," + (state.y1_drag - state.y0_drag) + ")"); }
            if (state.unselected_move_op)    { }
            if (state.layout_op_shape != "") { self.myUpdateLayoutOp(); }
        """,
        'downAllEntities':"""
            data.ctrlkey  = event.ctrlKey;
            data.shiftkey = event.shiftKey;
            if (event.button == 0) {
                    data.allentities_x0      = event.offsetX; 
                    data.allentities_y0      = event.offsetY; 
                    state.x0_drag            = event.offsetX;                
                    state.y0_drag            = event.offsetY;                
                    state.x1_drag            = event.offsetX;                
                    state.y1_drag            = event.offsetY;
                    state.unselected_move_op = true;
            }
        """,
        'downSelect':"""
            if (event.button == 0) {
                state.x0_drag  = event.offsetX;
                state.y0_drag  = event.offsetY;
                state.x1_drag  = event.offsetX;
                state.y1_drag  = event.offsetY;
                if (state.layout_op) { state.layout_op_shape = data.layout_mode; self.myUpdateLayoutOp();
                } else               { state.drag_op         = true;             self.myUpdateDragRect(); }
            } else if (event.button == 1) {
                data.x0_middle = data.x1_middle = event.offsetX;
                data.y0_middle = data.y1_middle = event.offsetY;
            }
        """,
        'downMove':"""
            if (event.button == 0) {
                state.x0_drag  = state.x1_drag  = event.offsetX;
                state.y0_drag  = state.y1_drag  = event.offsetY;
                state.move_op  = true;
            } else if (event.button == 1) {
                data.x0_middle = data.x1_middle = event.offsetX; 
                data.y0_middle = data.y1_middle = event.offsetY;
            }
        """,
        'myUpdateLayoutOp':"""
            var dx = state.x1_drag - state.x0_drag,
                dy = state.y1_drag - state.y0_drag;
            var reset_circle = true, reset_sunflower = true, reset_rect = true, reset_line = true;
            if        (state.layout_op_shape == "circle")    { reset_circle = false;
                layoutcircle.setAttribute("cx", state.x0_drag);
                layoutcircle.setAttribute("cy", state.y0_drag);
                layoutcircle.setAttribute("r",  Math.sqrt(dx*dx + dy*dy));
            } else if (state.layout_op_shape == "sunflower") { reset_sunflower = false;
                layoutsunflower.setAttribute("cx", state.x0_drag);
                layoutsunflower.setAttribute("cy", state.y0_drag);
                layoutsunflower.setAttribute("r",  Math.sqrt(dx*dx + dy*dy));            
            } else if (state.layout_op_shape == "grid")      { reset_rect = false;
                layoutrect.setAttribute("x", Math.min(state.x0_drag, state.x1_drag));
                layoutrect.setAttribute("y", Math.min(state.y0_drag, state.y1_drag));
                layoutrect.setAttribute("width",  Math.abs(dx));
                layoutrect.setAttribute("height", Math.abs(dy));
            } else if (state.layout_op_shape == "line")    { reset_line = false;
                layoutline.setAttribute("x1", state.x0_drag);
                layoutline.setAttribute("y1", state.y0_drag);
                layoutline.setAttribute("x2", state.x1_drag);
                layoutline.setAttribute("y2", state.y1_drag);
            } else if (state.layout_op_shape == "h-line")  { reset_line = false;
                layoutline.setAttribute("x1", state.x0_drag);
                layoutline.setAttribute("y1", state.y1_drag);
                layoutline.setAttribute("x2", state.x1_drag);
                layoutline.setAttribute("y2", state.y1_drag);
            } else if (state.layout_op_shape == "v-line")  { reset_line = false;
                layoutline.setAttribute("x1", state.x1_drag);
                layoutline.setAttribute("y1", state.y0_drag);
                layoutline.setAttribute("x2", state.x1_drag);
                layoutline.setAttribute("y2", state.y1_drag);
            } else { state.layout_op_shape == ""; }
            if (reset_circle)    { layoutcircle   .setAttribute("cx", -10); layoutcircle   .setAttribute("cy", -10); layoutcircle   .setAttribute("r",      5); }
            if (reset_sunflower) { layoutsunflower.setAttribute("cx", -10); layoutsunflower.setAttribute("cy", -10); layoutsunflower.setAttribute("r",      5); }
            if (reset_rect)      { layoutrect     .setAttribute("x",  -10); layoutrect     .setAttribute("y",  -10); layoutrect     .setAttribute("width",  5);  layoutrect.setAttribute("height",  5); }
            if (reset_line)      { layoutline     .setAttribute("x1", -10); layoutline     .setAttribute("y1", -10); layoutline     .setAttribute("x2",    -5);  layoutline.setAttribute("y2",     -5); }
        """,
        'upEverything':"""
            if (event.button == 0) {
                state.x1_drag         = event.offsetX; 
                state.y1_drag         = event.offsetY;
                if (state.drag_op) {
                    state.shiftkey        = event.shiftKey;
                    state.drag_op         = false;
                    self.myUpdateDragRect();
                    data.drag_x0          = state.x0_drag; 
                    data.drag_y0          = state.y0_drag; 
                    data.drag_x1          = state.x1_drag; 
                    data.drag_y1          = state.y1_drag;
                    data.drag_op_finished = true;
                } else if (state.move_op) {
                    state.move_op         = false;
                    data.drag_x0          = state.x0_drag; 
                    data.drag_y0          = state.y0_drag; 
                    data.drag_x1          = state.x1_drag; 
                    data.drag_y1          = state.y1_drag;
                    data.move_op_finished = true;                    
                } else if (state.layout_op_shape != "") {
                    data.drag_x0          = state.x0_drag; 
                    data.drag_y0          = state.y0_drag; 
                    data.drag_x1          = state.x1_drag; 
                    data.drag_y1          = state.y1_drag;
                    data.layout_shape     = state.layout_op_shape;
                    state.layout_op_shape = "";
                    self.myUpdateLayoutOp();
                } else if (state.unselected_move_op) {
                    data.ctrlkey  = event.ctrlKey;
                    data.shiftkey = event.shiftKey;
                    data.drag_x0  = state.x0_drag;
                    data.drag_y0  = state.y0_drag;
                    data.drag_x1  = state.x1_drag;
                    data.drag_y1  = state.y1_drag;
                    data.unselected_move_op_finished = true;
                    state.unselected_move_op = false;
                }
            } else if (event.button == 1) {
                data.x1_middle          = event.offsetX; 
                data.y1_middle          = event.offsetY;
                data.middle_op_finished = true;                
            }
        """,
        'mouseWheel':"""
            event.preventDefault();
            data.wheel_x = event.offsetX; data.wheel_y = event.offsetY; data.wheel_rots  = Math.round(10*event.deltaY);
            data.wheel_op_finished = true;
        """,
        'mod_inner':"""
            mod.innerHTML     = data.mod_inner;
            infostr.innerHTML = data.info_str;
            opstr.innerHTML   = data.op_str;
            svgparent.focus(); // else it loses focus on every render...
        """,
        'selectionpath':"""
            selectionlayer.setAttribute("d", data.selectionpath);
            svgparent.focus(); // else it loses focus on every render...
        """,
        'info_str': """
            infostr.innerHTML = data.info_str;
            opstr.innerHTML   = data.op_str;
            svgparent.focus(); // else it loses focus on every render...
        """,
        'op_str': """
            infostr.innerHTML = data.info_str;
            opstr.innerHTML   = data.op_str;
            svgparent.focus(); // else it loses focus on every render...
        """,
        'myUpdateDragRect':"""
            if (state.drag_op) {
                x = Math.min(state.x0_drag, state.x1_drag); 
                y = Math.min(state.y0_drag, state.y1_drag);
                w = Math.abs(state.x1_drag - state.x0_drag)
                h = Math.abs(state.y1_drag - state.y0_drag)
                drag.setAttribute('x',x);     drag.setAttribute('y',y);
                drag.setAttribute('width',w); drag.setAttribute('height',h);
                if      (data.op_str == "Intersect")  drag.setAttribute('stroke','#0000ff');
                else if (data.op_str == "Subtract")   drag.setAttribute('stroke','#ff0000');
                else if (data.op_str == "Add")        drag.setAttribute('stroke','#00ff00');
                else                                  drag.setAttribute('stroke','#000000');
            } else {
                drag.setAttribute('x',-10);   drag.setAttribute('y',-10);
                drag.setAttribute('width',5); drag.setAttribute('height',5);
            }
        """
    }

