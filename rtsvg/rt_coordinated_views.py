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

import threading

import panel as pn
import param

from panel.reactive import ReactiveHTML

from .rt_stackable import RTStackable

__name__ = 'rt_coordinated_views'

#
# ReactiveHTML Class for Coordinated Views Panel Implementation
#
class RTCoordinatedViews(ReactiveHTML, RTStackable):
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
            if isinstance(c, RTStackable): c.popStack(callers=callers)

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
            if isinstance(c, RTStackable): c.setStackPostion(i_found, callers=callers)

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
            if isinstance(c, RTStackable): c.pushStack(df, callers=callers)

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
                    if self.drag_shiftkey:
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
                                if len(self.dfs[i]) == len(_df) and self.dfs[i].equals(_df):
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
