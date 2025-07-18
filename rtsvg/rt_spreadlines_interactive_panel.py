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

import polars as pl

import threading
import copy

import panel as pn
import param

from panel.reactive import ReactiveHTML

from shapely import Polygon

from .rt_stackable import RTStackable, RTSelectable

__name__ = 'rt_spreadlines_interactive_panel'

#
# ReactiveHTML Class for Panel Implementation
#
class RTSpreadLinesInteractivePanel(ReactiveHTML, RTStackable, RTSelectable):
      #
      # Inner Modification for RT SVG Render
      #
      mod_inner         = param.String(default="""<circle cx="300" cy="200" r="10" fill="red" />""")

      #
      # All Entities Path
      #
      allentitiespath   = param.String(default="M -100 -100 l 10 0 l 0 10 l -10 0 l 0 -10 Z")

      #
      # Selection Path
      #
      selectionpath     = param.String(default="M -100 -100 l 10 0 l 0 10 l -10 0 l 0 -10 Z")

      #
      # viewBox
      #
      viewBox           = param.String(default="0 0 600 200")

      #
      # viewBox parameters
      #
      vx                = param.Number(default=0.0)
      vy                = param.Number(default=0.0)
      vw                = param.Number(default=600.0)
      vh                = param.Number(default=200.0)

      #
      # Panel Template
      # - rewritten in constructor with width and height filled in
      #
      _template = """
<svg id="svgparent" width="600" height="200" viewBox="${viewBox}" tabindex="0" 
     onkeydown="${script('myOnKeyDown')}" onkeyup="${script('myOnKeyUp')}">
    <svg id="mod" width="10000000" height="10000000"> ${mod_inner} </svg>
    <rect id="drag" x="-10" y="-10" width="5" height="5" stroke="#000000" stroke-width="2" fill="none" />
    <rect id="screen" x="0" y="0" width="10000000" height="10000000" opacity="0.05"
          onmouseover="${script('myOnMouseOver')}"      onmouseout="${script('myOnMouseOut')}"
          onmousedown="${script('myOnMouseDown')}"      onmousemove="${script('myOnMouseMove')}"
          onmouseup="${script('myOnMouseUp')}"          onmousewheel="${script('myOnMouseWheel')}" />
    <path id="allentitieslayer" d="${allentitiespath}" fill="#000000" fill-opacity="0.01" stroke="none"
          onmouseover="${script('myOnMouseOver')}"      onmouseout="${script('myOnMouseOut')}"
          onmousedown="${script('downAllEntities')}" onmousemove="${script('myOnMouseMove')}" 
          onmouseup="${script('myOnMouseUp')}"      onmousewheel="${script('myOnMouseWheel')}" />
    <path id="selectionlayer" d="${selectionpath}" fill="#ff0000" transform="" stroke="none"
          onmouseover="${script('myOnMouseOver')}"      onmouseout="${script('myOnMouseOut')}"
          onmousedown="${script('downMove')}"        onmousemove="${script('myOnMouseMove')}"
          onmouseup="${script('myOnMouseUp')}"      onmousewheel="${script('myOnMouseWheel')}" />
</svg>
"""

      #
      # Constructor
      #
      def __init__(self,
                   rt_self,              # RACETrack instance
                   df,                   # data frame
                   sl_params,            # spreadline params
                   w            = 600,   # width
                   h            = 200,   # height
                   **kwargs):
            # Setup specific instance information
            # - Copy the member variables
            self.rt_self           = rt_self
            self.sl_params         = sl_params
            self.w                 = w
            self.h                 = h
            self.kwargs            = kwargs

            # - Setup the dataframe variables
            self.df                = self.rt_self.copyDataFrame(df)
            self.df_level          = 0
            self.dfs               = [self.df]
            self.dfs_layout        = [self.__renderView__(self.df)]
            self.graphs            = [self.rt_self.createNetworkXGraph(self.df, sl_params['relationships'])]

            # 
            # So... the following has to *NOT* happen here... if it does (and the view gets the initial svg), then
            # the svg won't get updated later on when it's re-set to the same value...  because if the value
            # doesn't change, then the svg won't get updated.
            #
            # self.mod_inner         = self.dfs_layout[self.df_level]._repr_svg_()

            # - Setup the selected entities information
            self.selected_entities = set()

            self.lock = threading.Lock()

            # Rewrite the _template with width and height
            self._template = '''<svg id="svgparent" width="'''+str(self.w)+'''" height="'''+str(self.h)+'''" viewBox="${viewBox}" tabindex="0" ''' + \
                              '''    onkeydown="${script('myOnKeyDown')}" onkeyup="${script('myOnKeyUp')}">  ''' + \
                              '''<svg id="mod" width="10000000" height="10000000"> ${mod_inner} </svg>  ''' + \
                              '''<rect id="drag" x="-10" y="-10" width="5" height="5" stroke="#000000" stroke-width="2" fill="none" />  ''' + \
                              '''<rect id="screen" x="0" y="0" width="10000000" height="10000000" opacity="0.05"  ''' + \
                              '''     onmouseover="${script('myOnMouseOver')}"      onmouseout="${script('myOnMouseOut')}"  ''' + \
                              '''     onmousedown="${script('myOnMouseDown')}"      onmousemove="${script('myOnMouseMove')}"  ''' + \
                              '''     onmouseup="${script('myOnMouseUp')}"          onmousewheel="${script('myOnMouseWheel')}" />  ''' + \
                              '''<path id="allentitieslayer" d="${allentitiespath}" fill="#000000" fill-opacity="0.01" stroke="none"  ''' + \
                              '''     onmouseover="${script('myOnMouseOver')}"      onmouseout="${script('myOnMouseOut')}"  ''' + \
                              '''     onmousedown="${script('downAllEntities')}"    onmousemove="${script('myOnMouseMove')}"   ''' + \
                              '''     onmouseup="${script('myOnMouseUp')}"          onmousewheel="${script('myOnMouseWheel')}" />  ''' + \
                              '''<path id="selectionlayer" d="${selectionpath}" fill="#ff0000" transform="" stroke="none"  ''' + \
                              '''     onmouseover="${script('myOnMouseOver')}"      onmouseout="${script('myOnMouseOut')}"  ''' + \
                              '''     onmousedown="${script('downMove')}"           onmousemove="${script('myOnMouseMove')}"  ''' + \
                              '''     onmouseup="${script('myOnMouseUp')}"          onmousewheel="${script('myOnMouseWheel')}" />  ''' + \
                              '''</svg>  '''

            super().__init__(**kwargs)

            self.param.watch(self.applyDragOp,  'drag_op_finished')
            self.param.watch(self.applyKeyOp,   'key_op_finished')
            self.param.watch(self.applyWheelOp, 'wheel_op_finished')

            self.companions = []

      #
      # vvv -- These methods are for external callers
      #

      # register companion visualizations
      def register_companion_viz(self, viz):
            self.companions.append(viz)

      # unregister companion visualizations
      def unregister_companion_viz(self, viz):
            if viz in self.companions: self.companions.remove(viz)

      #
      # ^^^ -- These methods are for external callers
      #

      #
      # __renderView__() - render the view
      #
      def __renderView__(self, __df__):
            _sp_ = self.rt_self.spreadLines(__df__, w=self.w, h=self.h, include_svg_viewbox=False, **self.sl_params)
            return _sp_

      #
      # setSelectedEntitiesAndNotifyOthers() - set the selected entities & notify any companion views
      #
      def setSelectedEntitiesAndNotifyOthers(self, _set_, callers=None):
            if callers is not None and self in callers: return
            if callers is None: 
                  callers  = set([self])
                  im_first = True
            else:               
                  callers.add(self)
                  im_first = False

            self.selected_entities = set(_set_)
            if im_first == False: self.__refreshView__(comp=False, all_ents=False)

            for c in self.companions:
                  if isinstance(c, RTSelectable): c.setSelectedEntitiesAndNotifyOthers(_set_, callers=callers)

      #
      # applyKeyOp() - apply specified key operation
      #
      async def applyKeyOp(self,event):
            self.lock.acquire()
            try:
                  _sp_ = self.dfs_layout[self.df_level]
                  #
                  # 
                  #
                  if self.key_op_finished == 'f':
                        if len(self.selected_entities) > 0:
                              pass

                  #
                  # 'x'|'p' - remove selected nodes from the dataset (push the stack)
                  # ... 'X'|'P' restore removed nodes (pop the stack)
                  #                  
                  elif self.key_op_finished == 'x' or self.key_op_finished == 'X' or self.key_op_finished == 'p' or self.key_op_finished == 'P':
                        if (self.key_op_finished == 'X' or self.key_op_finished == 'P') and self.df_level > 0: # pop the stack
                              self.popStack()

                        elif len(self.selected_entities) > 0: # push the stack
                              _g_ = copy.deepcopy(self.graphs[self.df_level])
                              for _entity_ in self.selected_entities: _g_.remove_node(_entity_)
                              _df_ = self.rt_self.filterDataFrameByGraph(self.dfs[self.df_level], self.sl_params['relationships'], _g_)
                              if len(_df_) > 0: self.pushStack(_df_, _g_)

            finally:
                  self.key_op_finished = ''
                  self.lock.release()

      #
      # Drag operation state
      #
      drag_op_finished  = param.Boolean(default=False)
      drag_x0           = param.Number(default=0)
      drag_y0           = param.Number(default=0)
      drag_x1           = param.Number(default=10)
      drag_y1           = param.Number(default=10)

      # Key States
      shiftkey         = param.Boolean(default=False)
      ctrlkey          = param.Boolean(default=False)
      last_key         = param.String(default='')
      key_op_finished  = param.String(default='')

      # Mouse States
      x_mouse          = param.Number(default=0)
      y_mouse          = param.Number(default=0)
      has_focus        = param.Boolean(default=False)

      #
      # Wheel operation state & method
      #
      wheel_x           = param.Number(default=0)
      wheel_y           = param.Number(default=0)
      wheel_rots        = param.Integer(default=0) # Mult by 10 and rounded...
      wheel_op_finished = param.Boolean(default=False)

      #
      # applyDragOp() - select the nodes within the drag operations bounding box.
      #
      async def applyDragOp(self, event):
            self.lock.acquire()
            try:
                  if self.drag_op_finished:
                        _x0,_y0,_x1,_y1 = min(self.drag_x0, self.drag_x1), min(self.drag_y0, self.drag_y1), max(self.drag_x1, self.drag_x0), max(self.drag_y1, self.drag_y0)
                        if _x0 == _x1: _x1 += 1
                        if _y0 == _y1: _y1 += 1
                        _rect_ = Polygon([(_x0,_y0), (_x0,_y1), (_x1,_y1), (_x1,_y0)])
                        _overlapping_entities_  = set(self.dfs_layout[self.df_level].overlappingEntities(_rect_))
                        if _overlapping_entities_ is None: _overlapping_entities_ = set()

                        if   self.shiftkey and self.ctrlkey: self.setSelectedEntitiesAndNotifyOthers(set(self.selected_entities) & set(_overlapping_entities_))
                        elif self.shiftkey:                  self.setSelectedEntitiesAndNotifyOthers(set(self.selected_entities) - set(_overlapping_entities_))
                        elif self.ctrlkey:                   self.setSelectedEntitiesAndNotifyOthers(set(self.selected_entities) | set(_overlapping_entities_))
                        else:                                self.setSelectedEntitiesAndNotifyOthers(_overlapping_entities_)
                        
                        self.__refreshView__()
            finally:
                  self.drag_op_finished = False
                  self.lock.release()

      #
      # applyWheelOp() - apply mouse wheel operation (zoom in & out)
      #
      async def applyWheelOp(self, event):
            self.lock.acquire()
            try:
                  self.__refreshView__() # used to force a re-render the first time ... which cause the actual visualization to be displayed
            finally:
                  self.wheel_op_finished = False
                  self.wheel_rots        = 0            
                  self.lock.release()

      #
      # __refreshView__() - refresh the view
      #    comp        - refresh the visualization
      #    all_ents    - refresh all entities path
      #    sel_ents    - refresh selected entities path
      #
      def __refreshView__(self, comp=True, all_ents=True, sel_ents=True):
            if (comp):     
                  self.viewBox                         = self.dfs_layout[self.df_level].viewBox()
                  self.vx, self.vy, self.vw, self.vh   = self.dfs_layout[self.df_level].viewBoxRect()
                  self.mod_inner                       = self.dfs_layout[self.df_level]._repr_svg_()
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
                  if isinstance(c, RTStackable): c.popStack(callers=callers)

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
                  if isinstance(c, RTStackable): c.setStackPosition(i_found, callers=callers)

      #
      # pushStack() - push a dataframe onto the stack
      #
      def pushStack(self, df, g=None, callers=None):
            if callers is not None and self in callers: return
            if callers is None: callers = set([self])
            else:               callers.add(self)

            if g is None: g = self.rt_self.createNetworkXGraph(df, self.sl_params['relationships'])

            # This is necessary to shrink the stack
            if len(self.dfs_layout) > (self.df_level+1):
                  new_dfs, new_dfs_layout, new_graphs = [], [], []
                  for i in range(self.df_level+1):
                        new_dfs.append(self.dfs[i]), new_dfs_layout.append(self.dfs_layout[i]), new_graphs.append(self.graphs[i])
                  self.dfs, self.dfs_layout, self.graphs = new_dfs, new_dfs_layout, new_graphs

            # Render the new view and update all of the stack variables
            _sl_ = self.__renderView__(df)
            self.dfs        .append(df)
            self.dfs_layout .append(_sl_)
            self.graphs     .append(g)
            self.df_level += 1

            # Refresh the view
            self.setSelectedEntitiesAndNotifyOthers(self.selected_entities & g.nodes())
            self.__refreshView__()

            # Let companion vizualizations know
            for c in self.companions:
                  if isinstance(c, RTStackable): c.pushStack(df, callers=callers)

      #
      # Panel Javascript Definitions
      #
      _scripts = {
            'render':"""
                  mod.innerHTML            = data.mod_inner;
                  state.x0_drag            = state.y0_drag = -10;
                  state.x1_drag            = state.y1_drag =  -5;
                  state.x_raw              = state.y_raw   = -10;
                  state.x_trans            = state.y_trans = -10;
                  data.has_focus           = false;
                  data.shiftkey            = false;
                  data.ctrlkey             = false;
                  state.drag_op            = false;
            """,

            'myOnMouseOver':"""
                  data.has_focus = true;
                  svgparent.focus();
                  // screen.setAttribute("opacity", "0.05");
            """,

            'myOnMouseOut':"""
                  data.has_focus = false;
                  // screen.setAttribute("opacity", "0.10");
            """,

            'myOnKeyDown':"""
                  data.ctrlkey  = event.ctrlKey;
                  data.shiftkey = event.shiftKey;
                  data.last_key = event.key;
                  if      (event.key == "f") { data.key_op_finished = "f"; } // set the focus to the selected entities
                  else if (event.key == "p") { data.key_op_finished = "p"; }
                  else if (event.key == "P") { data.key_op_finished = "P"; }
                  else if (event.key == "x") { data.key_op_finished = "x"; }
                  else if (event.key == "X") { data.key_op_finished = "X"; }
            """,

            'myOnKeyUp':"""
                  data.ctrlkey  = event.ctrlKey;
                  data.shiftkey = event.shiftKey;
            """,

            'transCoords':"""
                  // Converts mouse position into transformed coordinates
                  sw        = svgparent.getAttribute("width"); sh        = svgparent.getAttribute("height"); _ratio_   = data.vw/sw;
                  if ((data.vw/data.vh) > (sw/sh)) { _ratio_ = data.vw/sw; } else { _ratio_ = data.vh/sh;}
                  state.x_trans = (data.vx + data.vw/2) + (state.x_raw - sw/2)*_ratio_; state.y_trans = (data.vy + data.vh/2) + (state.y_raw - sh/2)*_ratio_;
            """,

            'myOnMouseMove':"""
                  data.ctrlkey   = event.ctrlKey;
                  data.shiftkey  = event.shiftKey;

                  state.x_raw = event.offsetX; state.y_raw = event.offsetY; self.transCoords();

                  data.x_mouse   = state.x_trans; data.y_mouse   = state.y_trans;
                  state.x1_drag  = state.x_trans; state.y1_drag  = state.y_trans;

                  if (state.drag_op)               { self.myUpdateDragRect(); }
            """,

            'downAllEntities':"""
                  data.ctrlkey  = event.ctrlKey;
                  data.shiftkey = event.shiftKey;

                  state.x_raw = event.offsetX; state.y_raw = event.offsetY; self.transCoords();

                  if (event.button == 0) {
                              state.x0_drag            = state.x_trans; state.y0_drag            = state.y_trans;
                              state.x1_drag            = state.x_trans; state.y1_drag            = state.y_trans;
                  }
            """,
            'myOnMouseDown':"""
                  state.x_raw = event.offsetX; state.y_raw = event.offsetY; self.transCoords();
                  
                  if (event.button == 0) {
                        state.x0_drag  = state.x_trans; state.y0_drag  = state.y_trans;
                        state.x1_drag  = state.x_trans; state.y1_drag  = state.y_trans;
                        state.drag_op  = true;             
                        self.myUpdateDragRect();
                  }
            """,

            'downMove':"""
                  if (event.button == 0) {
                        state.x_raw = event.offsetX; state.y_raw = event.offsetY; self.transCoords();

                        state.x0_drag  = state.x1_drag  = state.x_trans;
                        state.y0_drag  = state.y1_drag  = state.y_trans;
                        state.move_op  = true;
                  }
            """,

            'myOnMouseUp':"""
                  if (event.button == 0) {
                        state.x_raw = event.offsetX; state.y_raw = event.offsetY; self.transCoords();
                        state.x1_drag         = state.x_trans; state.y1_drag         = state.y_trans;
                        if (state.drag_op) {
                              state.shiftkey        = event.shiftKey;
                              state.drag_op         = false;
                              self.myUpdateDragRect();
                              data.drag_x0          = state.x0_drag; data.drag_y0          = state.y0_drag; 
                              data.drag_x1          = state.x1_drag;  data.drag_y1          = state.y1_drag;
                              data.drag_op_finished = true;
                        }
                  }
            """,

            'myOnMouseWheel':"""
                  event.preventDefault();
                  state.x_raw = event.offsetX; state.y_raw = event.offsetY; self.transCoords();
                  data.wheel_x = state.x_trans; data.wheel_y = state.y_trans;
                  data.wheel_rots  = Math.round(10*event.deltaY);
                  data.wheel_op_finished = true;
            """,

            'mod_inner':"""
                  mod.innerHTML = data.mod_inner;
            """,

            'selectionpath':"""
                  selectionlayer.setAttribute("d", data.selectionpath);
            """,
            
            'myUpdateDragRect':"""
                  if (state.drag_op) {
                        x = Math.min(state.x0_drag, state.x1_drag); 
                        y = Math.min(state.y0_drag, state.y1_drag);
                        w = Math.abs(state.x1_drag - state.x0_drag)
                        h = Math.abs(state.y1_drag - state.y0_drag)
                        drag.setAttribute('x',x);     drag.setAttribute('y',y);
                        drag.setAttribute('width',w); drag.setAttribute('height',h);
                        if      (data.shftkey && data.ctrlkey)  drag.setAttribute('stroke','#0000ff');
                        else if (data.shftkey)                  drag.setAttribute('stroke','#ff0000');
                        else if (                data.ctrlkey)  drag.setAttribute('stroke','#00ff00');
                        else                                    drag.setAttribute('stroke','#000000');
                  } else {
                        drag.setAttribute('x',-10);   drag.setAttribute('y',-10);
                        drag.setAttribute('width',5); drag.setAttribute('height',5);
                  }
            """
      }




