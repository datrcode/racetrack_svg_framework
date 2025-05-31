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

#
# Things To Add (Loosely Grouped):
#
# - Add all records on (currently) visible edges (don't want to push onto the stack though... so unsure how to do this)
#   (if selected nodes, limit to them)
# - Add all records on (currently) visible nodes (don't want to push onto the stack though... so unsure how to do this)
#   (if selected nodes, limit to them)
# - Only keep a single record on visible edges (should make everything faster) (subsetOneRecordPerEdge)
#
#
# - Make the stack the top level and then whatever this one currently is... and get rid of everything else ... stack == |2|
#   (I don't care how I got here... get rid of all the intermediate stacks)
# - Make this current stack the top level... and get rid of everything else ... stack == |1|
#   (I only care about this data... nothing else... get rid of everything else) // seems like this could be "don't do this by accident"
# - Remove the current from the top stack and then show whatever is left over... stack == |2|
#   (This is the stuff I definitely don't want)
#
#
# - MOSTLY DONE - Treemap based layout on node colors (if selected nodes, only treemap those & fit them in the selection bounds)
#   (difference between collapse them to a node, do a grid layout, or do something smarter that organizes the nodes within the treemap layout)
#   ... // common theme of "optimizing node selection in a structure"
# - The templated layouts for simple patterns
#
#
# - Tagging (would require pre-setup work but then maybe key/key combos to do the following ... needs to be associated w/ node color lookups)
# -- Add tag to selected node(s) (replace version)
# -- Add tag to selected node(s) (in addition to already assigned tags)
# -- Propagate tags from selected node(s) to their neighbors ... not as simple as it looks because of edge cases
# -- Remove tag from selected node(s)
#    (would need a mechanism to select which tag is to be applied/removed)
# -- Probably need multiple labels ... node label (from the dataframe), node label lookups, and then the node color lookups
#    ... note that it looks like linkNode handles dataframe node name w/ label lookups differently than link does
#    ... and there's confusion about selectEntities() and selectedEntities() vis-a-vis node label lookups
# - Entity Tagging vs Record Tagging ... stuff above is about entity tagging
#   ... Record tagging is also relevant ... and there are some helper methods put together in the rt_annotations_mixin.py file
#       (however, that tag stuff is just set/add ... needs to include remove ... and it's unclear how to propagate to sub-dataframes...)
#   ... however, it's unclear how to propagate a record tag to all of the dataframes in the stack at once
#   ... linknode is entity tagging ... histogram is either entity or record tagging ... xy is record tagging ... temporal is record tagging
#   ... record tagging should really just be done as columns ... it could be a more complicated construct but that's just going to
#       mess up the purpose of things in columns...
#   ... entity tagging should underly the entire system and not be hacked together on a per component basis...
#
# Other Changes:
#
# (Selected nodes copied into system clipboard -- is that possible in javascript? or on the python side?)
# (Optimize the circular layout to minimize connection lengths / edge crossings) // common theme of "optimizing node selection in a structure"
# (Dynamic labels? ... maybe make a dummy label i the template & then dynamically reposition it? ... needs to all be done in Javascript...)
# (Edge labels? ... linkNode supports.... but not link)
# (Render help ... or at least the cheat sheet of key commands)
# (More granularity on selections -- especially when a cloud has some selected elements & some not)
# ... the selection implementation is complex because it modifies a single SVG path element...
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

from .rt_stackable import RTStackable, RTSelectable

__name__ = 'rt_graph_interactive_layout'

#
# ReactiveHTML Class for Panel Implementation
#
class RTGraphInteractiveLayout(ReactiveHTML, RTStackable, RTSelectable):
    #
    # Print Representation
    #
    def __str__(self): return """
External Methods:

register_companion_viz(viz)
unregister_companion_viz(viz)

saveLayout(filename)
loadLayout(filename)

selectEntities(selection, set_op='replace'|'add'|'subtract'|'intersect', method='exact'|'substring'|'regex', ignore_case=True)
selectedEntities() - labeled nodes (if labels are set)
selectedNodes() - dataframe nodes

forceRender() - force a re-render (e.g., if the ln parameters were externally changed)

-------------------------------------------------
Set Operations For Selection
------------+------------------------------------
<none>      | replace (default)
shift       | subtract from current selection
ctrl        | add to current selection
ctrl-shift  | intersect with current selection

-------------------------------------------------
Interactivity Key Commands
----+--------------------------------------------
c   | reset view or focus view on selected
    | shift-c focus view on selected + neighbors
e   | expand selection
    | shift-e       | expand selection (directed graph)
g   | layout upon next mouse drag
    | shift-g       | cycle through layout modes (rectangular, circular, or sunflower)
q   | invert selection
    | shift-q       | common neighbors
s   | set sticky labels
    | shift-s       | remove sticky labels from selected
    | ctrl-s        | add selected to sticky labels
    | ctrl-shift-s  | cycle label visibility (all | sticky | none) 
t   | consolidate all nodes at the mouse location
    | shift-t       | horizontally
    | ctrl-t        | vertically
u   | undo last layout action (limited undo's)
x   | remove selected nodes (push stack)
    | shift-x       | pop stack
y   | line layout
    | shift-y       | horizontally
    | ctrl-y        | vertically
z   | select node under mouse by color (shift, ctrl, and ctrl-shift apply)
1-6 | select numbered degree
7   | select degree 10 -> 20
8   | select degree 20 -> 50
9   | select degree 50 -> 100
0   | select degree 100 -> inf    
    """
    #
    # Inner Modification for RT SVG Render
    #
    mod_inner       = param.String(default="""<circle cx="300" cy="200" r="10" fill="red" />""")

    #
    # All Entities Path
    #
    allentitiespath = param.String(default="M -100 -100 l 10 0 l 0 10 l -10 0 l 0 -10 Z")

    #
    # Selection Path
    #
    selectionpath   = param.String(default="M -100 -100 l 10 0 l 0 10 l -10 0 l 0 -10 Z")

    #
    # Information String
    #
    info_str        = param.String(default=" | | grid")

    #
    # Layout Mode String
    #
    layout_mode     = param.String(default="grid")

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
                 rt_self,              # RACETrack instance
                 df,                   # data frame
                 ln_params,            # linknode params
                 w            = 600,   # width
                 h            = 400,   # height
                 use_linknode = False, # use linknode (vice link)
                 **kwargs):
        # Setup specific instance information
        # - Copy the member variables
        self.rt_self           = rt_self
        self.ln_params         = ln_params
        if 'pos' not in ln_params.keys(): ln_params['pos'] = {}
        self.pos               = ln_params['pos']
        self.w                 = w
        self.h                 = h
        self.use_linknode      = use_linknode
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
        self.selected_entities = set()

        # Constants
        self.GRID                 = 'grid'
        self.CIRCLE               = 'circle'
        self.SUNFLOWER            = 'sunflower'
        self.GRID_BY_COLOR        = 'grid (color)'
        self.GRID_BY_COLOR_CLOUDS = 'grid (color, clouds)'
        self.layout_modes         = [self.GRID, self.CIRCLE, self.SUNFLOWER, self.GRID_BY_COLOR, self.GRID_BY_COLOR_CLOUDS]

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
                         '''    <text id="infostr" x="5"   y="''' + str(self.h-4) + '''" fill="#000000" font-size="10px"> ${info_str} </text> ''' + \
                         '''    <path id="allentitieslayer" d="${allentitiespath}" fill="#000000" fill-opacity="0.01" stroke="none" ''' + \
                         '''          onmousedown="${script('downAllEntities')}" onmousemove="${script('moveEverything')}"  ''' + \
                         '''          onmouseup="${script('upEverything')}"      onmousewheel="${script('mouseWheel')}" /> ''' + \
                         '''    <path id="selectionlayer" d="${selectionpath}" fill="#ff0000" transform="" stroke="none" ''' + \
                         '''          onmousedown="${script('downMove')}"        onmousemove="${script('moveEverything')}" ''' + \
                         '''          onmouseup="${script('upEverything')}"      onmousewheel="${script('mouseWheel')}" /> ''' + \
                         '''</svg>'''

        # Previous layouts (for undo operations)
        self.previous_layouts = []
        self.max_undo_levels  = 20

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
                self.setSelectedEntitiesAndNotifyOthers(_set_)
            else: # just use the selection
                if ignore_case:
                    _set_ = set()
                    for _node_ in all_nodes:
                        _node_cased_ = str(_node_).lower()
                        if _node_cased_ in selection_as_set: _set_.add(_node_)
                else:
                    _set_ = selection_as_set & all_nodes

        if   set_op == 'replace':   self.setSelectedEntitiesAndNotifyOthers(_set_)
        elif set_op == 'add':       self.setSelectedEntitiesAndNotifyOthers(self.selected_entities | _set_)
        elif set_op == 'subtract':  self.setSelectedEntitiesAndNotifyOthers(self.selected_entities - _set_)
        elif set_op == 'intersect': self.setSelectedEntitiesAndNotifyOthers(self.selected_entities & _set_)

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
    # updateLinkNodeParam() - update a param & refresh the views
    # - performs at all levels of the stack
    #
    def updateLinkNodeParam(self, name, value):
        self.ln_params[name] = value
        for i in range(len(self.dfs_layout)): 
            setattr(self.dfs_layout[i], name, value)
            self.dfs_layout[i].invalidateRender()
        self.__refreshView__(comp=True)

    #
    # ^^^ -- These methods are for external callers
    #

    #
    # __renderView__() - render the view
    #
    def __renderView__(self, __df__):
        if self.use_linknode or self.rt_self.isPandas(__df__): _ln_ = self.rt_self.linkNode(__df__, w=self.w, h=self.h, **self.ln_params)
        else:                                                  _ln_ = self.rt_self.link    (__df__, w=self.w, h=self.h, **self.ln_params)
        return _ln_

    #
    # __cacheNodePositions__() - cache the node positions for undo operations
    #
    def __cacheNodePositions__(self):
        _copy_ = copy.deepcopy(self.dfs_layout[self.df_level].pos)
        self.previous_layouts.append(_copy_) # if len(self.previous_layouts) == 0 or self.previous_layouts[-1] != _copy_ # doesn't work with the output of nx.spring_layout()...
        while len(self.previous_layouts) > self.max_undo_levels: self.previous_layouts.pop(0)


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
                if   self.layout_shape == self.GRID:
                    pos_adj = self.rt_self.rectangularArrangement(self.graphs[self.df_level], as_list, bounds=(x0,y0,x1,y1))
                    self.__cacheNodePositions__()
                    for _node_ in pos_adj: _ln_.pos[_node_] = (float(_ln_.xT_inv(pos_adj[_node_][0])),float(_ln_.yT_inv(pos_adj[_node_][1])))
                    nodes_moved = True
                elif self.layout_shape == self.GRID_BY_COLOR or self.layout_shape == self.GRID_BY_COLOR_CLOUDS:
                    _node_to_color_ = {}
                    for _node_ in as_list: _node_to_color_[_node_] = _ln_.nodeColor(_node_)
                    pos_adj = self.rt_self.treeMapNodeColorPlacement(self.graphs[self.df_level], as_list, _node_to_color_, 
                                                                     collapse=(self.layout_shape == self.GRID_BY_COLOR_CLOUDS),
                                                                     bounds=(x0,y0,x1,y1))
                    for _node_ in pos_adj: _ln_.pos[_node_] = (float(_ln_.xT_inv(pos_adj[_node_][0])),float(_ln_.yT_inv(pos_adj[_node_][1])))
                    nodes_moved = True
                elif self.layout_shape == self.CIRCLE:
                    wx0, wy0 = _ln_.xT_inv(x0), _ln_.yT_inv(y0)
                    wx1, wy1 = _ln_.xT_inv(x1), _ln_.yT_inv(y1)
                    r = sqrt((wx0 - wx1)**2 + (wy0 - wy1)**2)
                    if r < 0.001: r = 0.001
                    pos_adj = self.rt_self.circularOptimizedArrangement(self.graphs[self.df_level], as_list, _ln_.pos, xy=(wx0,wy0), r=r)
                    self.__cacheNodePositions__()
                    for _node_ in pos_adj: _ln_.pos[_node_] = (pos_adj[_node_][0],pos_adj[_node_][1])
                    nodes_moved = True
                elif self.layout_shape == self.SUNFLOWER:
                    r = sqrt((x0 - x1)**2 + (y0 - y1)**2)
                    pos_adj = self.rt_self.sunflowerSeedArrangement(self.graphs[self.df_level], as_list, xy=(x0,y0), r_max=r)
                    self.__cacheNodePositions__()
                    for _node_ in pos_adj: _ln_.pos[_node_] = (float(_ln_.xT_inv(pos_adj[_node_][0])),float(_ln_.yT_inv(pos_adj[_node_][1])))
                    nodes_moved = True
                elif self.layout_shape == "line" or self.layout_shape == "v-line" or self.layout_shape == "h-line":
                    if   self.layout_shape == "v-line": x0, x1, dx = x1, x1, 0
                    elif self.layout_shape == "h-line": y0, y1, dy = y1, y1, 0
                    wx0, wy0 = _ln_.xT_inv(x0), _ln_.yT_inv(y0)
                    wx1, wy1 = _ln_.xT_inv(x1), _ln_.yT_inv(y1)
                    pos_adj = self.rt_self.linearOptimizedArrangement(self.graphs[self.df_level], as_list, _ln_.pos, ((wx0,wy0),(wx1,wy1)))
                    self.__cacheNodePositions__()
                    for _node_ in pos_adj: _ln_.pos[_node_] = (pos_adj[_node_][0],pos_adj[_node_][1])
                    nodes_moved = True
            elif len(as_list) == 1:
                self.__cacheNodePositions__()
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
        self.setSelectedEntitiesAndNotifyOthers(self.selected_entities & g.nodes())
        self.__refreshView__()

        for c in self.companions:
            if isinstance(c, RTStackable): c.pushStack(df, callers=callers)

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

                    self.setSelectedEntitiesAndNotifyOthers(_new_set_)
                else:
                    _new_set_ = set(self.selected_entities)
                    for _node_ in self.selected_entities:
                        for _nbor_ in self.graphs[self.df_level].neighbors(_node_):
                            _new_set_.add(_nbor_)
                    self.setSelectedEntitiesAndNotifyOthers(_new_set_)

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
                    self.setSelectedEntitiesAndNotifyOthers(_new_set_)

                self.__refreshView__(comp=False, all_ents=False)

            #
            # "S" - Set Sticky Labels & Remove Sticky Labels
            #
            elif self.key_op_finished == 's' or self.key_op_finished == 'S':
                _label_set_changed_ = True
                if   self.shiftkey and self.ctrlkey:
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
                    _label_set_changed_ = False
                    self.__refreshView__(all_ents=False, sel_ents=False)
                elif self.shiftkey:
                    self.sticky_labels  = self.sticky_labels - self.selected_entities # subtract from the current set
                elif                   self.ctrlkey:
                    self.sticky_labels = self.sticky_labels | self.selected_entities  # add to the current set
                else:
                    self.sticky_labels  = set(self.selected_entities)                 # make a new set object with the selected

                # if the set of sticky labels has changed, update the label set & refresh
                if _label_set_changed_:
                    if self.label_mode == 'sticky labels': _ln_.labelOnly(self.sticky_labels)
                    self.ln_params['label_only'] = self.sticky_labels
                    self.__refreshView__(info=False, all_ents=False, sel_ents=False)

            #
            # "T" - Collapse (to a point, horizontal line, or vertical line)
            #
            elif len(self.selected_entities) > 0 and (self.key_op_finished == 't' or self.key_op_finished == 'T'):
                self.__cacheNodePositions__()

                # Horizontal Collapse
                if   self.shiftkey:
                    for _entity_ in self.selected_entities:
                        xy = _ln_.pos[_entity_]
                        _ln_.pos[_entity_] = (xy[0], _ln_.yT_inv(self.y_mouse))

                # Vertical Collapse
                elif self.ctrlkey:
                    for _entity_ in self.selected_entities:
                        xy = _ln_.pos[_entity_]
                        _ln_.pos[_entity_] = (_ln_.xT_inv(self.x_mouse), xy[1])

                # Collapse to a single point
                else:
                    for _entity_ in self.selected_entities:
                        xy = _ln_.pos[_entity_]
                        _ln_.pos[_entity_] = (_ln_.xT_inv(self.x_mouse), _ln_.yT_inv(self.y_mouse))

                for i in range(len(self.dfs_layout)): self.dfs_layout[i].invalidateRender()
                self.__refreshView__(info=False)

            elif self.key_op_finished == 'u' and len(self.previous_layouts) > 0:
                # Restore the last layout
                _previous_pos_ = self.previous_layouts[-1]
                for _entity_ in _previous_pos_: _ln_.pos[_entity_] = _previous_pos_[_entity_]

                # Remove the last layout from the stack
                self.previous_layouts = self.previous_layouts[:-1]

                # Invalidate the renders and refresh the view
                for i in range(len(self.dfs_layout)): self.dfs_layout[i].invalidateRender()
                self.__refreshView__(info=False)
            
            #
            # "Z" - Select nodes with the same color as the one that the mouse is over
            #
            elif self.key_op_finished == 'z' or self.key_op_finished == 'Z':
                _entities_  = self.dfs_layout[self.df_level].entitiesAtPoint((self.x_mouse,self.y_mouse))
                if _entities_ is None: _entities_ = set()
                _colors_    = set()
                for _entity_ in _entities_: _colors_.add(self.dfs_layout[self.df_level].nodeColor(_entity_))
                _entities_  = set()
                for _color_ in _colors_: _entities_ = _entities_ | set(self.dfs_layout[self.df_level].nodesWithColor(_color_))
                if self.shiftkey and self.ctrlkey: _set_op_ = 'intersect'
                elif self.shiftkey:                _set_op_ = 'subtract'
                elif self.ctrlkey:                 _set_op_ = 'add'
                else:                              _set_op_ = 'replace'
                self.selectEntities(_entities_, _set_op_, 'exact')

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
            # 'x' - remove selected nodes from the dataset (push the stack)
            # ... 'X' restore removed nodes (pop the stack)
            #
            elif self.key_op_finished == 'x' or self.key_op_finished == 'X':
                if self.key_op_finished == 'X' and self.df_level > 0: # pop the stack
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

                if   self.shiftkey:               self.setSelectedEntitiesAndNotifyOthers(self.selected_entities - _match_)
                elif len(self.selected_entities): self.setSelectedEntitiesAndNotifyOthers(self.selected_entities & _match_)
                else:                             self.setSelectedEntitiesAndNotifyOthers(_match_)

                self.__refreshView__(comp=False, all_ents=False)

            #
            # Next Layout Option
            #
            elif self.key_op_finished == 'G':
                _index_          = self.layout_modes.index(self.layout_mode)
                self.layout_mode = self.layout_modes[(_index_+1) % len(self.layout_modes)]
            
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

                if   self.shiftkey and self.ctrlkey: self.setSelectedEntitiesAndNotifyOthers(set(self.selected_entities) & set(_overlapping_entities_))
                elif self.shiftkey:                  self.setSelectedEntitiesAndNotifyOthers(set(self.selected_entities) - set(_overlapping_entities_))
                elif self.ctrlkey:                   self.setSelectedEntitiesAndNotifyOthers(set(self.selected_entities) | set(_overlapping_entities_))
                else:                                self.setSelectedEntitiesAndNotifyOthers(_overlapping_entities_)
                
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
                if self.drag_x0 == self.drag_x1 and self.drag_y0 == self.drag_y1 and self.shiftkey:
                    _point_entities_  = self.dfs_layout[self.df_level].entitiesAtPoint((self.drag_x0,self.drag_y0))
                    self.__refreshView__(comp=False, all_ents=False)
                else:
                    self.__cacheNodePositions__()
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

                if   self.ctrlkey:  self.setSelectedEntitiesAndNotifyOthers((set(self.selected_entities) | set(_overlapping_entities_)))
                elif self.shiftkey: self.setSelectedEntitiesAndNotifyOthers((set(self.selected_entities) - set(_overlapping_entities_)))
                else:               self.setSelectedEntitiesAndNotifyOthers(set(_overlapping_entities_))

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
            infostr.innerHTML        = data.info_str;
            state.x0_drag            = state.y0_drag = -10;
            state.x1_drag            = state.y1_drag =  -5;
            data.shiftkey            = false;
            data.ctrlkey             = false;
            state.drag_op            = false;
            state.move_op            = false;
            state.unselected_move_op = false;
            state.layout_op          = false; // true if next mouse button 1 press is the begin of a layout
            state.layout_line_flag   = false; // true if the shape will be overrode by the line version
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
            else if (event.key == "g") { state.layout_op        = true; // Mouse press is layout shape
                                         state.layout_line_flag = false; } 
            else if (event.key == "G") { data.key_op_finished = 'G';  } // Iterate through layout shapes
            else if (event.key == "q") { data.key_op_finished = 'q';  } // Invert selection
            else if (event.key == "Q") { data.key_op_finished = 'Q';  } // Select common neighbors to selected nodes
            else if (event.key == "s") { data.key_op_finished = 's';  } // Set sticky labels
            else if (event.key == "S") { data.key_op_finished = 'S';  } // Subtract selected from sticky labels
            else if (event.key == "t") { data.key_op_finished = 't';  } // Collapse selected to a single point
            else if (event.key == "T") { data.key_op_finished = 'T';  } // Horizontally collapse selected
            else if (event.key == "u") { data.key_op_finished = 'u';  } // Undo last layout
            else if (event.key == "x") { data.key_op_finished = 'x';  } // push the stack (remove the selected from the current graph)
            else if (event.key == "X") { data.key_op_finished = 'X';  } // pop the stack (add removed nodes back in)
            else if (event.key == "y") { state.layout_op        = true; // Mouse press is layout line
                                         state.layout_line_flag = true;  }
            else if (event.key == "Y") { state.layout_op        = true; // Mouse press is layout line
                                         state.layout_line_flag = true;  }
            else if (event.key == "z" ||                                // Select nodes with the same color as the one under the mouse
                     event.key == "Z") { data.key_op_finished = 'z';     }
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
            if (event.key == "g" || event.key == "y" || event.key == "Y") { state.layout_op = state.layout_line_flag = false; }
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
                if (state.layout_op) { 
                    if (state.layout_line_flag) { 
                        if      (data.ctrlkey)  { state.layout_op_shape = "v-line"; }
                        else if (data.shiftkey) { state.layout_op_shape = "h-line"; }
                        else                    { state.layout_op_shape = "line";   }
                    }
                    else                        { state.layout_op_shape = data.layout_mode; }
                    self.myUpdateLayoutOp();
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
            } else if (state.layout_op_shape == "grid" || 
                       state.layout_op_shape == "grid (color)" || 
                       state.layout_op_shape == "grid (color, clouds)") { reset_rect = false;
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
            svgparent.focus(); // else it loses focus on every render...
        """,
        'selectionpath':"""
            selectionlayer.setAttribute("d", data.selectionpath);
            svgparent.focus(); // else it loses focus on every render...
        """,
        'info_str': """
            infostr.innerHTML = data.info_str;
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

