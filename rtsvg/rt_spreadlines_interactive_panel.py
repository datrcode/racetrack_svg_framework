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

import panel as pn
import param

from panel.reactive import ReactiveHTML

__name__ = 'rt_spreadlines_interactive_panel'

#
# ReactiveHTML Class for Panel Implementation
#
class RTSpreadLinesInteractivePanel(ReactiveHTML, RTStackable, RTSelectable):
    #
    # Inner Modification for RT SVG Render
    #
    mod_inner       = param.String(default="""<circle cx="300" cy="200" r="10" fill="red" />""")

    #
    # Panel Template
    # - rewritten in constructor with width and height filled in
    #
    _template = """
<svg id="svgparent" width="600" height="200" tabindex="0" 
     onkeypress="${script('keyPress')}" onkeydown="${script('keyDown')}" onkeyup="${script('keyUp')}">
    <svg id="mod" width="600" height="200"> ${mod_inner} </svg>
    <rect id="drag" x="-10" y="-10" width="5" height="5" stroke="#000000" stroke-width="2" fill="none" />
    <rect id="screen" x="0" y="0" width="600" height="200" opacity="0.05"
          onmousedown="${script('downSelect')}"          onmousemove="${script('moveEverything')}"
          onmouseup="${script('upEverything')}"          onmousewheel="${script('mouseWheel')}" />
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
        self.mod_inner         = self.dfs_layout[self.df_level]._repr_svg_()

        # - Setup the selected entities information
        self.selected_entities = set()

        self.lock = threading.lock()

        # Rewrite the _template with width and height
        self._template = '''<svg id="svgparent" width="'''+str(self.w)+'''" height="'''+str(self.h)+'''" tabindex="0" ''' + \
                         '''     onkeypress="${script('keyPress')}" onkeydown="${script('keyDown')}" onkeyup="${script('keyUp')}">  ''' + \
                         '''<svg id="mod" width="'''+str(self.w)+'''" height="'''+str(self.h)+'''"> ${mod_inner} </svg>  ''' + \
                         '''<rect id="drag" x="-10" y="-10" width="5" height="5" stroke="#000000" stroke-width="2" fill="none" />  ''' + \
                         '''<rect id="screen" x="0" y="0" width="'''+str(self.w)+'''" height="'''+str(self.h)+'''" opacity="0.05"  ''' + \
                         '''     onmousedown="${script('downSelect')}"          onmousemove="${script('moveEverything')}"  ''' + \
                         '''     onmouseup="${script('upEverything')}"          onmousewheel="${script('mouseWheel')}" />  ''' + \
                         '''<path id="allentitieslayer" d="${allentitiespath}" fill="#000000" fill-opacity="0.01" stroke="none"  ''' + \
                         '''     onmousedown="${script('downAllEntities')}" onmousemove="${script('moveEverything')}"   ''' + \
                         '''     onmouseup="${script('upEverything')}"      onmousewheel="${script('mouseWheel')}" />  ''' + \
                         '''<path id="selectionlayer" d="${selectionpath}" fill="#ff0000" transform="" stroke="none"  ''' + \
                         '''     onmousedown="${script('downMove')}"        onmousemove="${script('moveEverything')}"  ''' + \
                         '''     onmouseup="${script('upEverything')}"      onmousewheel="${script('mouseWheel')}" />  ''' + \
                         '''</svg>  '''

        super().__init__(**kwargs)

        self.param.watch(self.applyDragOp, 'drag_op_finished')
        self.param.watch(self.applyKeyOp,  'key_op_finished')

        self.companions = []
