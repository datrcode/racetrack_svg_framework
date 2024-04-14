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

__name__ = 'rt_component'

#
# RTComponent Base Class
# - Other than documentation, does this actually do any checking?
#
class RTComponent(object):

        # ===========================================================================================
        # Rendering
        # ===========================================================================================

        #
        # SVG Representation Renderer
        # - recommended that this method use a saved version of the last render:
        #   - self.last_render
        # - return value is an svg string
        # - for Jupyter integration, SVG must have the xmlns XML tag
        #
        def _repr_svg_(self):
            pass

        #
        # widgetId() - return the SVG widget id
        #
        def widgetId(self):
             return self.widget_id
        #
        # entityPositions() - return information about the entity geometry for rendering
        # - Empty list means either not implemented... or entity not in view...
        #
        # (originally developed in the RTChordDiagram component... probably overkill here //2024-03-31)
        #
        # - return the positions of the entity ... rendering had to have happened first
        def entityPositions(self, entity):
             return []

        #
        # renderSVG() - create the SVG Rendering
        # - recommend that this method save the rendering into the last_render member variable.
        # - return value is an svg string
        #
        def renderSVG(self, just_calc_max=False):
            pass

        # ===========================================================================================
        # Feature Vector Related Methods
        # ===========================================================================================

        #
        # smallMultipleFeatureVector()
        # ... feature vector for comparison with other small multiple instances of this class
        #
        def smallMultipleFeatureVector(self):
            pass

        # ===========================================================================================
        # Interactivity Related Methods
        # ===========================================================================================

        #
        # Determine which dataframe geometries overlap with a specific
        # - to_intersect should be a shapely shape
        # - return value is a pandas dataframe
        #
        def overlappingDataFrames(self, to_intersect):
            pass

        #
        # applyScrollEvent()
        # - scroll the view by the specified amount
        # - coordinate is optional / depends on the type of view
        # -- for example, histogram usage doesn't need the coordinate
        # -- however, linknode uses it for determining the zoom center
        # -- return True if the view actually changed (and needs a re-render)
        #
        def applyScrollEvent(self, scroll_amount, coordinate=None):
             return False

        #
        # applyMiddleClick()
        # - return True if the view actually changed (and needs a re-render)
        #
        def applyMiddleClick(self, coordinate):
            return False

        #
        # applyMiddleDrag()
        # - return True if the view actually changed (and needs a re-render)
        #        
        def applyMiddleDrag(self, coordinate, delta):
             return False

        #
        # applyViewConfiguration()
        # - apply the view configuration from another RTComponent (of the same type)
        # - return True if the view actually changed (and needs a re-render)
        #
        def applyViewConfiguration(self, other):
             return False
