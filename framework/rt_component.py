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
# Annotations Mixin
#
class RTComponent(object):
        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            pass

        #
        # renderSVG() - create the SVG
        #
        def renderSVG(self,just_calc_max=False):
            pass

        #
        # smallMultipleFeatureVector()
        # ... feature vector for comparison with other small multiple instances of this class
        #
        def smallMultipleFeatureVector(self):
            pass

        #
        # Determine which dataframe geometries overlap with a specific
        #
        def overlappingDataFrames(self, to_intersect):
            pass
