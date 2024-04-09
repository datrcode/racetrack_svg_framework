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

__name__ = 'rt_entity_position'

#
# RTEntityPosition Base Class
#
class RTEntityPosition(object):
    #
    # Constructor
    #
    def __init__(self, entity, rt, component_instance, point_to_xy, attachment_point_vec, svg_id, svg_markup, widget_id):
        self.entity                = entity
        self.rt                    = rt
        self.component_instance    = component_instance
        self.point_to_xy           = point_to_xy
        self.attachment_point_vecs = [attachment_point_vec]
        self.svg_id                = svg_id
        self.svg_markup            = svg_markup
        self.widget_id             = widget_id

    def xy(self):
        return self.point_to_xy
    def attachmentPointVecs(self):
        return self.attachment_point_vecs
    def svgId(self):
        return self.svg_id
    def svg(self):
        return self.svg_markup
    def widgetId():
        return self.widget_id