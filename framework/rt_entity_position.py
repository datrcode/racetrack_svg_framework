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
    def __init__(self, entity, rt, component_instance, point_to_xy, attachment_point_vec, svg_id, svg_markup, widget_id, xy_offset=(0,0)):
        self.entity                = entity
        self.rt                    = rt
        self.component_instance    = component_instance
        self.point_to_xy           = point_to_xy
        self.attachment_point_vecs = [attachment_point_vec]
        self.svg_id                = svg_id
        self.svg_markup            = svg_markup
        self.widget_id             = widget_id
        self.xy_offset             = xy_offset # for multi component layouts

    #
    # xy() - entity position
    #
    def xy(self):
        return self.point_to_xy
    
    #
    # xyOffset() - xy offset
    #
    def xyOffset(self, xy_offset=None):
        if xy_offset is not None:
            self.xy_offset = xy_offset
        return self.xy_offset
    
    #
    # attachmentPointVecs() - list of attachment point vectors
    #
    def attachmentPointVecs(self):
        return self.attachment_point_vecs
    
    #
    # addAttachmentPointVec() - add a new attachment point to this entity
    #
    def addAttachmentPointVec(self, attachment_point_vec):
        self.attachment_point_vecs.append(attachment_point_vec)

    #
    # svgId() - svg id of the entity within the markup
    #
    def svgId(self):
        return self.svg_id
    
    #
    # svg() - unadorned svg markup
    #
    def svg(self):
        return self.svg_markup
    
    #
    # widgetId() - containing widget id
    #
    def widgetId(self):
        return self.widget_id