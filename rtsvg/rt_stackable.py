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

__name__ = 'rt_stackable'

class RTStackable(object):
    def __init__(self):
        pass
    def popStack(self, callers=None):
        pass
    def setStackPostion(index, callers=None):
        pass
    def pushStack(self, df, callers=None):
        pass

class RTSelectable(object):
    def __init__(self):
        pass
    def setSelectedEntitiesAndNotifyOthers(self, _set_, callers=None):
        pass
