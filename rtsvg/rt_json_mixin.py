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
import numpy as np
import json

__name__ = 'rt_json_mixin'

#
# JSON Utilities Mixin
#
class RTJSONMixin(object):
    #
    # jsonRepr() - create an RTJSON object from a JSON reference
    #
    def jsonRepr(self, json_ref):
        if type(json_ref) == str:  json_ref = json.loads(json_ref)
        return self.RTJSON(self, json_ref)

    #
    # RTJSON() - class to parse JSON reference
    #
    class RTJSON(object):
        #
        # __init__()
        #
        def __init__(self, rt_self, _json_):
            self.rt_self    = rt_self
            self.fmto_stars = {}
            self.fmto_abs   = {}
            self.absEOL     = {}
            self.starEOL    = {}
            if type(_json_) == list:
                for i in range(len(_json_)):
                    self.__recursiveParseJSON__('$', '$', _json_[i])
            else:
                self.__recursiveParseJSON__('$', '$', _json_)

        #
        # __updateFMTO__()
        #
        def __updateFMTO__(self, _parent_, _child_, _abs_):
            if _abs_:
                if _parent_ not in self.fmto_abs: self.fmto_abs[_parent_] = []
                self.fmto_abs[_parent_].append(_child_)
            else:
                if _parent_ not in self.fmto_stars: self.fmto_stars[_parent_] = set()
                self.fmto_stars[_parent_].add(_child_)

        #
        # __updateEOL__() - update end of the line (actual leaf values of the JSON tree structure)
        #
        def __updateEOL__(self, _path_abs_, _path_star_, _value_):
            self.absEOL[_path_abs_] = _value_
            if _path_star_ not in self.starEOL: self.starEOL[_path_star_] = []
            self.starEOL[_path_star_].append(_value_)

        #
        # __recursiveParseJSON__()
        #
        def __recursiveParseJSON__(self, _path_abs_, _path_star_, _json_):
            if type(_json_) == dict:
                for k in _json_.keys():
                    _path_abs_child_, _path_star_child_  = _path_abs_+'.'+k, _path_star_+'.'+k
                    self.__updateFMTO__(_path_star_, _path_star_child_, False), self.__updateFMTO__(_path_abs_, _path_abs_child_, True)
                    self.__recursiveParseJSON__(_path_abs_child_, _path_star_child_, _json_[k])
            elif type(_json_) == list:
                _path_star_child_ = _path_star_+'[*]'
                self.__updateFMTO__(_path_star_, _path_star_child_, False)
                for i in range(len(_json_)):
                    _path_abs_child_  = _path_abs_+'['+str(i)+']'
                    self.__updateFMTO__(_path_abs_, _path_abs_child_, True)
                    self.__recursiveParseJSON__(_path_abs_child_, _path_star_child_, _json_[i])
            elif type(_json_) == str:
                self.__updateFMTO__(_path_star_, _path_star_+'-<str>', False), self.__updateFMTO__(_path_abs_, _path_abs_+f'-<str> {_json_}', True)
                self.__updateEOL__(_path_abs_, _path_star_, _json_)
            elif type(_json_) == int:
                self.__updateFMTO__(_path_star_, _path_star_+'-<int>', False), self.__updateFMTO__(_path_abs_, _path_abs_+f'-<int> {_json_}', True)
                self.__updateEOL__(_path_abs_, _path_star_, _json_)
            elif type(_json_) == float:
                self.__updateFMTO__(_path_star_, _path_star_+'-<float>', False), self.__updateFMTO__(_path_abs_, _path_abs_+f'-<float> {_json_}', True)
                self.__updateEOL__(_path_abs_, _path_star_, _json_)
            elif type(_json_) == bool:
                self.__updateFMTO__(_path_star_, _path_star_+'-<bool>', False), self.__updateFMTO__(_path_abs_, _path_abs_+f'-<bool> {_json_}', True)
                self.__updateEOL__(_path_abs_, _path_star_, _json_)
            elif _json_ is None:
                self.__updateFMTO__(_path_star_, _path_star_+'-[NONE]', False), self.__updateFMTO__(_path_abs_, _path_abs_+'-[NONE]', True)
                self.__updateEOL__(_path_abs_, _path_star_, _json_)
            else:
                raise Exception(f'Unknown type ("{type(_json_)}") for ("{_json_}") encountered in parseJSON()')

        #
        # __nodeLabels__()
        #
        def __nodeLabels__(self, _df_):
            _labels_ = {}
            _nodes_  = set(_df_['fm']) | set(_df_['to'])
            for _node_ in _nodes_:
                if   _node_.endswith('[*]'):    _labels_[_node_] = '[*]'
                elif _node_.endswith('<str>')   or \
                    _node_.endswith('<int>')    or \
                    _node_.endswith('<float>')  or \
                    _node_.endswith('<bool>'): _labels_[_node_]  = _node_[_node_.rindex('<'):]
                elif _node_.endswith('[NONE]'): _labels_[_node_] = 'None::None'
                elif '-<str> '   in _node_:     _labels_[_node_] = _node_[_node_.rindex('-<str> ')   +7:]
                elif '-<int> '   in _node_:     _labels_[_node_] = _node_[_node_.rindex('-<int> ')   +7:]
                elif '-<float> ' in _node_:     _labels_[_node_] = _node_[_node_.rindex('-<float> ') +9:]
                elif '-<bool> '  in _node_:     _labels_[_node_] = _node_[_node_.rindex('-<bool> ')  +8:]
                elif '.'         in _node_:     _labels_[_node_] = _node_[_node_.rindex('.')+1:]
                elif _node_ == '$':             _labels_[_node_] = 'ROOT'
            return _labels_
        #
        # absolutePathGraphDataFrame()
        #
        def absolutePathGraphDataFrame(self):
            _df_     = self.rt_self.graphDictToDataFrame(self.fmto_abs)
            _labels_ = self.__nodeLabels__(_df_)
            return _df_, [('fm','to')], _labels_

        #
        # starGraphDataFrame()
        #
        def starPathGraphDataFrame(self):
            _df_     = self.rt_self.graphDictToDataFrame(self.fmto_stars)
            _labels_ = self.__nodeLabels__(_df_)
            return _df_, [('fm','to')], _labels_

