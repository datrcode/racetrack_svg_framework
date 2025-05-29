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
import unittest

from rtsvg import *

class Testrt_annotations_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()

    def test_addToTag(self):
        _tests_ = [
            ('',                                None,              ''),
            (None,                              '',                ''),
            (None,                              None,              ''),
            ('color=blue',                      None,              'color=blue'),
            ('',                                'color=blue',      'color=blue'),
            ('color=blue',                      'color=yellow',    'color=yellow'),
            ('color=blue|color=yellow',         'color=red',       'color=red'),
            ('color=blue|color=yellow',         'color=blue',      'color=blue'),
            ('color=blue|color=yellow',         'color=yellow',    'color=yellow'),
            ('color=blue|color=yellow',         'color=green',     'color=green'),
            ('color=blue|color=yellow|invalid', 'color=green',     'color=green|invalid'),
            ('looked_at|valid|up',              'color=green',     'color=green|looked_at|up|valid'),
            ('up|down',                         None,              'down|up'),
            ('up|down',                         '',                'down|up'),
            ('up|down',                         'color=orange',    'color=orange|down|up'),
            ('||||',                            None,              ''),
            ('|||||',                           '',                ''),
            ('||x|||',                          'y',               'x|y'),
            ('||b||||||||c',                    'a',               'a|b|c'),
            ('b||b||||||||c',                   'a',               'a|b|c'),
            ('b|b|b|b||b|||||c',                'b',               'b|c'),
        ]
        for _test_ in _tests_: assert self.rt_self.__addToTag__(_test_[0], _test_[1]) == _test_[2]

    def test_tagNormalizer(self):
        _tests_ = [
            (None,                   ''),
            ('',                     ''),
            ('|',                    ''),
            ('||||||||||',           ''),
            ('|||||up||||||',        'up'),
            ('down',                 'down'),
            ('down|up',              'down|up'),
            ('down|up|down',         'down|up'),
            ('up|down',              'down|up'),
            ('up|up|down',           'down|up'),
            ('down|up||||||',        'down|up'),
            ('down|||up||||||',      'down|up'),
            ('down||||up',           'down|up'),
            ('color=red|color=blue', 'color=blue'), # first type/value found wins (by alphabetical order)
        ]
        for _test_ in _tests_: assert self.rt_self.__tagNormalizer__(_test_[0]) == _test_[1]
    
    def test_removeFromTag(self):
        _tests_ = [
            ('',                                    'mytag',      ''),
            ('mytag',                               'mytag',      ''),
            ('notmytag',                            'mytag',      'notmytag'),
            ('notmytag|mytag|mytag|mytag',          'mytag',      'notmytag'),
            ('color=red|color=blue',                'color=red',  'color=blue'),
            ('color=red|color=blue',                'color=blue', 'color=red'),
            ('color=red|color=blue',                'color',      ''),
            ('color=red|color=blue|mytag',          'color',      'mytag'),
            ('notmytag|color=red|color=blue|mytag', 'color',      'mytag|notmytag'),
        ]
        for _test_ in _tests_: assert self.rt_self.__removeFromTag__(_test_[0], _test_[1]) == _test_[2]


