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

__name__ = 'rt_art_mixin'

#
# Art Mixin
# - Icons & Other Types Of Art
#
class RTArtMixin(object):
    #
    # iconLinkNode()
    #
    def iconlinkNode(self):
        return """<svg width="200" height="200" viewbox="-10 -10 120 120" xmlns="http://www.w3.org/2000/svg">
                  <path d="M 10 0 L  90   0  C   95   0   100   5   100   10 
                                  L 100  90  C  100  95    95 100    90  100 
                                  L  10 100  C    5 100    0  95     0   90
                                  L  0  10   C    0   5    5   0     10   0" fill="none" stroke="#000000" stroke-width="2" />
                  <line x1="40"  y1="0"    x2="50" y2="30" stroke="#000000" stroke-width="3"/>
                  <line x1="70"  y1="0"    x2="50" y2="30" stroke="#000000" stroke-width="3"/>
                  <line x1="0"   y1="90"   x2="30" y2="70" stroke="#000000" stroke-width="3"/>
                  <line x1="0"   y1="70"   x2="30" y2="70" stroke="#000000" stroke-width="3"/>
                  <line x1="0"   y1="50"   x2="30" y2="70" stroke="#000000" stroke-width="3"/>
                  <line x1="100" y1="60"   x2="70" y2="70" stroke="#000000" stroke-width="3"/>
                  <line x1="80"  y1="100"  x2="70" y2="70" stroke="#000000" stroke-width="3"/>
                  <line x1="30"  y1="70"   x2="50" y2="30" stroke="#000000" stroke-width="3"/>
                  <line x1="70"  y1="70"   x2="50" y2="30" stroke="#000000" stroke-width="3"/>
                  <line x1="70"  y1="70"   x2="30" y2="70" stroke="#000000" stroke-width="3"/>
                  <circle cx="50" cy="30" r="10" stroke="#000000" stroke-width="3" fill="#ffffff" />
                  <circle cx="30" cy="70" r="10" stroke="#000000" stroke-width="3" fill="#ffffff" />
                  <circle cx="70" cy="70" r="10" stroke="#000000" stroke-width="3" fill="#ffffff" />
                  </svg>"""

    #
    # iconSetCurrentAsRoot()
    #
    def iconSetCurrentAsRoot(self):
        return """<svg width="200" height="200" viewbox="-10 -10 120 120" xmlns="http://www.w3.org/2000/svg">
                  <path d="M 10 0 L  90   0  C   95   0   100   5   100   10 
                                  L 100  90  C  100  95    95 100    90  100 
                                  L  10 100  C    5 100    0  95     0   90
                                  L  0  10   C    0   5    5   0     10   0" fill="none" stroke="#000000" stroke-width="2" />
                  <rect x="10" y="40" width="20" height="20" stroke="#000000" stroke-width="3" fill="#808080" />
                  <rect x="40" y="40" width="20" height="20" stroke="#000000" stroke-width="3" fill="#808080" />
                  <rect x="70" y="40" width="20" height="20" stroke="#000000" stroke-width="3" fill="none" />
                  <path d="M 60 35 C 50 10 20 10 20 35 L 12 20 L 20 35 L 35 25" stroke="#000000" stroke-width="5" fill="none" />
                  </svg>"""

    #
    # iconCurrentMinusRoot()
    #
    def iconCurrentMinusRoot(self):
        return """<svg width="200" height="200" viewbox="-10 -10 120 120" xmlns="http://www.w3.org/2000/svg">
                  <path d="M 10 0 L  90   0  C   95   0   100   5   100   10 
                                  L 100  90  C  100  95    95 100    90  100 
                                  L  10 100  C    5 100    0  95     0   90
                                  L  0  10   C    0   5    5   0     10   0" fill="none" stroke="#000000" stroke-width="2" />
                  <path d="M 10 40 L 30 40 L 30 50 L 20 50 L 20 60 L 10 60 Z" stroke="#000000" stroke-width="3" fill="#a0a0a0" />
                  <rect x="40" y="40" width="20" height="20" stroke="#ff0000" stroke-width="3" fill="#ffa0a0" />
                  <rect x="20" y="50" width="11" height="11" stroke="#ff0000" stroke-width="2" fill="#ffa0a0">
                      <animate attributeName="x" values="20;24;20" dur="4s" repeatCount="indefinite" />
                      <animate attributeName="y" values="50;54;50" dur="4s" repeatCount="indefinite" />
                  </rect>
                  <rect x="70" y="40" width="20" height="20" stroke="#000000" stroke-width="3" fill="none" />
                  <path d="M 60 35 C 50 10 20 10 20 35 L 12 20 L 20 35 L 35 25" stroke="#ff0000" stroke-width="5" fill="none" />
                  </svg>"""
