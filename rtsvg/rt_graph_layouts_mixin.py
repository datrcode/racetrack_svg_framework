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

import heapq

import networkx as nx
import numpy as np

import squarify # treemaps
import random

from math import sqrt, pi, cos, sin, ceil, floor, inf, atan2
from dataclasses import dataclass

from multiprocessing import Pool

__name__ = 'rt_graph_layouts_mixin'

#
# Graph Layouts Methods
#
class RTGraphLayoutsMixin(object):
    #
    # positionExtents()
    # - extents of all the nodes in the positions dictionary
    # - will add x or y space if there's only a single x or y coordinate
    #
    def positionExtents(self,
                        pos,            # pos['node'] = [x, y]
                        _graph = None): # if specified, will limit calculation to just the nodes in this graph
        x0 = y0 = x1 = y1 = None

        if _graph is not None:
            from_structure = set(_graph.nodes())
            in_pos         = set(pos.keys())
            if len(in_pos & from_structure) != len(from_structure):
                print(f'{in_pos=}\n{from_structure=}')
                raise Exception(f'Missing keys in position dictionary | AND={len(in_pos & from_structure)} SUBG={len(from_structure)}')
        else:  from_structure = set(pos.keys())

        for _node in from_structure:
            x = pos[_node][0]
            y = pos[_node][1]
            x0 = x if x0 is None else min(x0, x)
            y0 = y if y0 is None else min(y0, y)
            x1 = x if x1 is None else max(x1, x)
            y1 = y if y1 is None else max(y1, y)
        if x0 == x1:
            x0, x1 = x0 - 0.5, x1 + 0.5
        if y0 == y1:
            y0, y1 = y0 - 0.5, y1 + 0.5
        return x0,y0,x1,y1

    #
    # calculateLevelSet()
    #
    def calculateLevelSet(self,
                          pos,                   # dictionary of nodes to array of positions -- needs to be two (or more)
                          bounds_percent = .05,  # inset the graph into the view by this percent... so that the nodes aren't right at the edges 
                          w              = 256,  # view width to use for level set
                          h              = 256): # view height to use for level set
        # Determine the min and max positions
        x0,y0,x1,y1 = self.positionExtents(pos)

        # Provide border space
        x_inc = (x1-x0)*bounds_percent
        x0 -= x_inc
        x1 += x_inc
        y_inc = (y1-y0)*bounds_percent
        y0 -= y_inc
        y1 += y_inc

        # Translation lambdas
        xT = lambda x: int(w*(x-x0)/(x1-x0))
        yT = lambda y: int(h*(y-y0)/(y1-y0))

        # Allocate the level set
        node_info  = [[None for x in range(w)] for y in range(h)] # node that found the pixel
        found_time = [[None for x in range(w)] for y in range(h)] # when node was found

        # Distance lambda function
        dist = lambda _x0,_y0,_x1,_y1: sqrt((_x0-_x1)*(_x0-_x1)+(_y0-_y1)*(_y0-_y1))

        # Initialize the level set with the node positions
        _node_pos = {}
        _heap     = []
        for _node in pos.keys():
            xi = xT(pos[_node][0])
            yi = yT(pos[_node][1])
            _node_pos[_node] = (xi,yi)
            node_info [yi][xi] = _node
            found_time[yi][xi] = 0
            for dx in range(-1,2):
                for dy in range(-1,2):
                    xn = xi + dx
                    yn = yi + dy
                    if (dx == 0 and dy == 0) or xn < 0  or yn < 0 or xn >= w or yn >= h:
                        continue
                    else:
                        t  = dist(xi,yi,xn,yn)
                        heapq.heappush(_heap,(t, xn, yn, _node))

        # Go through the heap
        while len(_heap) > 0:
            t,xi,yi,_node = heapq.heappop(_heap)
            if found_time[yi][xi] is None or found_time[yi][xi] > t:
                node_info [yi][xi] = _node
                found_time[yi][xi] = t
                for dx in range(-1,2):
                    for dy in range(-1,2):
                        xn = xi + dx
                        yn = yi + dy
                        if (dx == 0 and dy == 0) or xn < 0  or yn < 0 or xn >= w or yn >= h:
                            continue
                        else:
                            xp,yp = _node_pos[_node]
                            t  = dist(xp,yp,xn,yn)
                            if found_time[yn][xn] is None or \
                               found_time[yn][xn] > t:
                                heapq.heappush(_heap,(t, xn, yn, _node))

        return node_info,found_time


    #
    # levelSetSVG()
    # - create a level set representation in svg format
    #
    def levelSetSVG(self,
                    node_info,
                    found_time):
        # Determine width and height of the levelset
        h = len(node_info)
        w = len(node_info[0])

        # Determine the maximum value
        max_t = 1
        for yi in range(0,h):
            for xi in range(0,w):
                if found_time[yi][xi] is not None and found_time[yi][xi] > max_t:
                    max_t = found_time[yi][xi]

        svg  = f'<svg x="0" y="0" width="{2*w}" height="{h}">'
        for yi in range(0,h):
            for xi in range(0,w):
                if found_time[yi][xi] is not None:
                    _co = self.co_mgr.spectrum(found_time[yi][xi],0,max_t)
                    svg += f'<rect x="{xi}" y="{yi}" width="1" height="1" fill="{_co}" stroke="none" stroke-opacity="0.0" />'
                    _co = self.co_mgr.getColor(node_info[yi][xi])
                    svg += f'<rect x="{xi+w}" y="{yi}" width="1" height="1" fill="{_co}" stroke="none" stroke-opacity="0.0" />'

        svg += '</svg>'
        return svg
        
    #
    # adjustNodePositionsBasedOnLevelSet()
    # ... doesn't produce good results... does fill in the whitespace but without regards to edges
    #
    def adjustNodePositionsBasedOnLevelSet(self,
                                           node_info,
                                           pos):
        # Determine width and height of the levelset
        h = len(node_info)
        w = len(node_info[0])

        # Find the center of mass and place the node there
        xsum    = {}
        ysum    = {}
        samples = {}
        for yi in range(0,h):
            for xi in range(0,w):
                if node_info[yi][xi] is not None:
                    _node = node_info[yi][xi]
                    if _node not in xsum.keys():
                        xsum[_node] = 0
                        ysum[_node] = 0
                        samples[_node] = 0
                    xsum[_node] += xi
                    ysum[_node] += yi
                    samples[_node] += 1

        new_pos = {}
        for _node in samples.keys():
            new_pos[_node] = [xsum[_node]/samples[_node], ysum[_node]/samples[_node]]

        not_set = set(pos.keys()) - set(new_pos.keys())
        for x in not_set:
            new_pos[x] = pos[x]

        return new_pos

    #
    # __highDegreeNodes__() - returns higher degree nodes.. could've done better...
    #
    def __highDegreeNodes__(self, _graph):
        _nodes   = _graph.nodes()
        _degrees = []
        for _node in _nodes: _degrees.append(_graph.degree(_node))
        _degrees.sort()
        _degrees.reverse()
        top_ten = _degrees[:int(len(_degrees)*0.1)]
        return top_ten

    #
    # rectangularArrangement() - arrange a list of nodes in a rectangular shape.
    # - bounds = (x0,y0,x1,y1) where x0 < x1 and y0 < y1
    #
    def rectangularArrangement(self, g, nodes, pos=None, bounds=(0,0,1,1)):
        x0, y0, x1, y1 = bounds
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0
        dx, dy = float(x1-x0), float(y1-y0)
        if type(nodes) is not list: nodes = list(nodes)
        if pos is None: pos = {}
        if   len(nodes) == 1:
            pos[nodes[0]] = (x0 + dx/2.0, y0 + dy/2.0)
        elif len(nodes) == 2:
            pos[nodes[0]], pos[nodes[1]] = (x0, y0+dy/2), (x1, y0+dy/2.0)
        elif len(nodes) == 3:
            pos[nodes[0]], pos[nodes[1]], pos[nodes[2]] = (x0, y0), (x0 + dx/2.0, y1), (x1, y0)
        elif len(nodes) == 4 or len(nodes) == 5:
            pos[nodes[0]], pos[nodes[1]] = (x0,y0),(x1,y0)
            pos[nodes[2]], pos[nodes[3]] = (x0,y1),(x1,y1)
            if len(nodes) == 5: pos[nodes[4]] = ((x0+x1)/2.0, (y0+y1)/2.0)
        else:
            if x0 >= x1: x1 = x0 + 1.0
            if y0 >= y1: y1 = y0 + 1.0
            dx, dy = x1 - x0, y1 - y0
            n = ceil(sqrt(len(nodes)))
            if (dx/dy) > 1.5 or (dy/dx) > 1.5: # rectangular
                closest_d = inf
                for i in range(1,n+1):
                    other = len(nodes)/i
                    ratio = other/i
                    d     = abs(ratio - dx/dy)
                    if d < closest_d:
                        max_x_i = i     if (i > other) else other
                        max_y_i = other if (i > other) else i
                    else:
                        max_x_i = other if (i > other) else i
                        max_y_i = i     if (i > other) else other
            else:                              # roughly square
                max_x_i = max_y_i = n

            _sorter_  = []
            _degrees_ = g.degree(nodes)
            for node in nodes: 
                _degrees_ = g.degree(node)
                if type(_degrees_) == int: _sorter_.append((_degrees_,      node))
                else:                      _sorter_.append((len(_degrees_), node))
            _sorter_ = sorted(_sorter_, reverse=True)

            x_i, y_i = 0, 0
            for i in range(len(nodes)):
                _x_ = x0 + x_i * (dx/max_x_i)
                _y_ = y0 + y_i * (dy/max_y_i)
                pos[_sorter_[i][1]] = (_x_,_y_)
                x_i += 1
                if x_i >= max_x_i:
                    y_i += 1
                    x_i  = 0

        return pos

    #
    # sunflowerSeedArrangement() - arrange a list of nodes in a sunflower arrangement
    #
    def sunflowerSeedArrangement(self, g, nodes, pos=None, xy=None, r_max=1.0):
        if type(nodes) is not list: nodes = list(nodes)
        if xy is None: xy = (0,0)
        n = len(nodes)

        # place highest degree nodes in the center
        _sorter_  = []
        _degrees_ = g.degree(nodes)
        for node in nodes: 
            _degrees_ = g.degree(node)
            if type(_degrees_) == int: _sorter_.append((_degrees_,      node))
            else:                      _sorter_.append((len(_degrees_), node))
        _sorter_ = sorted(_sorter_, reverse=True)

        if pos is None:  pos = {}
        r_max_formula = np.sqrt(n)
        _golden_ratio_ = (1 + np.sqrt(5)) / 2
        for i in range(n):
            _angle_  = i * 2 * np.pi / _golden_ratio_
            _radius_ = r_max * np.sqrt(i) / r_max_formula
            pos[_sorter_[i][1]] = (xy[0] + _radius_ * np.cos(_angle_), 
                                   xy[1] + _radius_ * np.sin(_angle_))
        return pos


    #
    # linearOptimizedArrangement() - attempt to place the nodes in their best positions in a line
    # - _segment_ = [(x0,y0),(x1,y1)]
    #
    def linearOptimizedArrangement(self, g, nodes, pos, segment=((0.0, 0.0), (1.0, 1.0))):
        if len(nodes) == 1: return {nodes[0]:((segment[0][0]+segment[1][0])/2.0, (segment[0][1]+segment[1][1])/2.0)}
        adj_pos, as_set = {}, set(nodes)
        # Break the nodes into externally (any) connected and internally (only) connected
        _externals_, _internals_ = set(), set()
        for _node_ in nodes:
            for _nbor_ in g.neighbors(_node_):
                if _nbor_ not in as_set: 
                    _externals_.add(_node_)  # any external connection, makes it external
                    break
            if _node_ not in _externals_: _internals_.add(_node_) # if it wasn't added in the loop, it's an internal

        # Create blanks for positions on the line
        dx, dy = segment[1][0] - segment[0][0], segment[1][1] - segment[0][1]
        _filled_with_, _locations_ = [], []
        for i in range(len(nodes)): 
            _filled_with_.append(None)
            perc = i/float(len(nodes)-1)
            _locations_.append((segment[0][0] + dx*perc, segment[0][1] + dy*perc))

        def placeNodeIntoClosestSlot(node_to_place, nodes_xy=None):
            _closest_ = 0
            if nodes_xy is not None:
                _closest_pt_ = self.closestPointOnSegment(segment, nodes_xy)[1]
                if    _closest_pt_ == segment[0]: _closest_ = 0
                elif  _closest_pt_ == segment[1]: _closest_ = len(_locations_)-1
                else:
                    _closest_d_ = sqrt((_closest_pt_[0]-_locations_[0][0])**2 + (_closest_pt_[1]-_locations_[0][1])**2)
                    for i in range(1, len(_locations_)):
                        _d_ = sqrt((_closest_pt_[0]-_locations_[i][0])**2 + (_closest_pt_[1]-_locations_[i][1])**2)
                        if _d_ < _closest_d_: _closest_, _closest_d_ = i, _d_

            if _filled_with_[_closest_] is None: 
                _filled_with_[_closest_] = node_to_place
                adj_pos[node_to_place]   = _locations_[_closest_]
            else:
                for j in range(1, len(_locations_)):
                    up = _closest_+j
                    dn = _closest_-j
                    if   up < len(_locations_) and _filled_with_[up] is None:
                        _filled_with_[up]        = node_to_place
                        adj_pos[node_to_place]   = _locations_[up]
                        break
                    elif dn >= 0 and _filled_with_[dn] is None:
                        _filled_with_[dn]        = node_to_place
                        adj_pos[node_to_place]   = _locations_[dn]
                        break

        # Place the external nodes -- start with the highest degree node
        _sorter_ = []
        for _node_ in _externals_: _sorter_.append((g.degree(_node_), _node_))
        _sorter_ = sorted(_sorter_, reverse=True)
        for _tuple_ in _sorter_:
            _node_ = _tuple_[1]
            _x_sum_, _y_sum_, _samples_ = 0.0, 0.0, 0
            for _nbor_ in g.neighbors(_node_):
                if _nbor_ in as_set: continue
                _x_sum_   += pos[_nbor_][0]
                _y_sum_   += pos[_nbor_][1]
                _samples_ += 1
            _x_, _y_ = _x_sum_ / _samples_, _y_sum_ / _samples_
            placeNodeIntoClosestSlot(_node_, (_x_, _y_))
        
        # Place the rest of the nodes
        for _node_ in _internals_: placeNodeIntoClosestSlot(_node_)

        return adj_pos

    #
    # circularOptimizedArrangement() - attempt to place the nodes in their best positions around a circle
    # - returns a dictionary of the nodes that were adjusted -- should be equal to the nodes passed in
    #
    def circularOptimizedArrangement(self, g, nodes, pos, xy=(0.0,0.0), r=1.0):
        adj_pos, as_set  = {}, set(nodes)

        # Break the nodes into externally (any) connected and internally (only) connected
        _externals_, _internals_ = set(), set()
        for _node_ in nodes:
            for _nbor_ in g.neighbors(_node_):
                if _nbor_ not in as_set: 
                    _externals_.add(_node_)  # any external connection, makes it external
                    break
            if _node_ not in _externals_: _internals_.add(_node_) # if it wasn't added in the loop, it's an internal

        # Create blanks for positions around the circle
        _filled_with_, _angulars_, _angular_locations_ = [], [], []
        for i in range(len(nodes)):
            _angle_            = i * 2 * pi / len(nodes)
            _filled_with_      .append(None)
            _angulars_         .append(_angle_)
            _angular_locations_.append((xy[0] + r * cos(_angle_), xy[1] + r * sin(_angle_)))

        # For a given angle, find the closest available slot to place the node
        def placeNodeIntoClosestSlot(node_to_place, nodes_xy=None):
            _closest_ = 0
            if nodes_xy is None: _closest_ = random.randint(0, len(_angulars_)-1)
            else:
                _closest_, _closest_d_ = 0, 1e9
                for i in range(0, len(_angular_locations_)):
                    dx, dy = nodes_xy[0] - _angular_locations_[i][0], nodes_xy[1] - _angular_locations_[i][1]
                    d      = sqrt(dx*dx + dy*dy)
                    if d < _closest_d_: _closest_, _closest_d_ = i, d
                    
            if _filled_with_[_closest_] is None: 
                _filled_with_[_closest_] = node_to_place
                adj_pos[node_to_place]   = _angular_locations_[_closest_]
            else:
                for j in range(1, len(_angulars_)):
                    up =  (_closest_+j)                  % len(_angulars_)
                    dn = ((_closest_-j)+len(_angulars_)) % len(_angulars_)
                    if _filled_with_[up] is None:
                        _filled_with_[up]        = node_to_place
                        adj_pos[node_to_place]   = _angular_locations_[up]
                        break
                    elif _filled_with_[dn] is None:
                        _filled_with_[dn]        = node_to_place
                        adj_pos[node_to_place]   = _angular_locations_[dn]
                        break

        # Arrange the externally connected nodes first ... put them in the closest slot
        for _node_ in _externals_:
            _x_sum_, _y_sum_, _samples_ = 0.0, 0.0, 0
            for _nbor_ in g.neighbors(_node_):
                if _nbor_ not in _internals_ and _nbor_ not in _externals_:
                    _nbor_xy_   =  pos[_nbor_]
                    _x_sum_     += _nbor_xy_[0]
                    _y_sum_     += _nbor_xy_[1]
                    _samples_   += 1
            _x_, _y_ = _x_sum_ / _samples_, _y_sum_ / _samples_
            placeNodeIntoClosestSlot(_node_, (_x_,_y_))

        # Arrange the internally connected nodes next ... start with the ones connected to the most externals
        _sorter_ = []
        for _node_ in _internals_:
            _externally_connected_ = 0
            for _nbor_ in g.neighbors(_node_):
                if _nbor_ in _externals_: _externally_connected_ += 1
            _sorter_.append((_externally_connected_, _node_))
        _sorter_ = sorted(_sorter_, reverse=True)
        for _tuple_ in _sorter_:
            _node_ = _tuple_[1]
            if _tuple_[0] > 0:
                _x_sum_, _y_sum_, _samples_ = 0.0, 0.0, 0
                for _nbor_ in g.neighbors(_node_):
                    if _nbor_ in _filled_with_:
                        i           =  _filled_with_.index(_nbor_)
                        _x_sum_     += _angular_locations_[i][0]
                        _y_sum_     += _angular_locations_[i][1]
                        _samples_   += 1
                _x_, _y_ = _x_sum_ / _samples_, _y_sum_ / _samples_
                placeNodeIntoClosestSlot(_node_, (_x_,_y_))
            else:
                placeNodeIntoClosestSlot(_node_)

        return adj_pos

    #
    # circularLayout() - from the java version
    #
    def circularLayout(self, g, selection=None, _radius_=100):
        if selection is None: selection = self.__highDegreeNodes__(g)

        pos, center_angle = {}, {}
        for i in range(len(selection)):
            _node = selection[i]
            _angle_ = i*2*pi/len(selection)
            center_angle[_node] = _angle_
            pos[_node] = (_radius_*cos(_angle_),_radius_*sin(_angle_))

        outer_rings   = {}
        _plus_        = _radius_ * 0.2
        _radius_plus_ = _radius_ + _plus_ + _plus_
        for _node in g.nodes():
            if _node not in pos.keys():
                attachments        = set()
                attachments_coords = set()
                for _center in selection:
                    if _center in g[_node]:
                        attachments.add(_center)
                        attachments_coords.add(pos[_center])
                if   len(attachments) == 0:
                    pos[_node] = (0,0)
                elif len(attachments) == 1:
                    center      = attachments.pop()
                    if center not in outer_rings: outer_rings[center] = set()
                    outer_rings[center].add(_node)
                else:
                    x_sum, y_sum = 0, 0
                    for xy in attachments_coords:
                        x_sum += xy[0]
                        y_sum += xy[1]
                    _subangle_  = 2*pi*random.random()
                    _subradius_ = _plus_ * random.random()
                    pos[_node] = (x_sum/len(attachments_coords) + _subradius_*cos(_subangle_), 
                                  y_sum/len(attachments_coords) + _subradius_*sin(_subangle_))

        # Layout the outer rings using a sunflower arrangement
        for center in outer_rings.keys():
            _angle_  = center_angle[center]
            xy       = (_radius_plus_*cos(_angle_), _radius_plus_*sin(_angle_))
            pos      = self.sunflowerSeedArrangement(g, outer_rings[center], pos, xy, _plus_)

        return pos
    
    #
    # Count the nodes in a subtree of a tree
    #
    def __countSubTreeNodes__(self, _graph, _node, _ignore, _child_count):
        # Check the cache
        if _node in _child_count.keys():
            return _child_count[_node] + 1 # children plus this node
        
        # Else recursively count children
        _sum = 0
        for x in _graph[_node]:
            if x == _ignore:
                continue
            _sum += self.__countSubTreeNodes__(_graph, x, _node, _child_count)

        # Cache the value
        if _child_count is not None:
            _child_count[_node] = _sum
        
        # Results are this node plus children
        return _sum+1

    #
    # Count the total number of leaves
    # ... recursive
    #
    def __totalLeaves__(self, _graph, _parent, _node, _leaf_count):
        if len(_graph[_node]) == 1:
            _leaf_count[_node] = 0 # it's a leaf...
            return 1
        else:
            _sum = 0
            for x in _graph[_node]:
                if x != _parent:
                    _sum += self.__totalLeaves__(_graph, _node, x, _leaf_count)
            _leaf_count[_node] = _sum
            return _sum

    #
    # Calculate the depth of the tree
    # ... recursive
    #
    def __treeDepth__(self, _graph, _parent, _node):
        if len(_graph[_node]) == 1:
            return 1
        else:
            _max_depth = 0
            for x in _graph[_node]:
                if x != _parent:
                    _depth = self.__treeDepth__(_graph, _node, x)
                    if _depth > _max_depth:
                        _max_depth = _depth
            return _max_depth + 1

    #
    # dagLeavesOnly() - for a directed acycle graph, return a set of the leaves
    #
    def dagLeavesOnly(self, G):
        leaves = set()
        for node in G.nodes():
            nbor_count = 0
            for nbor in G.neighbors(node):
                nbor_count += 1
            if nbor_count == 1:
                leaves.add(node)
        return leaves

    #
    # hyperTreeLayout()
    # - create a hypertree layout
    #
    def hyperTreeLayout(self,
                        _graph,                        # networkx graph
                        roots                 = None,  # root(s) to use... if not set, will be calculated
                        touch_up_with_springs = False, # touch up the center of the layout with a spring layout
                        bounds_percent        = 0.1):  # for tree map positioning
        # Make sure root is a list
        if type(roots) != list: roots = [roots]

        # Separate graph into connected components
        _graph = nx.to_undirected(_graph)
        S = [_graph.subgraph(c).copy() for c in nx.connected_components(_graph)]

        pos = {}
        _child_count = {}
        # For each connected component...
        for _subgraph in S:
            # Calculate a minimum spanning tree and convert it to an undirected graph
            G = nx.to_undirected(nx.minimum_spanning_tree(_subgraph))

            # Process small graphs separately
            if len(G) <= 4:
                as_list = list(G.nodes())
                if len(G) >= 1: pos[as_list[0]] = (0,0)
                if len(G) >= 2: pos[as_list[1]] = (1,1)
                if len(G) >= 3: pos[as_list[2]] = (1,0)
                if len(G) >= 4: pos[as_list[3]] = (0,1)
                continue

            # Determine the root if not set
            my_root = None
            if roots is not None:
                for possible_root in roots:
                    if possible_root in G:
                        my_root = possible_root
            
            # Only set my root to the best if it wasn't already set...
            if my_root is None:
                f = G.copy()
                while len(f) > 2: # while there are more than 2 nodes, remove all the leaves
                    to_be_removed = [x for  x in f.nodes() if f.degree(x) <= 1]
                    f.remove_nodes_from(to_be_removed)
                my_root = list(f)[0]

            # Count the number of children
            _child_count = {}
            for x in G[my_root]: self.__countSubTreeNodes__(G, x, my_root, _child_count)
            root_children_count = 0
            for x in G[my_root]: root_children_count += 1 + _child_count[x]
            _child_count[my_root] = root_children_count

            # Place root
            _leaf_count = {}
            ht_state = HTState(angle=0.0, angle_inc=2.0*pi/self.__totalLeaves__(G, my_root, my_root, _leaf_count), max_depth=self.__treeDepth__(G,my_root,my_root))

            _R_ = 8.0
            def placeChildren(_parent, _node, _depth):
                # Place Leaves Directly
                if _child_count[_node] == 0:
                    pos[_node] = (_depth * _R_ * cos(ht_state.angle) / ht_state.max_depth,
                                  _depth * _R_ * sin(ht_state.angle) / ht_state.max_depth)
                    ht_state.angle += ht_state.angle_inc
                # Interior Node...
                else:
                    begin_angle = ht_state.angle
                    _heap = []
                    for x in G[_node]:
                        if x != _parent: heapq.heappush(_heap, (1/(_child_count[x]+1), x))
                    while len(_heap) > 0:
                        x = heapq.heappop(_heap)[1]
                        placeChildren(_node, x, _depth+1)
                    end_angle = ht_state.angle
                    half_angle = (begin_angle + end_angle)/2
                    pos[_node] = (_depth * _R_ * cos(half_angle) / ht_state.max_depth, _depth * _R_ * sin(half_angle) / ht_state.max_depth)

            for x in G[my_root]: placeChildren(my_root, my_root, 0)
            
            # Touch up center w/ spring layout
            if touch_up_with_springs:
                dists   = dict(nx.all_pairs_shortest_path_length(G))
                leaves  = self.dagLeavesOnly(G)
                centers = set(G.nodes()) - leaves
                pos = self.springLayout(G, pos, centers, iterations=20, spring_exp=0.1, only_sel_adj=True, dists=dists)
                # pos = self.springLayout(G, pos, leaves,  iterations=200, spring_exp=0.1, only_sel_adj=True, dists=dists)

        # Separate the connected components
        if len(S) > 1: return self.treeMapGraphComponentPlacement(_graph,pos,bounds_percent)
        else:          return pos

    #
    # Place children within the hypertree structure
    #
    def __hyperTreePlaceChildren__(self,
                                   pos,
                                   _graph,
                                   _parent,
                                   _node,
                                   _depth,
                                   ht_state,
                                   cen_x,
                                   cen_y,
                                   _child_count,
                                   _leaf_count):
        _R_ = 8.0
        # Place Leaves Directly
        if _child_count[_node] == 0:
            pos[_node] = (cen_x + _depth * _R_ * cos(ht_state.angle) / ht_state.max_depth,
                          cen_y + _depth * _R_ * sin(ht_state.angle) / ht_state.max_depth)
            ht_state.angle += ht_state.angle_inc
        # Interior Node...
        else:
            begin_angle = ht_state.angle
            _heap = []
            for x in _graph[_node]:
                if x != _parent: heapq.heappush(_heap, (1/(_child_count[x]+1), x))
            while len(_heap) > 0:
                x = heapq.heappop(_heap)[1]
                self.__hyperTreePlaceChildren__(pos, _graph, _node, x, _depth+1, ht_state, cen_x, cen_y, _child_count, _leaf_count)
            end_angle = ht_state.angle
            half_angle = (begin_angle + end_angle)/2
            pos[_node] = (cen_x + _depth * _R_ * cos(half_angle) / ht_state.max_depth, cen_y + _depth * _R_ * sin(half_angle) / ht_state.max_depth)


    #
    # treeMapGraphComponentPlacement()
    # - returns a new position map for the graph
    #
    def treeMapGraphComponentPlacement(self, 
                                       _graph,              # graph to place
                                       pos,                 # original positions
                                       bounds_percent=0.1): # border region for the treemap
        # Separate graph into connected components // make sure there are two or more components
        _graph = nx.to_undirected(_graph)
        S = [_graph.subgraph(c).copy() for c in nx.connected_components(_graph)]
        if len(S) <= 1: return pos
        
        # Order the graphs from largest to smallest
        my_order = []
        for _subgraph in S: my_order.append((len(_subgraph),_subgraph))
        my_order.sort(key=lambda x:x[0], reverse=True)

        # Transpose the sizes into an array
        nodes = []
        for tup in my_order: nodes.append(tup[0])

        # Calculate the treemap via the squarify library
        normalized_sizes   = squarify.normalize_sizes(nodes, 1024, 1024)
        treemap_rectangles = squarify.squarify(normalized_sizes,0,0,1024,1024)
        # padded_rectangles  = squarify.padded_squarify(normalized_sizes,0,0,1024,1024) # supposed to add padding...

        # For each subgraph, place within the square
        new_pos = {}
        for i in range(0,len(my_order)):
            _subgraph = my_order[i][1]
            _rect     = treemap_rectangles[i]

            # Adjust for the border percent
            rx,ry,rdx,rdy = _rect['x'],_rect['y'],_rect['dx'],_rect['dy']
            if bounds_percent > 0.0 and bounds_percent < 1.0:
                xperc,yperc = rdx*bounds_percent,rdy*bounds_percent
                rx += xperc/2
                ry += yperc/2
                rdx -= xperc
                rdy -= yperc

            # Determine the current extents
            x0,y0,x1,y1 = self.positionExtents(pos,_subgraph)

            for _node in _subgraph.nodes():
                x = (pos[_node][0] - x0)/(x1-x0)
                y = (pos[_node][1] - y0)/(y1-y0)
                new_pos[_node] = [x*rdx + rx, y*rdy + ry]

        return new_pos

    #
    # randomLayout()
    # - return a random layout.
    #
    def randomLayout(self,_graph):
        pos = {}
        for x in _graph.nodes():
            pos[x] = [random.random(),random.random()]
        return pos

    #
    # jitterLayout()
    # - add jitter to existing layout
    # - mostly used for debugging to separate combined nodes...
    #
    def jitterLayout(self,pos,amount=0.1):
        new_pos = {}
        for k in pos.keys():
            x = pos[k][0]
            y = pos[k][1]
            new_pos[k] = [x + random.random()*amount - amount/2, y + random.random()*amount - amount/2]
        return new_pos
            
    #
    # shortestPathLayout()
    # - for two or more nodes, create a layout that shows the shortest paths for those nodes
    #
    def shortestPathLayout(self,
                           _graph,                  # networkx graph
                           _nodes,                  # list of nodes for the shortest path
                           use_weight     = None,   # parameter for the networkx shortest_path method
                           use_undirected = True):  # use the undirected version of the _graph
        pos    = {}
        placed = set()
        if use_undirected:
            _ugraph = nx.to_undirected(_graph)
        if len(_nodes) >= 2:           
            # Iterative create the shortest path... 
            # ... then  place those nodes... exclude them from the graph... then repeat
            level = 0.0
            _path = []
            while _path is not None:
                # Find the next shortest path and add that positioning
                try:
                    _path = nx.shortest_path(_ugraph, _nodes[0], _nodes[1],weight=use_weight)
                except:
                    _path = None

                if _path is not None:
                    for i in range(0,len(_path)):
                        _node = _path[i]
                        if _node not in placed:
                            x = i/(len(_path)-1)
                            y = (i%2) * 0.1 + level
                            pos[_node] = [x,y]
                            placed.add(_node)
                    
                    # Unfreeze the graph
                    _ugraph = nx.Graph(_ugraph)
                    for _node in placed:
                        if _node in _ugraph.nodes() and _node != _nodes[0] and _node != _nodes[1]:
                            _ugraph.remove_node(_node)

                level += 0.2                    
            
            # Place the remaining... in some type of fashion
            still_not_placed = set()
            for _node in _graph.nodes():
                if _node not in placed:
                    existing = set(x for x in _graph.neighbors(_node)) & placed
                    if len(existing) > 0:
                        x_sum = 0
                        for _nbor in existing:
                            x_sum += pos[_nbor][0]
                        x = x_sum / len(existing)
                        pos[_node] = [x,level]
                        placed.add(_node)
                    else:
                        still_not_placed.add(_node)
            for _node in still_not_placed:
                pos[_node] = [random.random(), level+0.2]

            return pos
        else:
            print(f'shortestPathLayout() requires two or more nodes')
            return self.randomLayout(_graph) # Until implemented
    
    #
    # springLayout()
    # - modeled after the Yet Another Spring Layout java implementation
    # - probably just a reference implementation...  networkx version works much faster...
    #
    def springLayout(self,
                     _graph,               # networkx graph
                     pos          = None,  # If none, will be randomized... otherwise, positions supplied
                                           # will be used as a starting point
                     selection    = None,  # nodes that will be adjusted / None means all nodes
                     only_sel_adj = False, # for each iteration, only use the selection to adjust node positions
                     iterations   = None,  # number of iterations... None means a heuristic will be used
                     dists        = None,  # distance dictionary... if None, will be calculated
                     use_weights  = False, # if true, uses edge weights based on dijkstra's algorithm
                     spring_exp   = 1.0):  # spring exponent
        # Make graph undirected... and separate graph into connected components
        _graph = nx.to_undirected(_graph)
        S = [_graph.subgraph(c).copy() for c in nx.connected_components(_graph)]

        # For each connected component...
        new_pos = {}
        for _subgraph in S:
            # Compute the distance matrix if it hasn't been done already
            if dists is None:
                if use_weights:
                    dists = dict(nx.all_pairs_dijkstra_path_length(_subgraph))
                else:
                    dists = dict(nx.all_pairs_shortest_path_length(_subgraph))

            # How much to move each node
            mu = 1.0/len(_subgraph)

            # Rule of thumb -- spring layout function of the number of nodes
            if iterations is None:
                iterations = len(_subgraph)

            # Determine which nodes the adjustment will apply to
            if selection is None:
                _nodes = set(_subgraph.nodes())
                _space = set(_subgraph.nodes())
            else:
                _nodes = set(_subgraph.nodes()) & set(selection)
                if only_sel_adj:
                    _space = set(_subgraph.nodes()) & set(selection)
                else:
                    _space = set(_subgraph.nodes())
            
            # Initial placement (or copy from the pos parameter)
            for _node in _subgraph.nodes():
                if pos is None or _node not in pos.keys():
                    new_pos[_node] = [random.random(),random.random()]
                else:
                    new_pos[_node] = pos[_node]
                
            # Iterate
            for i in range(0,iterations):
                # Calculate the node adjustment
                x_adj          = {}
                y_adj          = {}
                overall_stress = 0.0
                for _node in _nodes:
                    sum_dx,sum_dy,sum_stress = 0.0,0.0,0.0
                    for _dest in _space:
                        t  = dists[_node][_dest]
                        dx  = new_pos[_node][0] - new_pos[_dest][0]
                        dy  = new_pos[_node][1] - new_pos[_dest][1]
                        dx2 = dx*dx
                        dy2 = dy*dy
                        d   = sqrt(dx2+dy2)
                        e   = pow(t,spring_exp)
                        if d < 0.001:
                            d = 0.001
                        if e < 0.001:
                            e = 0.001
                        sum_dx += (2*dx*(1.0 - t/d))/e
                        sum_dy += (2*dy*(1.0 - t/d))/e
                        sum_stress += (t-d)*(t-d)
                    x_adj[_node] = -mu * sum_dx
                    y_adj[_node] = -mu * sum_dy
                    overall_stress += sum_stress/len(_space)

                # Apply the node adjustment
                for _node in _nodes:
                    new_pos[_node] = [new_pos[_node][0] + x_adj[_node],
                                      new_pos[_node][1] + y_adj[_node]]
        
        return new_pos

    #
    # barycentricLayout()
    # - place the selected nodes 
    #
    def barycentricLayout(self,
                          _graph,                # networkx graph
                          pos,                   # positions of the non-selection
                          selection,             # nodes that will be adjusted
                          dists = None,          # distance dictionary... if None, will be calculated
                          use_weights = False):  # use the weights on the edges
        # Make graph undirected... and separate graph into connected components
        _graph = nx.to_undirected(_graph)
        S = [_graph.subgraph(c).copy() for c in nx.connected_components(_graph)]

        # For each connected component...
        new_pos = {}
        for _subgraph in S:
            # Compute the distance matrix if it hasn't been done already
            if dists is None:
                if use_weights:
                    dists = dict(nx.all_pairs_dijkstra_path_length(_subgraph))
                else:
                    dists = dict(nx.all_pairs_shortest_path_length(_subgraph))
            
            # Pare down to intersection of selection and this subgraph
            to_adjust = selection & _subgraph.nodes()
            fixed     = _subgraph.nodes() - to_adjust
            for _fix in fixed:
                new_pos[_fix] = pos[_fix]

            # Iterate and place
            for _node in to_adjust:
                xsum,ysum,samples = 0,0,0
                for _fix in fixed:
                    xsum += pos[_fix][0] * dists[_node][_fix]
                    ysum += pos[_fix][1] * dists[_node][_fix]
                    samples += 1
                new_pos[_node] = [xsum/samples, ysum/samples]
        
        # Return the new positions
        return new_pos

    #
    # springLayout()
    # - modeled after the Yet Another Spring Layout java implementation
    # - probably just a reference implementation...
    # - threading doesn't really exist unless you want to copy over the complete memory space...
    #
    def springLayoutThreaded(self,
                     _graph,               # networkx graph
                     pos          = None,  # If none, will be randomized... otherwise, positions supplied
                                           # will be used as a starting point
                     selection    = None,  # nodes that will be adjusted / None means all nodes
                     only_sel_adj = False, # for each iteration, only use the selection to adjust node positions
                     iterations   = None,  # number of iterations... None means a heuristic will be used
                     dists        = None,  # distance dictionary... if None, will be calculated
                     use_weights  = False, # if true, uses edge weights based on dijkstra's algorithm
                     spring_exp   = 1.0):  # spring exponent
        #
        # --- vvv --- STRAIGHT COPY FROM REFERENCE --- vvv ---
        #
        # Make graph undirected... and separate graph into connected components
        _graph = nx.to_undirected(_graph)
        S = [_graph.subgraph(c).copy() for c in nx.connected_components(_graph)]

        # For each connected component...
        new_pos = {}
        for _subgraph in S:
            # Compute the distance matrix if it hasn't been done already
            if dists is None:
                if use_weights:
                    dists = dict(nx.all_pairs_dijkstra_path_length(_subgraph))
                else:
                    dists = dict(nx.all_pairs_shortest_path_length(_subgraph))

            # How much to move each node
            mu = 1.0/len(_subgraph)

            # Rule of thumb -- spring layout function of the number of nodes
            if iterations is None:
                iterations = len(_subgraph)

            # Determine which nodes the adjustment will apply to
            if selection is None:
                _nodes = set(_subgraph.nodes())
                _space = set(_subgraph.nodes())
            else:
                _nodes = set(_subgraph.nodes()) & set(selection)
                if only_sel_adj:
                    _space = set(_subgraph.nodes()) & set(selection)
                else:
                    _space = set(_subgraph.nodes())
            
            # Initial placement (or copy from the pos parameter)
            for _node in _subgraph.nodes():
                if pos is None or _node not in pos.keys():
                    new_pos[_node] = [random.random(),random.random()]
                else:
                    new_pos[_node] = pos[_node]

        #
        # --- ^^^ --- STRAIGHT COPY FROM REFERENCE --- ^^^ ---
        #

            # Multi-threaded spring state/holder
            mt_spring = MTSpring(dists, _space, mu, new_pos, spring_exp)

            # Make the nodes into a list for consistency
            node_list = list(_nodes)

            # Iterate
            for i in range(0,iterations):
                # Calculate the node adjustment
                p = Pool(16)
                with p:
                    adjs = p.map(mt_spring.calculateAdjustment, node_list)

                # Apply the node adjustment
                for i in range(0,len(adjs)):
                    _node = node_list[i]
                    new_pos[_node] = [new_pos[_node][0] + adjs[i][0],
                                      new_pos[_node][1] + adjs[i][1]]

        return new_pos


#
# Multi-threaded spring layout... doesn't really work any better...
# Suspect that the issue is that the cost of creating the processes
# is too high -- i.e., each has to get a copy of the distance matrix
# and the graph...
#
class MTSpring(object):
    #
    # Constructor
    #
    def __init__(self, 
                 dists,        # distance calculation 
                 space,        # space to adjust over
                 mu,           # how much to adjust each node
                 pos,          # current positions of the nodes
                 spring_exp):  # spring exponent 
        self.dists      = dists
        self.space      = space
        self.mu         = mu
        self.pos        = pos
        self.spring_exp = spring_exp
    
    #
    # Calculate the adjustment for a single node
    #
    def calculateAdjustment(self, node):
        sum_dx,sum_dy,sum_stress = 0.0,0.0,0.0
        for _dest in self.space:
            t  = self.dists[node][_dest]
            dx  = self.pos[node][0] - self.pos[_dest][0]
            dy  = self.pos[node][1] - self.pos[_dest][1]
            dx2 = dx*dx
            dy2 = dy*dy
            d   = sqrt(dx2+dy2)
            e   = pow(t,self.spring_exp)
            if d < 0.001:
                d = 0.001
            if e < 0.001:
                e = 0.001
            sum_dx += (2*dx*(1.0 - t/d))/e
            sum_dy += (2*dy*(1.0 - t/d))/e
            sum_stress += (t-d)*(t-d)
        #    
        #      x_adj,     y_adj,     stress
        #
        return -self.mu*sum_dx,-self.mu*sum_dy,sum_stress/len(self.space) 

#
# HyperTree state holder/struct
#
@dataclass
class HTState:
    angle:     float
    angle_inc: float
    max_depth: int


