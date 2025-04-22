import copy
import queue
from math import sqrt, acos, pi

#
# CirclePacker()
# 
# Implementation of the following paper:
#
#@inproceedings{10.1145/1124772.1124851,
#               author = {Wang, Weixin and Wang, Hui and Dai, Guozhong and Wang, Hongan},
#               title = {Visualization of large hierarchical data by circle packing},
#               year = {2006},
#               isbn = {1595933727},
#               publisher = {Association for Computing Machinery},
#               address = {New York, NY, USA},
#               url = {https://doi.org/10.1145/1124772.1124851},
#               doi = {10.1145/1124772.1124851},
#               abstract = {In this paper a novel approach is described for tree visualization using nested circles. 
#                           The brother nodes at the same level are represented by externally tangent circles; 
#                           the tree nodes at different levels are displayed by using 2D nested circles or 3D nested cylinders. 
#                           A new layout algorithm for tree structure is described. It provides a good overview for large data sets. 
#                           It is easy to see all the branches and leaves of the tree. The new method has been applied to the 
#                           visualization of file systems.},
#               booktitle = {Proceedings of the SIGCHI Conference on Human Factors in Computing Systems},
#               pages = {517–520},
#               numpages = {4},
#               keywords = {tree visualization, nested circles, file system, circle packing},
#               location = {Montr\'{e}al, Qu\'{e}bec, Canada},
#               series = {CHI '06} }
#
class CirclePacker(object):
    #
    # __init__()
    #
    def __init__(self, rt_self, circles, epsilon=0.01):
        self.rt_self             = rt_self
        self.circles             = circles

        self.r_min = self.r_max  = self.circles[0][2]
        for c in self.circles: self.r_min, self.r_max = min(self.r_min, c[2]), max(self.r_max, c[2])

        if (self.r_max / self.r_min) > 4.125:
            self.circles = sorted(self.circles, key=lambda x: x[2], reverse=True)

        self.circles_left        = copy.deepcopy(self.circles)
        self.epsilon             = epsilon
        self.packed              = []
        self.fwd                 = {}
        self.bck                 = {}
        self.r_max_so_far        = 0.0 # for optimization
        self.nearest             = queue.PriorityQueue()



        # Pack the first circles
        self.__packFirstCircles__()
        # Capture the maximum radius seen so far (for the optimization implementation)
        for i in range(len(self.packed)): self.r_max_so_far = max(self.r_max_so_far, self.packed[i][2])
        # Create a priority queue to find the circle w/in the chain that is closest to the origin
        for i in range(len(self.packed)):
            if i not in self.fwd: continue
            c = self.packed[i]
            self.nearest.put((c[0]**2 + c[1]**2, i))
        # Pack the circles iteratively
        while len(self.circles_left) > 0: self.__packNextCircle__()
    
    #
    # __packFirstThreeCircles__()
    #
    def __packFirstCircles__(self):
        # Circle 1
        cx0, cy0, r0  = self.circles_left.pop(0)
        cx0 = cy0 = 0.0
        self.packed.append((cx0, cy0, r0))
        self.fwd, self.bck = {0:0}, {0:0}
        if len(self.circles_left) == 0: return

        # Circle 2
        cx1, cy1, r1  = self.circles_left.pop(0)
        cy1           = 0.0
        cx1           = r0 + r1
        self.packed.append((cx1, 0.0, r1))
        self.fwd, self.bck = {0:1, 1:0}, {0:1, 1:0}
        if len(self.circles_left) == 0: return

        # Circle 3
        cx2, cy2, r2  = self.circles_left.pop(0)
        xy0, xy1      = self.rt_self.overlappingCirclesIntersections((cx0,cy0,r0+r2),(cx1,cy1,r1+r2))
        cx2, cy2      = xy0[0], xy0[1]
        self.packed.append((cx2, cy2, r2))
        self.fwd, self.bck = {0:1, 1:2, 2:0}, {1:0, 2:1, 0:2}
        if len(self.circles_left) == 0: return

        # Circle 4
        cx3, cy3, r3  = self.circles_left.pop(0)
        xy0, xy1      = self.rt_self.overlappingCirclesIntersections((cx1,cy1,r1+r3),(cx2,cy2,r2+r3))
        cx3, cy3      = xy0[0], xy0[1]

        if self.rt_self.circlesOverlap((cx0, cy0, r0), (cx3, cy3, r3)):
            cx3, cy3      = xy1[0], xy1[1]
            if self.rt_self.circlesOverlap((cx0, cy0, r0), (cx3, cy3, r3)):
                xy0, xy1      = self.rt_self.overlappingCirclesIntersections((cx0,cy0,r0+r3),(cx2,cy2,r2+r3))
                cx3, cy3      = xy0[0], xy0[1]
                if self.rt_self.circlesOverlap((cx1, cy1, r1), (cx3, cy3, r3)):
                    cx3, cy3      = xy1[0], xy1[1]
                    if self.rt_self.circlesOverlap((cx1, cy1, r1), (cx3, cy3, r3)):
                        xy0, xy1      = self.rt_self.overlappingCirclesIntersections((cx0,cy0,r0+r3),(cx1,cy1,r1+r3))
                        cx3, cy3      = xy0[0], xy0[1]
                        if self.rt_self.circlesOverlap((cx2, cy2, r2), (cx3, cy3, r3)):
                            cx3, cy3      = xy1[0], xy1[1]
                            if self.rt_self.circlesOverlap((cx2, cy2, r2), (cx3, cy3, r3)):
                                raise Exception('__packFirstCircles__() - 6 - should not happen')
                            else:
                                raise Exception('__packFirstCircles__() - 5 - should not happen')
                        else:
                            self.fwd, self.bck = {0:3, 3:1, 1:0}, {3:0, 1:3, 0:1}        
                    else:
                        raise Exception('__packFirstCircles__() - 3 - should not happen')
                else:
                    self.fwd, self.bck = {0:3, 3:2, 2:1, 1:0}, {3:0, 2:3, 1:2, 0:1}
            else:
                self.fwd, self.bck = {0:2, 2:3, 3:1, 1:0}, {2:0, 3:2, 1:3, 0:1}
        else:            
            if   self.packed[0][2] <= self.packed[1][2] and self.packed[0][2] <= self.packed[2][2] and self.packed[0][2] <= r3: # 0 is the smallest
                self.fwd, self.bck = {3:2, 2:1, 1:3}, {2:3, 1:2, 3:1}
            elif self.packed[1][2] <= self.packed[0][2] and self.packed[1][2] <= self.packed[2][2] and self.packed[1][2] <= r3: # 1 is the smallest
                raise Exception('__packFirstCircles__() - 0 - should not happen (01)')
            elif self.packed[2][2] <= self.packed[0][2] and self.packed[2][2] <= self.packed[1][2] and self.packed[2][2] <= r3: # 2 is the smallest
                raise Exception('__packFirstCircles__() - 0 - should not happen (02)')
            else:                                                                                                               # 3 is the smallest
                self.fwd, self.bck = {0:2, 2:1, 1:0}, {2:0, 1:2, 0:1}

        self.packed.append((cx3, cy3, r3))

    #
    # __packNextCircle__() - pack the next circle in this list
    #
    def __packNextCircle__(self):
        def approximateCircleArcLength(cir0, cir1):
            a = b = (sqrt(cir0[0]**2 + cir0[1]**2) + sqrt(cir1[0]**2 + cir1[1]**2)) / 2.0 # average radius
            c     = self.rt_self.segmentLength((cir0, cir1))
            gamma         = acos((a**2 + b**2 - c**2)/(2.0*a*b))
            circumference = 2.0*pi*a
            arc_length    = circumference*gamma/(2.0*pi)
            return arc_length
        # Pick up the next circle
        c = self.circles_left.pop(0)
        # Setup the variables per the paper
        cm_i, cn_i          = self.nearest.queue[0][1], self.fwd[self.nearest.queue[0][1]]
        cm,   cn            = self.packed[cm_i], self.packed[cn_i]
        xy0, xy1            = self.rt_self.overlappingCirclesIntersections((cm[0], cm[1], cm[2] + c[2]), (cn[0], cn[1], cn[2] + c[2]))
        c                   = (xy0[0], xy0[1], c[2])
        # Repeat until the circle is placed
        circle_placed = False
        while circle_placed == False:
            prev, next        = self.bck[cm_i], self.fwd[cn_i]
            seen              = set([cn_i, cm_i])
            overlapped_after  = None
            overlapped_before = None
            while overlapped_after is None and overlapped_before is None and next not in seen and prev not in seen:
                if self.rt_self.circlesOverlap((c[0], c[1], c[2]-self.epsilon), self.packed[next]): overlapped_after  = next
                if self.rt_self.circlesOverlap((c[0], c[1], c[2]-self.epsilon), self.packed[prev]): overlapped_before = prev
                seen.add(next), seen.add(prev)
                next, prev = self.fwd[next], self.bck[prev]
                if 0 not in self.fwd and len(self.packed) > 50: # 0 == first circle packed... so for this to take effect, the first circle needs to be enclosed by others
                    circumference_next = approximateCircleArcLength(self.packed[next], c)
                    circumference_prev = approximateCircleArcLength(self.packed[prev], c)
                    next_far_enough    = circumference_next > 2.0 * (self.r_max_so_far + c[2])
                    prev_far_enough    = circumference_prev > 2.0 * (self.r_max_so_far + c[2])
                    if next_far_enough and prev_far_enough: break
            if   overlapped_after is not None and overlapped_before is not None:
                self.__eraseChain__(self.fwd[overlapped_before], self.bck[overlapped_after])
                while self.nearest.queue[0][1] not in self.fwd.keys(): self.nearest.get() # find the next nearest circle that is still in the chain
                cm_i     = overlapped_before
                cm       = self.packed[cm_i]
                cn_i     = overlapped_after
                cn       = self.packed[cn_i]
                xy0, xy1 = self.rt_self.overlappingCirclesIntersections((cm[0], cm[1], cm[2] + c[2]), (cn[0], cn[1], cn[2] + c[2]))
                c        = (xy0[0], xy0[1], c[2])
            elif overlapped_after  is not None:
                self.__eraseChain__(cn_i, self.bck[overlapped_after])
                while self.nearest.queue[0][1] not in self.fwd.keys(): self.nearest.get() # find the next nearest circle that is still in the chain
                cn_i     = overlapped_after
                cn       = self.packed[cn_i]
                xy0, xy1 = self.rt_self.overlappingCirclesIntersections((cm[0], cm[1], cm[2] + c[2]), (cn[0], cn[1], cn[2] + c[2]))
                c        = (xy0[0], xy0[1], c[2])
            elif overlapped_before is not None:
                self.__eraseChain__(self.fwd[overlapped_before], cm_i)
                while self.nearest.queue[0][1] not in self.fwd.keys(): self.nearest.get() # find the next nearest circle that is still in the chain
                cm_i     = overlapped_before
                cm       = self.packed[cm_i]
                xy0, xy1 = self.rt_self.overlappingCirclesIntersections((cm[0], cm[1], cm[2] + c[2]), (cn[0], cn[1], cn[2] + c[2]))
                c        = (xy0[0], xy0[1], c[2])
            else:
                self.packed.append(c)
                _index_           = len(self.packed) - 1
                self.fwd[cm_i], self.bck[_index_] = _index_, cm_i
                self.bck[cn_i], self.fwd[_index_] = _index_, cn_i
                circle_placed     = True
                self.r_max_so_far = max(self.r_max_so_far, self.packed[-1][2])
                self.nearest.put((c[0]**2 + c[1]**2, _index_))
                while self.nearest.queue[0][1] not in self.fwd.keys(): self.nearest.get() # find the next nearest circle that is still in the chain

    #
    # __eraseChain__() - erase the chain from fm_i to to_i
    # ... inclusive -- fm_i and to_i will be deleted
    # ... at the end, the chain will be reconnected
    #
    def __eraseChain__(self, fm_i, to_i):
        i_start = self.bck[fm_i]
        i_end   = self.fwd[to_i]
        while fm_i != i_end:
            i_next = self.fwd[fm_i]
            del self.fwd[fm_i]
            fm_i   = i_next
        while to_i != i_start:
            i_prev = self.bck[to_i]
            del self.bck[to_i]
            to_i   = i_prev
        self.fwd[i_start], self.bck[i_end] = i_end, i_start
    
    #
    # __validateChains__()
    #
    def __validateChains__(self):
        if len(self.fwd) != len(self.bck):   raise Exception('Chains Are Different Lengths')
        for i in self.fwd.keys():
            if i not in self.bck.keys():     raise Exception('Forward Chain Has Key Not In Backward Chain')
            if self.bck[self.fwd[i]] != i:   raise Exception('Backward Chain Has Key Not In Forward Chain')
        
    #
    # __validateNoOverlaps__()
    #
    def __validateNoOverlaps__(self):
        for i in range(len(self.packed)):
            c0 = self.packed[i]
            for j in range(i+1, len(self.packed)):
                c1 = self.packed[j]
                if self.rt_self.circlesOverlap((c0[0], c0[1], c0[2]), (c1[0], c1[1], c1[2]-self.epsilon)): return False
        return True

    #
    # _repr_svg_() - return an SVG representation
    #
    def _repr_svg_(self):
        w, h    = 250, 250
        _pkd_   = self.packed
        _chn_   = self.fwd
        x0, y0, x1, y1 = _pkd_[0][0] - _pkd_[0][2] - 3, _pkd_[0][1] - _pkd_[0][2] - 3, _pkd_[0][0] + _pkd_[0][2] + 3, _pkd_[0][1] + _pkd_[0][2] + 3
        for i in range(1, len(_pkd_)):
            x0, y0, x1, y1 = min(x0, _pkd_[i][0] - _pkd_[i][2] - 3), min(y0, _pkd_[i][1] - _pkd_[i][2] - 3), max(x1, _pkd_[i][0] + _pkd_[i][2] + 3), max(y1, _pkd_[i][1] + _pkd_[i][2] + 3)
        svg = [f'<svg x="0" y="0" width="{w}" height="{h}" viewBox="{x0} {y0} {x1-x0} {y1-y0}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#ffffff" />')
        svg.append(f'<line x1="0.0"  y1="{y0}" x2="0.0"  y2="{y1}" stroke="#ff0000" stroke-width="0.1" />')
        svg.append(f'<line x1="{x0}" y1="0.0"  x2="{x1}" y2="0.0"  stroke="#ff0000" stroke-width="0.1" />')
        for i in range(len(_pkd_)):
            _c_ = _pkd_[i]
            if len(self.nearest.queue) > 0 and i == self.nearest.queue[0][1]: _color_ = '#ff0000'
            else:                                                             _color_ = '#000000'
            svg.append(f'<circle cx="{_c_[0]}" cy="{_c_[1]}" r="{_c_[2]}" fill="none" stroke="{_color_}" stroke-width="0.2" />')
        _color_ = '#000000'
        if len(_chn_.keys()) > 0: 
            _index_ = list(_chn_.keys())[0]
            for i in range(len(_chn_.keys())+1):
                _index_next_ = _chn_[_index_]
                xy0, xy1 = _pkd_[_index_], _pkd_[_index_next_]
                svg.append(f'<line x1="{xy0[0]}" y1="{xy0[1]}" x2="{xy1[0]}" y2="{xy1[1]}" stroke="{_color_}" stroke-width="0.2" />')
                uv       = self.rt_self.unitVector((xy0, xy1))
                perp     = (-uv[1], uv[0])
                svg.append(f'<line x1="{xy1[0]}" y1="{xy1[1]}" x2="{xy1[0] - 1*uv[0] + 0.5*perp[0]}" y2="{xy1[1] - 1*uv[1] + 0.5*perp[1]}" stroke="{_color_}" stroke-width="0.1" />')
                svg.append(f'<line x1="{xy1[0]}" y1="{xy1[1]}" x2="{xy1[0] - 1*uv[0] - 0.5*perp[0]}" y2="{xy1[1] - 1*uv[1] - 0.5*perp[1]}" stroke="{_color_}" stroke-width="0.1" />')
                _index_  = _index_next_
        svg.append('</svg>')
        return ''.join(svg)



class DEBUG_CirclePacker(object):
    #
    # __init__()
    #
    def __init__(self, rt_self, circles, epsilon=0.01, capture_progression=True, validate_chains=True, check_overlaps_every_append=False):
        self.rt_self             = rt_self
        self.circles             = circles
        # self.circles             = sorted(self.circles, key=lambda c: c[2], reverse=True) # solves the issue w/ mixing too small & too large circles
        self.circles_left        = copy.deepcopy(self.circles)
        self.epsilon             = epsilon
        self.packed              = []
        self.fwd                 = {}
        self.bck                 = {}
        self.debug_str           = ''  # for debug / should be deleted when finalized
        self.progression         = []  # for debug / should be deleted when finalized
        self.before_and_afters   = []  # for debug / should be deleted when finalized
        self.check_overlaps_every_append = check_overlaps_every_append # for debug
        self.overlap_found_at    = None
        self.placing_circle      = None
        self.capture_progression = capture_progression
        self.validate_chains     = validate_chains
        self.r_max_so_far        = 0.0 # for optimization
        self.r_min               = self.circles[0][2]
        self.r_max               = self.circles[0][2]
        for c in self.circles:
            self.r_min = min(self.r_min, c[2])
            self.r_max = max(self.r_max, c[2])
        self.nearest             = queue.PriorityQueue()

    def pack(self): # for debug ... should be rejoined with constructor once finalized
        # Pack the first Three circles
        self.__packFirstCircles__()
        # Capture the maximum radius seen so far (for the optimization implementation)
        for i in range(len(self.packed)): self.r_max_so_far = max(self.r_max_so_far, self.packed[i][2])
        # Create a priority queue to find the circle w/in the chain that is closest to the origin
        for i in range(len(self.packed)):
            if i not in self.fwd: continue
            c = self.packed[i]
            self.nearest.put((c[0]**2 + c[1]**2, i))
        if self.validate_chains: self.__validateChains__()
        if self.capture_progression: self.progression.append(self._repr_svg_())
        # Pack the circles iteratively
        while len(self.circles_left) > 0: self.__packNextCircle__()
    
    #
    # __packFirstThreeCircles__()
    #
    def __packFirstCircles__(self):
        # Circle 1
        cx0, cy0, r0  = self.circles_left.pop(0)
        cx0 = cy0 = 0.0
        self.packed.append((cx0, cy0, r0))
        self.fwd, self.bck = {0:0}, {0:0}
        if len(self.circles_left) == 0: return

        # Circle 2
        cx1, cy1, r1  = self.circles_left.pop(0)
        cy1           = 0.0
        cx1           = r0 + r1
        self.packed.append((cx1, 0.0, r1))
        self.fwd, self.bck = {0:1, 1:0}, {0:1, 1:0}
        if len(self.circles_left) == 0: return

        # Circle 3
        cx2, cy2, r2  = self.circles_left.pop(0)
        xy0, xy1      = self.rt_self.overlappingCirclesIntersections((cx0,cy0,r0+r2),(cx1,cy1,r1+r2))
        cx2, cy2      = xy0[0], xy0[1]
        self.packed.append((cx2, cy2, r2))
        self.fwd, self.bck = {0:1, 1:2, 2:0}, {1:0, 2:1, 0:2}
        if len(self.circles_left) == 0: return

        # Circle 4
        cx3, cy3, r3  = self.circles_left.pop(0)
        xy0, xy1      = self.rt_self.overlappingCirclesIntersections((cx1,cy1,r1+r3),(cx2,cy2,r2+r3))
        cx3, cy3      = xy0[0], xy0[1]

        if self.rt_self.circlesOverlap((cx0, cy0, r0), (cx3, cy3, r3)):
            cx3, cy3      = xy1[0], xy1[1]
            if self.rt_self.circlesOverlap((cx0, cy0, r0), (cx3, cy3, r3)):
                xy0, xy1      = self.rt_self.overlappingCirclesIntersections((cx0,cy0,r0+r3),(cx2,cy2,r2+r3))
                cx3, cy3      = xy0[0], xy0[1]
                if self.rt_self.circlesOverlap((cx1, cy1, r1), (cx3, cy3, r3)):
                    cx3, cy3      = xy1[0], xy1[1]
                    if self.rt_self.circlesOverlap((cx1, cy1, r1), (cx3, cy3, r3)):
                        xy0, xy1      = self.rt_self.overlappingCirclesIntersections((cx0,cy0,r0+r3),(cx1,cy1,r1+r3))
                        cx3, cy3      = xy0[0], xy0[1]
                        if self.rt_self.circlesOverlap((cx2, cy2, r2), (cx3, cy3, r3)):
                            cx3, cy3      = xy1[0], xy1[1]
                            if self.rt_self.circlesOverlap((cx2, cy2, r2), (cx3, cy3, r3)):
                                self.debug_str = '6'
                                raise Exception('__packFirstCircles__() - 6 - should not happen')
                            else:
                                self.debug_str = '5'
                                raise Exception('__packFirstCircles__() - 5 - should not happen')
                        else:
                            self.debug_str = '4'
                            self.fwd, self.bck = {0:3, 3:1, 1:0}, {3:0, 1:3, 0:1}        
                    else:
                        self.debug_str = '3'
                        raise Exception('__packFirstCircles__() - 3 - should not happen')
                else:
                    self.debug_str = '2'
                    self.fwd, self.bck = {0:3, 3:2, 2:1, 1:0}, {3:0, 2:3, 1:2, 0:1}
            else:
                self.debug_str = '1'
                self.fwd, self.bck = {0:2, 2:3, 3:1, 1:0}, {2:0, 3:2, 1:3, 0:1}
        else:            
            if   self.packed[0][2] <= self.packed[1][2] and self.packed[0][2] <= self.packed[2][2] and self.packed[0][2] <= r3: # 0 is the smallest
                self.debug_str = '00' # there are several variations
                self.fwd, self.bck = {3:2, 2:1, 1:3}, {2:3, 1:2, 3:1}
            elif self.packed[1][2] <= self.packed[0][2] and self.packed[1][2] <= self.packed[2][2] and self.packed[1][2] <= r3: # 1 is the smallest
                self.debug_str = '01' # there are several variations
                raise Exception('__packFirstCircles__() - 0 - should not happen (01)')
            elif self.packed[2][2] <= self.packed[0][2] and self.packed[2][2] <= self.packed[1][2] and self.packed[2][2] <= r3: # 2 is the smallest
                self.debug_str = '02' # there are several variations
                raise Exception('__packFirstCircles__() - 0 - should not happen (02)')
            else:                                                                                                               # 3 is the smallest
                self.debug_str = '03' # there are several variations
                self.fwd, self.bck = {0:2, 2:1, 1:0}, {2:0, 1:2, 0:1}

        self.packed.append((cx3, cy3, r3))

    #
    # _repr_svg_() - return an SVG representation
    #
    def _repr_svg_(self):
        w, h    = 250, 250
        _pkd_   = self.packed
        _chn_   = self.fwd
        x0, y0, x1, y1 = _pkd_[0][0] - _pkd_[0][2] - 3, _pkd_[0][1] - _pkd_[0][2] - 3, _pkd_[0][0] + _pkd_[0][2] + 3, _pkd_[0][1] + _pkd_[0][2] + 3
        for i in range(1, len(_pkd_)):
            x0, y0, x1, y1 = min(x0, _pkd_[i][0] - _pkd_[i][2] - 3), min(y0, _pkd_[i][1] - _pkd_[i][2] - 3), max(x1, _pkd_[i][0] + _pkd_[i][2] + 3), max(y1, _pkd_[i][1] + _pkd_[i][2] + 3)
        svg = [f'<svg x="0" y="0" width="{w}" height="{h}" viewBox="{x0} {y0} {x1-x0} {y1-y0}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#ffffff" />')
        svg.append(f'<line x1="0.0"  y1="{y0}" x2="0.0"  y2="{y1}" stroke="#ff0000" stroke-width="0.1" />')
        svg.append(f'<line x1="{x0}" y1="0.0"  x2="{x1}" y2="0.0"  stroke="#ff0000" stroke-width="0.1" />')
        for i in range(len(_pkd_)):
            _c_ = _pkd_[i]
            if len(self.nearest.queue) > 0 and i == self.nearest.queue[0][1]: _color_ = '#ff0000'
            else:                                                             _color_ = '#000000'
            svg.append(f'<circle cx="{_c_[0]}" cy="{_c_[1]}" r="{_c_[2]}" fill="none" stroke="{_color_}" stroke-width="0.2" />')
        _color_ = '#000000'
        if len(_chn_.keys()) > 0: 
            _index_ = list(_chn_.keys())[0]
            for i in range(len(_chn_.keys())+1):
                _index_next_ = _chn_[_index_]
                xy0, xy1 = _pkd_[_index_], _pkd_[_index_next_]
                svg.append(f'<line x1="{xy0[0]}" y1="{xy0[1]}" x2="{xy1[0]}" y2="{xy1[1]}" stroke="{_color_}" stroke-width="0.2" />')
                uv       = self.rt_self.unitVector((xy0, xy1))
                perp     = (-uv[1], uv[0])
                svg.append(f'<line x1="{xy1[0]}" y1="{xy1[1]}" x2="{xy1[0] - 1*uv[0] + 0.5*perp[0]}" y2="{xy1[1] - 1*uv[1] + 0.5*perp[1]}" stroke="{_color_}" stroke-width="0.1" />')
                svg.append(f'<line x1="{xy1[0]}" y1="{xy1[1]}" x2="{xy1[0] - 1*uv[0] - 0.5*perp[0]}" y2="{xy1[1] - 1*uv[1] - 0.5*perp[1]}" stroke="{_color_}" stroke-width="0.1" />')
                _index_  = _index_next_
        if self.debug_str is not None and self.debug_str != '': svg.append(self.rt_self.svgText(self.debug_str, x0, y1))
        if self.placing_circle is not None: 
            svg.append(f'<circle cx="{self.placing_circle[0]}" cy="{self.placing_circle[1]}" r="{self.placing_circle[2]}" fill="none" stroke="#00a000" stroke-width="0.2" stroke-dasharray="0.1 0.2"/>')
            for i in range(len(self.packed)):
                _c_ = self.packed[i]
                if self.rt_self.circlesOverlap(_c_, self.placing_circle):
                    svg.append(f'<line x1="{_c_[0]}" y1="{_c_[1]}" x2="{self.placing_circle[0]}" y2="{self.placing_circle[1]}" stroke="#00a000" stroke-width="0.1" />')
        svg.append('</svg>')
        return ''.join(svg)

    #
    # __packNextCircle__() - pack the next circle in this list
    #
    def __packNextCircle__(self):
        self.debug_str = ''
        def approximateCircleArcLength(cir0, cir1):
            a = b = (sqrt(cir0[0]**2 + cir0[1]**2) + sqrt(cir1[0]**2 + cir1[1]**2)) / 2.0 # average radius
            c     = self.rt_self.segmentLength((cir0, cir1))
            gamma         = acos((a**2 + b**2 - c**2)/(2.0*a*b))
            circumference = 2.0*pi*a
            arc_length    = circumference*gamma/(2.0*pi)
            return arc_length
        # Pick up the next circle
        c = self.circles_left.pop(0)
        # Setup the variables per the paper
        cm_i, cn_i          = self.nearest.queue[0][1], self.fwd[self.nearest.queue[0][1]]
        cm,   cn            = self.packed[cm_i], self.packed[cn_i]
        xy0, xy1            = self.rt_self.overlappingCirclesIntersections((cm[0], cm[1], cm[2] + c[2]), (cn[0], cn[1], cn[2] + c[2]))
        c                   = (xy0[0], xy0[1], c[2])
        self.placing_circle = c
        # Repeat until the circle is placed
        circle_placed = False
        while circle_placed == False:
            if self.capture_progression: self.progression.append(self._repr_svg_())
            prev, next        = self.bck[cm_i], self.fwd[cn_i]
            seen              = set([cn_i, cm_i])
            overlapped_after  = None
            overlapped_before = None
            while overlapped_after is None and overlapped_before is None and next not in seen and prev not in seen:
                if self.rt_self.circlesOverlap((c[0], c[1], c[2]-self.epsilon), self.packed[next]): overlapped_after  = next
                if self.rt_self.circlesOverlap((c[0], c[1], c[2]-self.epsilon), self.packed[prev]): overlapped_before = prev
                seen.add(next), seen.add(prev)
                next, prev = self.fwd[next], self.bck[prev]
                if 0 not in self.fwd and len(self.packed) > 50: # 0 == first circle packed... so for this to take effect, the first circle needs to be enclosed by others
                    circumference_next = approximateCircleArcLength(self.packed[next], c)
                    circumference_prev = approximateCircleArcLength(self.packed[prev], c)
                    next_far_enough    = circumference_next > 2.0 * (self.r_max_so_far + c[2])
                    prev_far_enough    = circumference_prev > 2.0 * (self.r_max_so_far + c[2])
                    if next_far_enough and prev_far_enough: break
            if   overlapped_after is not None and overlapped_before is not None:
                self.debug_str += '@'
                self.before_and_afters.append(self._repr_svg_())
                self.__eraseChain__(self.fwd[overlapped_before], self.bck[overlapped_after])
                while self.nearest.queue[0][1] not in self.fwd.keys(): self.nearest.get() # find the next nearest circle that is still in the chain
                cm_i     = overlapped_before
                cm       = self.packed[cm_i]
                cn_i     = overlapped_after
                cn       = self.packed[cn_i]
                xy0, xy1 = self.rt_self.overlappingCirclesIntersections((cm[0], cm[1], cm[2] + c[2]), (cn[0], cn[1], cn[2] + c[2]))
                c        = (xy0[0], xy0[1], c[2])
                self.placing_circle = c
                if self.capture_progression: self.progression.append(self._repr_svg_())
            elif overlapped_after  is not None:
                self.debug_str += 'a'
                self.__eraseChain__(cn_i, self.bck[overlapped_after])
                while self.nearest.queue[0][1] not in self.fwd.keys(): self.nearest.get() # find the next nearest circle that is still in the chain
                cn_i     = overlapped_after
                cn       = self.packed[cn_i]
                xy0, xy1 = self.rt_self.overlappingCirclesIntersections((cm[0], cm[1], cm[2] + c[2]), (cn[0], cn[1], cn[2] + c[2]))
                c        = (xy0[0], xy0[1], c[2])
                self.placing_circle = c
                if self.capture_progression: self.progression.append(self._repr_svg_())
            elif overlapped_before is not None:
                self.debug_str += 'b'
                self.__eraseChain__(self.fwd[overlapped_before], cm_i)
                while self.nearest.queue[0][1] not in self.fwd.keys(): self.nearest.get() # find the next nearest circle that is still in the chain
                cm_i     = overlapped_before
                cm       = self.packed[cm_i]
                xy0, xy1 = self.rt_self.overlappingCirclesIntersections((cm[0], cm[1], cm[2] + c[2]), (cn[0], cn[1], cn[2] + c[2]))
                c        = (xy0[0], xy0[1], c[2])
                self.placing_circle = c
                if self.capture_progression: self.progression.append(self._repr_svg_())
            else:
                self.debug_str += 'p'
                self.packed.append(c)
                if self.check_overlaps_every_append:
                    if self.__validateNoOverlaps__() == False:
                        if self.overlap_found_at is None: self.overlap_found_at = len(self.packed)
                self.placing_circle = None
                self.r_max_so_far = max(self.r_max_so_far, c[2])
                _index_           = len(self.packed) - 1
                self.fwd[cm_i], self.bck[_index_] = _index_, cm_i
                self.bck[cn_i], self.fwd[_index_] = _index_, cn_i
                circle_placed     = True
                self.r_max_so_far = max(self.r_max_so_far, self.packed[-1][2])
                self.nearest.put((c[0]**2 + c[1]**2, _index_))
                while self.nearest.queue[0][1] not in self.fwd.keys(): self.nearest.get() # find the next nearest circle that is still in the chain
                # Debugging support
                if self.validate_chains:     self.__validateChains__()
                if self.capture_progression: self.progression.append(self._repr_svg_())

    #
    # __eraseChain__() - erase the chain from fm_i to to_i
    # ... inclusive -- fm_i and to_i will be deleted
    # ... at the end, the chain will be reconnected
    #
    def __eraseChain__(self, fm_i, to_i):
        i_start = self.bck[fm_i]
        i_end   = self.fwd[to_i]
        while fm_i != i_end:
            i_next = self.fwd[fm_i]
            del self.fwd[fm_i]
            fm_i   = i_next
        while to_i != i_start:
            i_prev = self.bck[to_i]
            del self.bck[to_i]
            to_i   = i_prev
        self.fwd[i_start], self.bck[i_end] = i_end, i_start
        if self.validate_chains: self.__validateChains__()
    
    #
    # __validateChains__()
    #
    def __validateChains__(self):
        if len(self.fwd) != len(self.bck):   raise Exception('Chains Are Different Lengths')
        for i in self.fwd.keys():
            if i not in self.bck.keys():     raise Exception('Forward Chain Has Key Not In Backward Chain')
            if self.bck[self.fwd[i]] != i:   raise Exception('Backward Chain Has Key Not In Forward Chain')
        
    #
    # __validateNoOverlaps__()
    #
    def __validateNoOverlaps__(self):
        for i in range(len(self.packed)):
            c0 = self.packed[i]
            for j in range(i+1, len(self.packed)):
                c1 = self.packed[j]
                if self.rt_self.circlesOverlap((c0[0], c0[1], c0[2]), (c1[0], c1[1], c1[2]-self.epsilon)): return False
        return True

