import copy
import queue
from math import sqrt, acos, pi, atan2

__name__ = 'circle_packer'

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
    def __init__(self, rt_self, circles, epsilon=0.01, largest_to_smallest=True, keep_order=True):
        self.rt_self             = rt_self
        self.circles             = circles

        # Find the min and max radii
        circles_with_i = []
        self.r_min = self.r_max  = self.circles[0][2]
        for i in range(len(self.circles)):
            c = self.circles[i]
            self.r_min, self.r_max = min(self.r_min, c[2]), max(self.r_max, c[2])
            circles_with_i.append((c[0], c[1], c[2], i))
        self.circles = circles_with_i

        # If the delta is too great, then sort
        if (self.r_max / self.r_min) > 4.125: 
            self.circles = sorted(self.circles, key=lambda x: x[2], reverse=largest_to_smallest)
            pre_sorted   = True
        else:
            pre_sorted   = False

        self.circles_left        = copy.deepcopy(self.circles)
        self.epsilon             = epsilon
        self.packed              = []
        self.fwd                 = {}
        self.bck                 = {}
        self.r_max_so_far        = 0.0 # for optimization
        self.nearest             = queue.PriorityQueue()

        # Pack the first circles and center them
        self.__packFirstCircles__()
        self.__recenterPackedCircles__()

        # Capture the maximum radius seen so far (for the optimization implementation)
        for i in range(len(self.packed)): self.r_max_so_far = max(self.r_max_so_far, self.packed[i][2])
        # Create a priority queue to find the circle w/in the chain that is closest to the origin
        for i in range(len(self.packed)):
            if i not in self.fwd: continue
            c = self.packed[i]
            self.nearest.put((sqrt(c[0]**2 + c[1]**2)+c[2], i))
        # Pack the circles iteratively
        while len(self.circles_left) > 0: self.__packNextCircle__()

        # Unsort it back into the original ... then remove the sorting index
        # ... note that unsorting will mess up the chains
        if pre_sorted and keep_order:
            new_fwd, new_bck = {}, {}
            for k in self.fwd:
                new_k = self.packed[k][3]
                new_v = self.packed[self.fwd[k]][3]
                new_fwd[new_k] = new_v
                new_bck[new_v] = new_k
            self.fwd, self.bck = new_fwd, new_bck
            self.packed = sorted(self.packed, key=lambda x: x[3])
        wout_i = []
        for c in self.packed: wout_i.append((c[0], c[1], c[2]))
        self.packed = wout_i

    #
    # packedCircles() - return the packed circles
    #
    def packedCircles(self, into_circle=None):
        if into_circle is None: return copy.deepcopy(self.packed)
        _inscribed_ = self.optimalInscriptionCircle()
        scale       = _inscribed_[2] / into_circle[2]
        _return_    = []
        for i in range(len(self.packed)):
            _to_transform_ = self.packed[i]
            x = (_to_transform_[0] - _inscribed_[0])/scale + into_circle[0]
            y = (_to_transform_[1] - _inscribed_[1])/scale + into_circle[1]
            r = _to_transform_[2] / scale
            _return_.append((x,y,r))
        return _return_

    #
    # __packedCircleExtents__() - determine the extents of the packed circles to include the radii
    #
    def __packedCircleExtents__(self):
        _pkd_ = self.packed
        x0, y0, x1, y1 = _pkd_[0][0] - _pkd_[0][2], _pkd_[0][1] - _pkd_[0][2], _pkd_[0][0] + _pkd_[0][2], _pkd_[0][1] + _pkd_[0][2]
        for i in range(1, len(_pkd_)): x0, y0, x1, y1 = min(x0, _pkd_[i][0] - _pkd_[i][2]), min(y0, _pkd_[i][1] - _pkd_[i][2]), max(x1, _pkd_[i][0] + _pkd_[i][2]), max(y1, _pkd_[i][1] + _pkd_[i][2])
        return x0, y0, x1, y1

    #
    # __recenterPackedCircles__() - recenter the packed circles in place
    #
    def __recenterPackedCircles__(self):
        x0, y0, x1, y1 = self.__packedCircleExtents__()
        for i in range(len(self.packed)): self.packed[i] = (self.packed[i][0] - (x0 + x1)/2.0, self.packed[i][1] - (y0 + y1)/2.0, self.packed[i][2], self.packed[i][3])

    #
    # __packFirstThreeCircles__()
    #
    def __packFirstCircles__(self):
        # Circle 1
        cx0, cy0, r0, i0  = self.circles_left.pop(0)
        cx0 = cy0 = 0.0
        self.packed.append((cx0, cy0, r0, i0))
        self.fwd, self.bck = {0:0}, {0:0}
        if len(self.circles_left) == 0: return

        # Circle 2
        cx1, cy1, r1, i1  = self.circles_left.pop(0)
        cy1               = 0.0
        cx1               = r0 + r1
        self.packed.append((cx1, 0.0, r1, i1))
        self.fwd, self.bck = {0:1, 1:0}, {0:1, 1:0}
        if len(self.circles_left) == 0: return

        # Circle 3
        cx2, cy2, r2, i2  = self.circles_left.pop(0)
        xy0, xy1          = self.rt_self.overlappingCirclesIntersections((cx0,cy0,r0+r2),(cx1,cy1,r1+r2))
        cx2, cy2          = xy0[0], xy0[1]
        self.packed.append((cx2, cy2, r2, i2))
        self.fwd, self.bck = {0:1, 1:2, 2:0}, {1:0, 2:1, 0:2}
        if len(self.circles_left) == 0: return

        # Circle 4
        cx3, cy3, r3, i3  = self.circles_left.pop(0)
        xy0, xy1          = self.rt_self.overlappingCirclesIntersections((cx1,cy1,r1+r3),(cx2,cy2,r2+r3))
        cx3, cy3          = xy0[0], xy0[1]

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
                            if self.rt_self.circlesOverlap((cx2, cy2, r2), (cx3, cy3, r3)): raise Exception('__packFirstCircles__() - 6 - should not happen')
                            else:                                                           raise Exception('__packFirstCircles__() - 5 - should not happen')
                        else:
                            self.fwd, self.bck = {0:3, 3:1, 1:0}, {3:0, 1:3, 0:1}        
                    else:
                        raise Exception('__packFirstCircles__() - 3 - should not happen')
                else:
                    self.fwd, self.bck = {0:3, 3:2, 2:1, 1:0}, {3:0, 2:3, 1:2, 0:1}
            else:
                self.fwd, self.bck = {0:2, 1:0, 2:3, 3:1}, {2:0, 0:1, 3:2, 1:3}
        else:            
            if   self.packed[0][2] <= self.packed[1][2] and self.packed[0][2] <= self.packed[2][2] and self.packed[0][2] <= r3: # 0 is the smallest
                self.fwd, self.bck = {3:2, 2:1, 1:3}, {2:3, 1:2, 3:1}
            elif self.packed[1][2] <= self.packed[0][2] and self.packed[1][2] <= self.packed[2][2] and self.packed[1][2] <= r3: # 1 is the smallest
                raise Exception('__packFirstCircles__() - 0 - should not happen (01)')
            elif self.packed[2][2] <= self.packed[0][2] and self.packed[2][2] <= self.packed[1][2] and self.packed[2][2] <= r3: # 2 is the smallest
                raise Exception('__packFirstCircles__() - 0 - should not happen (02)')
            else:                                                                                                               # 3 is the smallest
                self.fwd, self.bck = {0:2, 2:1, 1:0}, {2:0, 1:2, 0:1}

        self.packed.append((cx3, cy3, r3, i3))

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
        c                   = (xy0[0], xy0[1], c[2], c[3])
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
                c        = (xy0[0], xy0[1], c[2], c[3])
            elif overlapped_after  is not None:
                self.__eraseChain__(cn_i, self.bck[overlapped_after])
                while self.nearest.queue[0][1] not in self.fwd.keys(): self.nearest.get() # find the next nearest circle that is still in the chain
                cn_i     = overlapped_after
                cn       = self.packed[cn_i]
                xy0, xy1 = self.rt_self.overlappingCirclesIntersections((cm[0], cm[1], cm[2] + c[2]), (cn[0], cn[1], cn[2] + c[2]))
                c        = (xy0[0], xy0[1], c[2], c[3])
            elif overlapped_before is not None:
                self.__eraseChain__(self.fwd[overlapped_before], cm_i)
                while self.nearest.queue[0][1] not in self.fwd.keys(): self.nearest.get() # find the next nearest circle that is still in the chain
                cm_i     = overlapped_before
                cm       = self.packed[cm_i]
                xy0, xy1 = self.rt_self.overlappingCirclesIntersections((cm[0], cm[1], cm[2] + c[2]), (cn[0], cn[1], cn[2] + c[2]))
                c        = (xy0[0], xy0[1], c[2], c[3])
            else:
                self.packed.append(c)
                _index_           = len(self.packed) - 1
                self.fwd[cm_i], self.bck[_index_] = _index_, cm_i
                self.bck[cn_i], self.fwd[_index_] = _index_, cn_i
                circle_placed     = True
                self.r_max_so_far = max(self.r_max_so_far, self.packed[-1][2])
                self.nearest.put((sqrt(c[0]**2 + c[1]**2)+c[2], _index_))
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
        x0, y0, x1, y1 = self.__packedCircleExtents__()
        x0, y0, x1, y1 = x0-3, y0-3, x1+3, y1+3
        svg = [f'<svg x="0" y="0" width="{w}" height="{h}" viewBox="{x0} {y0} {x1-x0} {y1-y0}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#ffffff" />')
        svg.append(f'<line x1="0.0"  y1="{y0}" x2="0.0"  y2="{y1}" stroke="#ff0000" stroke-width="0.1" />')
        svg.append(f'<line x1="{x0}" y1="0.0"  x2="{x1}" y2="0.0"  stroke="#ff0000" stroke-width="0.1" />')
        for i in range(len(_pkd_)):
            _c_ = _pkd_[i]
            if len(self.nearest.queue) > 0 and i == self.nearest.queue[0][1]: _color_ = '#ff0000'
            else:                                                             _color_ = '#000000'
            svg.append(f'<circle cx="{_c_[0]}" cy="{_c_[1]}" r="{_c_[2]}" fill="none" stroke="{_color_}" stroke-width="0.2" />')
            #svg.append(self.rt_self.svgText(str(i), _c_[0], _c_[1], txt_h=3))
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
        
        _optimal_ = self.optimalInscriptionCircle()
        svg.append(f'<circle cx="{_optimal_[0]}" cy="{_optimal_[1]}" r="{_optimal_[2]}" fill="none" stroke="#000000" stroke-width="0.2" />')

        svg.append('</svg>')
        return ''.join(svg)

    #
    # __approximateInscribedCircle__() - approximate the inscribed circle
    # ... given an initial point, determine the three furthest points within the packed circles
    # ... from these points, calculate the inscribed circle
    # ... not guaranteed to even return a circle that completely contains all of the packed circles
    #
    def __approximateInscribedCircle__(self, xy=None, angular_distance=pi/3.0):
        # Get just the border circles
        _border_circles_ = []
        _packed_circles_ = self.packedCircles()
        for k, v in self.fwd.items(): _border_circles_.append(_packed_circles_[v])
        if xy is None:
            _x0_, _y0_, _x1_, _y1_ = self.__packedCircleExtents__()
            _xcen_, _ycen_ = (_x0_ + _x1_) / 2.0, (_y0_ + _y1_) / 2.0
        else: _xcen_, _ycen_ = xy
        # For all of the border circles, calculate how far they are (furthest point) from the center
        _sorter_ = []
        _furthest_ = None
        for i in range(len(_border_circles_)):
            _c_  = _border_circles_[i]
            _l_  = sqrt((_c_[0] - _xcen_)**2 + (_c_[1] - _ycen_)**2) + _c_[2]
            _uv_ = self.rt_self.unitVector(((_xcen_, _ycen_), (_c_[0], _c_[1])))
            _xouter_, _youter_ = _xcen_ + _l_ * _uv_[0], _ycen_ + _l_ * _uv_[1]
            if _furthest_ is None or _l_ > _furthest_: _furthest_ = _l_
            _sorter_.append((_l_, _xouter_, _youter_, atan2(_youter_ - _ycen_, _xouter_ - _xcen_), i))
        # Pick the three largest distances as long as they have sufficient angular distances
        _sorter_.sort(reverse=True)
        # Choose the first point
        i0 = 0
        # Find the second point
        i1 = None
        i  = 1
        while i1 is None and i < len(_sorter_):
            _angle_d_  = pi - abs(abs(_sorter_[i][3] - _sorter_[i0][3]) - pi)
            if _angle_d_ >= angular_distance: i1 = i
            i += 1
        if i1 is None: i1 = 1
        # Find the third point
        i2 = None
        i  = i1 + 1
        while i2 is None and i < len(_sorter_):
            _angle0_d_  = pi - abs(abs(_sorter_[i][3] - _sorter_[i0][3]) - pi)
            _angle1_d_  = pi - abs(abs(_sorter_[i][3] - _sorter_[i1][3]) - pi)
            if _angle0_d_ >= angular_distance and _angle1_d_ >= angular_distance: i2 = i
            i += 1
        if i2 is None: # just take the first point that isn't one of the other points
            i = 1 # i0 + 1
            while i2 is None and i < len(_sorter_):
                if i != i1: i2 = i
                i += 1
        # Create the circle from those three points:
        p0, p1, p2 = (_sorter_[i0][1], _sorter_[i0][2]), (_sorter_[i1][1], _sorter_[i1][2]), (_sorter_[i2][1], _sorter_[i2][2])
        _circle_ = self.rt_self.threePointCircle(p0, p1, p2)
        return _circle_, [p0, p1, p2]

    #
    # optimalInscriptionCircle() - find an optimal circle that inscribes all of the circles
    # ... I couldn't fit in "bestThingThatICouldDoWithMyMeagerSkillsInscriptionCircle()"
    #
    def optimalInscriptionCircle(self, iterations=10, angular_distance=pi/3.0):
        _border_circles_ = []
        _packed_circles_ = self.packedCircles()
        for k, v in self.fwd.items(): _border_circles_.append(_packed_circles_[v])
        def __findRMax__(_xycoord_):
            _r_max_ = 0.0
            for _bc_ in _border_circles_:
                _r_ = self.rt_self.segmentLength((_xycoord_, _bc_)) + _bc_[2]
                _r_max_ = max(_r_max_, _r_)
            return _r_max_
        # Start with the initial approximation based on the extents
        # ... for some reason, this works best for four circles
        _x0_, _y0_, _x1_, _y1_ = self.__packedCircleExtents__()
        _xy_ = (_x0_ + _x1_) / 2.0, (_y0_ + _y1_) / 2.0
        _r_best_, _xy_best_ = __findRMax__(_xy_), _xy_
        if len(_border_circles_) < 3: return (_xy_best_[0], _xy_best_[1], _r_best_) # technique doesn't work for 2 circles (or less)
        # Iterate on the appoximation X tiles
        for i in range(iterations):
            _circle_, _points_ = self.__approximateInscribedCircle__(xy=_xy_, angular_distance=angular_distance)
            _xy_ = (_circle_[0], _circle_[1])
            # Find the new r
            _r_max_ = __findRMax__(_xy_)
            for _bc_ in _border_circles_:
                _r_ = self.rt_self.segmentLength((_xy_, _bc_)) + _bc_[2]
                _r_max_ = max(_r_max_, _r_)
            if _r_best_ is None or _r_max_ < _r_best_:
                _r_best_ = _r_max_
                _xy_best_ = _xy_
        return (_xy_best_[0], _xy_best_[1], _r_best_)
