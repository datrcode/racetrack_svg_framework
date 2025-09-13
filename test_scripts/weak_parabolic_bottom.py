import polars as pl
import numpy as np
import random
import rtsvg

__name__ = 'weak_parabolic_bottom'

#
# WeakParabolicBottom - find the bottom of a parabolic-like function (weakly)
# ... bottom is at a positive x value (with zero & negative x values being invalid)
# ... function is like a parabola but not exactly
# ... parabola curves upward in general
# ... likely solution is a sub 1000 number
#
class WeakParabolicBottom(object):
    def __init__(self, fn):
        def approxEqual(xy0, xy1):
            if round(xy0[0],3) == round(xy1[0],3) and round(xy0[1],3) == round(xy1[1],3): return True
            return False

        # Make some initial points at the earliest possible location
        self.xys = []
        self.xys.append((1.0,   fn(1.0)))
        self.xys.append((10.0,  fn(10.0)))
        self.xys.append((100.0, fn(100.0)))
        # Fit a parabola
        a, b, c = self.fitParabolaNumpy(self.xys)
        self.parabolas = [(a,b,c)]
        # Find the bottom
        self.bottoms   = [self.parabolaBottom(a,b,c)]
        # Iterate until convergence (for now, it will be just iterate three times)
        for i in range(100):
            # Add the bottom to the xys array
            x = self.bottoms[-1][0]
            
            # Use the last three xys to fit a parabola & then find the bottom
            a, b, c = self.fitParabolaNumpy(self.xys[-3:])
            x, y    = self.parabolaBottom(a,b,c)
            if x > 0.0:
                self.parabolas.append((a,b,c))
                self.bottoms.append((x,y))
            else:
                x = 0.1 + random.random() * 100.0
            f_x = fn(x)
            self.xys.append((x, fn(x)))    
            # If they've converged, stop
            if approxEqual(self.xys[-1], self.xys[-2]): break

    #
    # parabolaBottom() - given a, b, and c, calculate the bottom of the parabola
    # - assumes the parabola opens upward
    #
    def parabolaBottom(self, a, b, c):
        x = -b / (2 * a)
        y = a * x**2 + b * x + c 
        return x, y

    #
    # ChatGPT Response (2025-09-10)
    # Prompt: "Given three points (p0, p1, and p2), calculate the parabolic parameters a, b, and c."
    #
    def fitParabolaNumpy(self, points):
        (x0,y0),(x1,y1),(x2,y2) = points
        A = np.array([[x0*x0, x0, 1.0],
                      [x1*x1, x1, 1.0],
                      [x2*x2, x2, 1.0]], dtype=float)
        y = np.array([y0,y1,y2], dtype=float)
        a,b,c = np.linalg.solve(A,y)
        return a,b,c

    def _repr_svg_(self):
        _lu_ = {'x':[], 'y':[], 'group':[]}
        for x,y in self.xys: _lu_['x'].append(x), _lu_['y'].append(y), _lu_['group'].append('xys')
        _df_xys_ = pl.DataFrame(_lu_)

        x0, x1 = self.xys[0][0], self.xys[0][0]
        for _xy_ in self.xys:
            x0 = min(x0, _xy_[0])
            x1 = max(x1, _xy_[0])

        _lu_ = {'x':[], 'y':[], 'group':[]}
        for _parabola_ in self.parabolas:
            a, b, c = _parabola_   
            x = x0
            while x <= x1:
                _lu_['x'].append(x), _lu_['y'].append(a * x**2 + b * x + c), _lu_['group'].append(f'parabola {_parabola_}')
                x += (x1-x0)/100.0
        _df_parabolas_ = pl.DataFrame(_lu_)
        _df_ = pl.concat([_df_xys_, _df_parabolas_])

        return rtsvg.RACETrack().xy(_df_, x_field='x', y_field='y', color_by='group', dot_size=None, line_groupby_field='group',
                                    w=900, h=600)._repr_svg_()
        

