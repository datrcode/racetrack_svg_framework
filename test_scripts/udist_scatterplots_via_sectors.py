#
# Polarse implementation of the following:
#
# H. Rave, V. Molchanov and L. Linsen, "Uniform Sample Distribution in Scatterplots via Sector-based Transformation," 
# 2024 IEEE Visualization and Visual Analytics (VIS), St. Pete Beach, FL, USA, 2024, pp. 156-160, 
# doi: 10.1109/VIS55277.2024.00039. 
# keywords: {Data analysis;Visual analytics;Clutter;Scatterplot de-cluttering;spatial transformation},
#
import polars  as     pl
import numpy   as     np
from   math    import pi, sin, cos, atan2
from   shapely import Polygon
import time

__name__ = 'udist_scatterplots_via_sectors'

class UDistScatterPlotsViaSectors(object):
    def __init__(self, x_vals=[], y_vals=[], weights=None, colors=None, vector_scalar=0.01, iterations=32, debug=True):
        self.vector_scalar = vector_scalar
        self.iterations    = iterations

        # Create the debugging structures
        self.df_at_iteration_start    = []
        self.df_sector_determinations = []
        self.df_sector_angles         = []
        self.df_fully_filled          = []
        self.df_uv                    = []

        # Create weights if none were set
        if weights is None: weights = np.ones(len(x_vals))

        # Prepare the initial dataframe
        if colors is None: df      = pl.DataFrame({'x':x_vals, 'y':y_vals, 'w':weights})            .with_row_index('__index__')
        else:              df      = pl.DataFrame({'x':x_vals, 'y':y_vals, 'w':weights, 'c':colors}).with_row_index('__index__')
        df_orig = df.clone()

        #
        # Perform each iteration
        #
        for _iteration_ in range(iterations):
            # Determine the overall weight sums
            df_weight_sum = df['w'].sum()

            #
            # Normalize the points to 0.02 to 0.98 (want to give it a little space around the edges to that there are sectors to move into)
            #
            df = df.with_columns((0.02 + 0.96 * (pl.col('x') - pl.col('x').min())/(pl.col('x').max() - pl.col('x').min())).alias('x'), 
                                 (0.02 + 0.96 * (pl.col('y') - pl.col('y').min())/(pl.col('y').max() - pl.col('y').min())).alias('y'))

            if debug: self.df_at_iteration_start.append(df.clone())

            #
            # Multiply out the points against all the other points
            # ... greatly explodes the dataframe
            #
            df = df.with_columns(pl.struct(['x','y']).implode().alias('_implode_')) \
                   .explode('_implode_')                                            \
                   .with_columns(pl.col('_implode_').struct.field('x').alias('_xo_'),
                                 pl.col('_implode_').struct.field('y').alias('_yo_'))

            #
            # Determine the sector for the other point in relationship to this point...
            #
            _dx_ = pl.col('_xo_') - pl.col('x')
            _dy_ = pl.col('_yo_') - pl.col('y')
            df   = df.with_columns(((16*(pl.arctan2(_dy_, _dx_) + pl.lit(pi))/(pl.lit(2*pi))).cast(pl.Int64)).alias('sector'))

            if debug: self.df_sector_determinations.append(df.clone())

            #
            # Sum the weights for each sector ... the smaller this dataframe, the better for the next step
            #
            df   = df.group_by(['__index__','x','y','sector']).agg((pl.col('w').sum()).alias('_w_sum_'), (pl.col('w').sum() / df_weight_sum).alias('_w_ratio_'))

            #
            # Create the sector angle dataframe
            # ... this is a small dataframe that covers just 16 sectors ...
            # ... it will be joined with the points dataframe to calculate the area of each sector for each point
            #
            _lu_ = {'sector':[], 
                    'a0':[],       'a0u':[],       'a0v':[],                 # Ray 0 angle & uv components
                    'a1':[],       'a1u':[],       'a1v':[],                 # Ray 1 angle & uv components
                    'corner_x':[], 'corner_y':[],                            # Corner between segment0 and segment 1
                    'anchor_a':[], 'anchor_u':[],  'anchor_v':[],            # Anchor angle & uv components
                    's0x0':[],     's0x1':[],      's0y0':[],     's0y1':[], # Segment 0
                    's1x0':[],     's1x1':[],      's1y0':[],     's1y1':[]} # Segment 1
            for _sector_ in range(16):
                #_lu_['sector'].append(_sector_)
                _sector_align_ = (_sector_ + 8)%16
                _lu_['sector'].append(_sector_align_)
                _a0_ = _sector_*pi/8.0
                _lu_['a0'].append(_a0_), _lu_['a0u'].append(cos(_a0_)), _lu_['a0v'].append(sin(_a0_))
                _a1_ = (_sector_+1)*pi/8.0
                _lu_['a1'].append(_a1_), _lu_['a1u'].append(cos(_a1_)), _lu_['a1v'].append(sin(_a1_))
                #_anchor_ = _a0_ + pi / 2.0 + pi / 16.0 # half angle on top of that
                _anchor_ = _a0_ + pi + pi / 16.0 # half angle on top of that
                _lu_['anchor_a'].append(_anchor_), _lu_['anchor_u'].append(cos(_anchor_)), _lu_['anchor_v'].append(sin(_anchor_))
                if   _sector_ >= 0 and _sector_ <  4:
                    _lu_['s0x0']    .append(1.0), _lu_['s0x1']    .append(1.0), _lu_['s0y0'].append(0.0), _lu_['s0y1'].append(1.0) # segment 0 (x0,y0) -> (x1,y1) (1,0) -> (1,1)
                    _lu_['s1x0']    .append(0.0), _lu_['s1x1']    .append(1.0), _lu_['s1y0'].append(1.0), _lu_['s1y1'].append(1.0) # segment 1 (x0,y0) -> (x1,y1) (0,1) -> (1,1)
                    _lu_['corner_x'].append(1.0), _lu_['corner_y'].append(1.0)
                elif _sector_ >= 4 and _sector_ <  8:
                    _lu_['s0x0']    .append(0.0), _lu_['s0x1']    .append(1.0), _lu_['s0y0'].append(1.0), _lu_['s0y1'].append(1.0) # (0,1) -> (1,1)
                    _lu_['s1x0']    .append(0.0), _lu_['s1x1']    .append(0.0), _lu_['s1y0'].append(0.0), _lu_['s1y1'].append(1.0) # (0,0) -> (0,1)
                    _lu_['corner_x'].append(0.0), _lu_['corner_y'].append(1.0)
                elif _sector_ >= 8 and _sector_ < 12:
                    _lu_['s0x0']    .append(0.0), _lu_['s0x1']    .append(0.0), _lu_['s0y0'].append(0.0), _lu_['s0y1'].append(1.0) # (0,0) -> (0,1)
                    _lu_['s1x0']    .append(0.0), _lu_['s1x1']    .append(1.0), _lu_['s1y0'].append(0.0), _lu_['s1y1'].append(0.0) # (0,0) -> (1,0)
                    _lu_['corner_x'].append(0.0), _lu_['corner_y'].append(0.0)
                else:
                    _lu_['s0x0']    .append(0.0), _lu_['s0x1']    .append(1.0), _lu_['s0y0'].append(0.0), _lu_['s0y1'].append(0.0) # (0,0) -> (1,0)
                    _lu_['s1x0']    .append(1.0), _lu_['s1x1']    .append(1.0), _lu_['s1y0'].append(0.0), _lu_['s1y1'].append(1.0) # (1,0) -> (1,1)
                    _lu_['corner_x'].append(1.0), _lu_['corner_y'].append(0.0)
            df_sector_angles = pl.DataFrame(_lu_)
            df_sector_angles = df_sector_angles.with_columns((pl.col('s0x1') - pl.col('s0x0')).alias('s0u'), (pl.col('s0y1') - pl.col('s0y0')).alias('s0v'),
                                                             (pl.col('s1x1') - pl.col('s1x0')).alias('s1u'), (pl.col('s1y1') - pl.col('s1y0')).alias('s1v'))

            if debug: self.df_sector_angles.append(df_sector_angles)

            # Join w/ sector information
            df = df.join(df_sector_angles, on='sector')

            # Create rays for each sector angles
            df = df.with_columns((pl.col('a0').cos()).alias('xa0'), (pl.col('a0').sin()).alias('ya0'), # uv for angle 0
                                 (pl.col('a1').cos()).alias('xa1'), (pl.col('a1').sin()).alias('ya1')) # uv for angle 1

            # Intersect each ray with each segment (uses the multistep version of ray-segment intersection)
            # ... determinate "r0s0_det" (ray 0, segment 0) .... done four ways (ray 0 & 1 ... segment 0 & 1)
            df = df.with_columns((-pl.col('a0u') * pl.col('s0v') + pl.col('a0v') * pl.col('s0u')).alias('r0s0_det'),
                                 (-pl.col('a0u') * pl.col('s1v') + pl.col('a0v') * pl.col('s1u')).alias('r0s1_det'),
                                 (-pl.col('a1u') * pl.col('s0v') + pl.col('a1v') * pl.col('s0u')).alias('r1s0_det'),
                                 (-pl.col('a1u') * pl.col('s1v') + pl.col('a1v') * pl.col('s1u')).alias('r1s1_det'))
            # ... "t" ("r0s0_t") and "u" values ("r0s0_u") (ray 0, segment 0) for all four ways (ray 0 & 1 ... segment 0 & 1)
            df = df.with_columns((((pl.col('x') - pl.col('s0x0')) * pl.col('s0v') - (pl.col('y') - pl.col('s0y0')) * pl.col('s0u')) / pl.col('r0s0_det')).alias('r0s0_t'),
                                 (((pl.col('x') - pl.col('s0x0')) * pl.col('a0v') - (pl.col('y') - pl.col('s0y0')) * pl.col('a0u')) / pl.col('r0s0_det')).alias('r0s0_u'),

                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('s1v') - (pl.col('y') - pl.col('s1y0')) * pl.col('s1u')) / pl.col('r0s1_det')).alias('r0s1_t'),
                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('a0v') - (pl.col('y') - pl.col('s1y0')) * pl.col('a0u')) / pl.col('r0s1_det')).alias('r0s1_u'),

                                 (((pl.col('x') - pl.col('s0x0')) * pl.col('s0v') - (pl.col('y') - pl.col('s0y0')) * pl.col('s0u')) / pl.col('r1s0_det')).alias('r1s0_t'),
                                 (((pl.col('x') - pl.col('s0x0')) * pl.col('a1v') - (pl.col('y') - pl.col('s0y0')) * pl.col('a1u')) / pl.col('r1s0_det')).alias('r1s0_u'),

                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('s1v') - (pl.col('y') - pl.col('s1y0')) * pl.col('s1u')) / pl.col('r1s1_det')).alias('r1s1_t'),
                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('a1v') - (pl.col('y') - pl.col('s1y0')) * pl.col('a1u')) / pl.col('r1s1_det')).alias('r1s1_u'),)
            # ... the x and y intersects (r0s0_xi, r0s0_yi) (ray 0, segment 0) for all four ways (ray 0 & 1) and segment (0 & 1)
            df = df.with_columns(pl.when((pl.col('r0s0_t') >= 0.0) & (pl.col('r0s0_u') >= 0.0) & (pl.col('r0s0_u') <= 1.0)).then(pl.col('x') + pl.col('r0s0_t') * pl.col('a0u')).otherwise(None).alias('r0s0_xi'),
                                 pl.when((pl.col('r0s0_t') >= 0.0) & (pl.col('r0s0_u') >= 0.0) & (pl.col('r0s0_u') <= 1.0)).then(pl.col('y') + pl.col('r0s0_t') * pl.col('a0v')).otherwise(None).alias('r0s0_yi'),

                                 pl.when((pl.col('r0s1_t') >= 0.0) & (pl.col('r0s1_u') >= 0.0) & (pl.col('r0s1_u') <= 1.0)).then(pl.col('x') + pl.col('r0s1_t') * pl.col('a0u')).otherwise(None).alias('r0s1_xi'),
                                 pl.when((pl.col('r0s1_t') >= 0.0) & (pl.col('r0s1_u') >= 0.0) & (pl.col('r0s1_u') <= 1.0)).then(pl.col('y') + pl.col('r0s1_t') * pl.col('a0v')).otherwise(None).alias('r0s1_yi'),

                                 pl.when((pl.col('r1s0_t') >= 0.0) & (pl.col('r1s0_u') >= 0.0) & (pl.col('r1s0_u') <= 1.0)).then(pl.col('x') + pl.col('r1s0_t') * pl.col('a1u')).otherwise(None).alias('r1s0_xi'),
                                 pl.when((pl.col('r1s0_t') >= 0.0) & (pl.col('r1s0_u') >= 0.0) & (pl.col('r1s0_u') <= 1.0)).then(pl.col('y') + pl.col('r1s0_t') * pl.col('a1v')).otherwise(None).alias('r1s0_yi'),

                                 pl.when((pl.col('r1s1_t') >= 0.0) & (pl.col('r1s1_u') >= 0.0) & (pl.col('r1s1_u') <= 1.0)).then(pl.col('x') + pl.col('r1s1_t') * pl.col('a1u')).otherwise(None).alias('r1s1_xi'),
                                 pl.when((pl.col('r1s1_t') >= 0.0) & (pl.col('r1s1_u') >= 0.0) & (pl.col('r1s1_u') <= 1.0)).then(pl.col('y') + pl.col('r1s1_t') * pl.col('a1v')).otherwise(None).alias('r1s1_yi'),)

            #
            # Area Calculation using Shoelace Formula
            #
            # Case 0 ... which is X_X_ ... which is the first and second ray both hit the first segment
            _c0_0p_x_, _c0_0p_y_, _c0_0q_x_, _c0_0q_y_ = pl.col('r0s0_xi'), pl.col('r0s0_yi'), pl.col('r1s0_xi'), pl.col('r1s0_yi')
            _c0_1p_x_, _c0_1p_y_, _c0_1q_x_, _c0_1q_y_ = pl.col('r1s0_xi'), pl.col('r1s0_yi'), pl.col('x'),       pl.col('y')
            _c0_2p_x_, _c0_2p_y_, _c0_2q_x_, _c0_2q_y_ = pl.col('x'),       pl.col('y'),       pl.col('r0s0_xi'), pl.col('r0s0_yi')
            _c0_op_ = (((_c0_0p_x_*_c0_0q_y_ - _c0_0q_x_*_c0_0p_y_) + (_c0_1p_x_*_c0_1q_y_ - _c0_1q_x_*_c0_1p_y_) + (_c0_2p_x_*_c0_2q_y_ - _c0_2q_x_*_c0_2p_y_))/2.0).abs().alias('area')

            # Case 1 ... which is _X_X ... which is the first and second ray both hit the second segment
            _c1_0p_x_, _c1_0p_y_, _c1_0q_x_, _c1_0q_y_ = pl.col('r0s1_xi'), pl.col('r0s1_yi'), pl.col('r1s1_xi'), pl.col('r1s1_yi')
            _c1_1p_x_, _c1_1p_y_, _c1_1q_x_, _c1_1q_y_ = pl.col('r1s1_xi'), pl.col('r1s1_yi'), pl.col('x'),       pl.col('y')
            _c1_2p_x_, _c1_2p_y_, _c1_2q_x_, _c1_2q_y_ = pl.col('x'),       pl.col('y'),       pl.col('r0s1_xi'), pl.col('r0s1_yi')
            _c1_op_ = (((_c1_0p_x_*_c1_0q_y_ - _c1_0q_x_*_c1_0p_y_) + (_c1_1p_x_*_c1_1q_y_ - _c1_1q_x_*_c1_1p_y_) + (_c1_2p_x_*_c1_2q_y_ - _c1_2q_x_*_c1_2p_y_))/2.0).abs().alias('area')

            # Case 2 ... which is X__X ... which is the first and second ray both hit different segments... so needs the corner position
            _c2_0p_x_, _c2_0p_y_, _c2_0q_x_, _c2_0q_y_ = pl.col('r0s0_xi'),  pl.col('r0s0_yi'),  pl.col('corner_x'), pl.col('corner_y')
            _c2_1p_x_, _c2_1p_y_, _c2_1q_x_, _c2_1q_y_ = pl.col('corner_x'), pl.col('corner_y'), pl.col('r1s1_xi'),  pl.col('r1s1_yi')
            _c2_2p_x_, _c2_2p_y_, _c2_2q_x_, _c2_2q_y_ = pl.col('r1s1_xi'), pl.col('r1s1_yi'),   pl.col('x'),       pl.col('y')
            _c2_3p_x_, _c2_3p_y_, _c2_3q_x_, _c2_3q_y_ = pl.col('x'),       pl.col('y'),         pl.col('r0s0_xi'), pl.col('r0s0_yi')
            _c2_op_ = (((_c2_0p_x_*_c2_0q_y_ - _c2_0q_x_*_c2_0p_y_) + 
                        (_c2_1p_x_*_c2_1q_y_ - _c2_1q_x_*_c2_1p_y_) + 
                        (_c2_2p_x_*_c2_2q_y_ - _c2_2q_x_*_c2_2p_y_) +
                        (_c2_3p_x_*_c2_3q_y_ - _c2_3q_x_*_c2_3p_y_))/2.0).abs().alias('area')

            df = df.with_columns(pl.when(pl.col('r0s0_xi').is_not_null() & pl.col('r1s0_xi').is_not_null()).then(_c0_op_)
                                   .when(pl.col('r0s1_xi').is_not_null() & pl.col('r1s1_xi').is_not_null()).then(_c1_op_)
                                   .when(pl.col('r0s0_xi').is_not_null() & pl.col('r1s1_xi').is_not_null()).then(_c2_op_)
                                   .otherwise(pl.lit(None).alias('area')))

            if debug: self.df_fully_filled.append(df)

            #
            # With the sector sums, adjust the point based on the ratio of the sector area / sector density...
            # ... results of this iteration will be stored in the _xnext_ and _ynext_ fields of the dataframe
            #
            _diff_op_ = (pl.col('_w_ratio_') - pl.col('area'))
            df_uv     = df.group_by(['__index__','x','y']).agg( (vector_scalar * _diff_op_ * pl.col('anchor_u')).sum().alias('_u_'),
                                                                (vector_scalar * _diff_op_ * pl.col('anchor_v')).sum().alias('_v_'))
        
            if debug: self.df_uv.append(df_uv.clone())

            df_uv     = df_uv.with_columns((pl.col('x') + pl.col('_u_')).alias('x'), 
                                           (pl.col('y') + pl.col('_v_')).alias('y'))
            df        = df_uv.join(df_orig, on=['__index__'], how='left') # add the weight back in

            self.df_results = df

#
#
# The following are reference implementations that are straightforward python
#
#

#
# xyUniformSampleDistributionSectorTransform() - implementation of the referenced paper
#
def xyUniformSampleDistributionSectorTransformDEBUG(rt, xvals, yvals, weights=None, colors=None, iterations=4, sectors=16, border_perc=0.01, vector_scalar=0.1):
    svgs, svgs_for_sectors = [], []
    # Normalize the coordinates to be between 0.0 and 1.0
    def normalizeCoordinates(xs, ys):
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        if xmin == xmax: xmin -= 0.0001; xmax += 0.0001
        if ymin == ymax: ymin -= 0.0001; ymax += 0.0001
        xs_new, ys_new = [], []
        for x, y in zip(xs, ys):
            xs_new.append((x-xmin)/(xmax-xmin))
            ys_new.append((y-ymin)/(ymax-ymin))
        return xs_new, ys_new
    # Force all the coordinates to be between 0 and 1
    xvals, yvals = normalizeCoordinates(xvals, yvals)    
    xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0
    xperc, yperc = (xmax-xmin)*border_perc, (ymax-ymin)*border_perc
    xmin, ymin, xmax, ymax = xmin-xperc, ymin-yperc, xmax+xperc, ymax+yperc
    # Determine the average density (used for expected density calculations)
    if weights is None: weights = np.ones(len(xvals))
    weight_sum = sum(weights)
    area_total = ((xmax-xmin)*(ymax-ymin))
    density_avg = weight_sum / area_total
    # Determine the side and xy that a specific ray hits
    def sideAndXY(xy, uv):
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmin, ymin), (xmax, ymin))
        if _xyi_ is not None: return 0, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmax, ymin), (xmax, ymax))
        if _xyi_ is not None: return 1, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmax, ymax), (xmin, ymax))
        if _xyi_ is not None: return 2, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmin, ymax), (xmin, ymin))
        if _xyi_ is not None: return 3, _xyi_
        # hacking the corner cases ... literally the corners
        if xy[0] >= xmin and xy[0] <= xmax and xy[1] >= ymin and xy[1] <= ymax:
            if uv == (0.0, 0.0):
                print(xy, uv, (xmin,ymin,xmax,ymax))
                raise Exception('No Intersection Found for sideAndXY() ... ray is (0,0)')
            else:
                xp, yp, up, vp = round(xy[0], 2), round(xy[1], 2), round(uv[0], 2), round(uv[1], 2)
                if abs(xp) == abs(yp) and abs(up) == abs(vp):
                    if   up < 0.0 and vp < 0.0: return 0, (xmin, ymin)
                    elif up < 0.0 and vp > 0.0: return 1, (xmax, ymin)
                    elif up > 0.0 and vp > 0.0: return 2, (xmax, ymax)
                    elif up > 0.0 and vp < 0.0: return 3, (xmin, ymax)
                print(xy, uv, (xmin,ymin,xmax,ymax))
                raise Exception('No Intersection Found for sideAndXY() ... xy or uv are not equal to one another')
        else:
            print(xy, uv, (xmin,ymin,xmax,ymax))
            raise Exception('No Intersection Found for sideAndXY() ... point not within bounds')
    # Calculate the sector angles
    _sector_angles_, _sector_anchor_ = [], []
    a, ainc = 0.0, 2*pi/sectors
    for s in range(sectors):
        _sector_angles_.append((a, a+ainc))
        _sector_anchor_.append(a + pi + ainc/2.0)
        a += ainc
    # Calculate the UV vector for a specific point
    def ptUVVec(x,y):
        svg_sectors = [f'<svg x="0" y="0" width="512" height="512" viewBox="{xmin} {ymin} {xmax-xmin} {ymax-ymin}" xmlns="http://www.w3.org/2000/svg">']
        svg_sectors.append(f'<rect x="{xmin}" y="{ymin}" width="{xmax-xmin}" height="{ymax-ymin}" fill="#ffffff" />')
        _sector_sum_ = {}
        for s in range(sectors): _sector_sum_[s] = 0.0
        # Iterate over all points ... adding to the sector sum for the correct sector
        for i in range(len(xvals)):
            _x_, _y_, _w_ = xvals[i], yvals[i], weights[i]
            if _x_ == x and _y_ == y: continue
            _dx_, _dy_ = _x_ - x, _y_ - y
            a = atan2(_dy_, _dx_)
            if a < 0.0: a += 2*pi
            _sector_found_ = False
            for s in range(sectors):
                if a >= _sector_angles_[s][0] and a < _sector_angles_[s][1]:
                    _sector_sum_[s] += _w_
                    _color_ = rt.co_mgr.getColor(s)
                    svg_sectors.append(f'<circle cx="{_x_}" cy="{_y_}" r="0.01" stroke="#000000" stroke-width="0.001" fill="{_color_}" />')
                    svg_sectors.append(f'<line x1="{x}" y1="{y}" x2="{_x_}" y2="{_y_}" stroke="#000000" stroke-width="0.001" />')
                    _sector_found_ = True
                    break
            if not _sector_found_: print('No sector found for point', _x_, _y_, a)
        # Determine the area for each sector (from this points perspective)
        _sector_area_, _poly_definition_ = {}, {}
        for s in range(sectors):
            uv          = (cos(_sector_angles_[s][0]), sin(_sector_angles_[s][0]))
            side_and_xy_a0 = sideAndXY((x,y), uv)
            uv = (cos(_sector_angles_[s][1]), sin(_sector_angles_[s][1]))
            side_and_xy_a1 = sideAndXY((x,y), uv)
            if side_and_xy_a0[0] == side_and_xy_a1[0]: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], side_and_xy_a1[1]] # same side
            else:
                if   side_and_xy_a0[0] == 0 and side_and_xy_a1[0] == 1: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmax,ymin), side_and_xy_a1[1]] # top 
                elif side_and_xy_a0[0] == 1 and side_and_xy_a1[0] == 2: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmax,ymax), side_and_xy_a1[1]] # right
                elif side_and_xy_a0[0] == 2 and side_and_xy_a1[0] == 3: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmin,ymax), side_and_xy_a1[1]] # bottom
                elif side_and_xy_a0[0] == 3 and side_and_xy_a1[0] == 0: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmin,ymin), side_and_xy_a1[1]] # left
            _poly_ = Polygon(_poly_definition_[s])
            _sector_area_[s] = _poly_.area
        # From the paper ... weight the anchor the difference between the expected and actual density
        _scalar_ = vector_scalar
        u, v = 0.0, 0.0
        for s in range(sectors):
            _diff_ = (_sector_sum_[s]/weight_sum) - (_sector_area_[s]/area_total)
            u, v   = u + _scalar_ * _diff_ * cos(_sector_anchor_[s]), v + _scalar_ * _diff_ * sin(_sector_anchor_[s])
            _poly_coords_ = _poly_definition_[s]
            d      = f'M {_poly_coords_[0][0]} {_poly_coords_[0][1]} '
            for i in range(1, len(_poly_coords_)): d += f'L {_poly_coords_[i][0]} {_poly_coords_[i][1]} '
            d += 'Z'
            if _diff_ < 0.0: _color_ = rt.co_mgr.getColor(s) # '#0000ff'
            else:            _color_ = rt.co_mgr.getColor(s) # '#ff0000'
            svg_sectors.append(f'<path d="{d}" stroke="{rt.co_mgr.getColor(s)}" fill="{_color_}" fill-opacity="0.3" stroke-width="0.002"/>')
        # Return the value
        svg_sectors.append(f'<line x1="{x}" y1="{y}" x2="{x+3*u}" y2="{y+3*v}" stroke="#ff0000" stroke-width="0.01" />')
        svg_sectors.append('</svg>')
        svgs_for_sectors.append(''.join(svg_sectors))
        return u,v

    # Iterations...
    for iters in range(iterations):
        svg = [f'<svg x="0" y="0" width="256" height="256" viewBox="{xmin} {ymin} {xmax-xmin} {ymax-ymin}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="{xmin}" y="{ymin}" width="{xmax-xmin}" height="{ymax-ymin}" x="0" y="0" fill="#ffffff" />')
        xvals_next, yvals_next = [], []
        for j in range(len(xvals)):
            _x_, _y_ = xvals[j], yvals[j]
            uv = ptUVVec(_x_, _y_)
            svg.append(f'<line x1="{_x_}" y1="{_y_}" x2="{_x_+uv[0]}" y2="{_y_+uv[1]}" stroke="#a0a0a0" stroke-width="0.001" />')
            _color_ = colors[j] if colors is not None else '#000000'
            svg.append(f'<circle cx="{_x_}" cy="{_y_}" r="0.004" fill="{_color_}" />')
            _x_next_, _y_next_ = _x_ + uv[0], _y_ + uv[1]
            xvals_next.append(_x_next_), yvals_next.append(_y_next_)
        svg.append('</svg>')
        svgs.append(''.join(svg))
        xvals, yvals = xvals_next, yvals_next
        xvals, yvals = normalizeCoordinates(xvals, yvals)    
        xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0
        xperc, yperc = (xmax-xmin)*border_perc, (ymax-ymin)*border_perc
        xmin, ymin, xmax, ymax = xmin-xperc, ymin-yperc, xmax+xperc, ymax+yperc

    # Return
    return xvals, yvals, svgs, svgs_for_sectors

#
# xyUniformSampleDistributionSectorTransform() - implementation of the referenced paper
# ... the above version is debug ... this removes all of the svg creation
#
def xyUniformSampleDistributionSectorTransform(rt, xvals, yvals, weights=None, iterations=4, sectors=16, border_perc=0.01, vector_scalar=0.1):
    # Normalize the coordinates to be between 0.0 and 1.0
    def normalizeCoordinates(xs, ys):
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        if xmin == xmax: xmin -= 0.0001; xmax += 0.0001
        if ymin == ymax: ymin -= 0.0001; ymax += 0.0001
        xs_new, ys_new = [], []
        for x, y in zip(xs, ys):
            xs_new.append((x-xmin)/(xmax-xmin))
            ys_new.append((y-ymin)/(ymax-ymin))
        return xs_new, ys_new
    # Force all the coordinates to be between 0 and 1
    xvals, yvals = normalizeCoordinates(xvals, yvals)    
    xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0
    xperc, yperc = (xmax-xmin)*border_perc, (ymax-ymin)*border_perc
    xmin, ymin, xmax, ymax = xmin-xperc, ymin-yperc, xmax+xperc, ymax+yperc
    # Determine the average density (used for expected density calculations)
    if weights is None: weights = np.ones(len(xvals))
    weight_sum = sum(weights)
    area_total = ((xmax-xmin)*(ymax-ymin))
    density_avg = weight_sum / area_total
    # Determine the side and xy that a specific ray hits
    def sideAndXY(xy, uv):
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmin, ymin), (xmax, ymin))
        if _xyi_ is not None: return 0, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmax, ymin), (xmax, ymax))
        if _xyi_ is not None: return 1, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmax, ymax), (xmin, ymax))
        if _xyi_ is not None: return 2, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmin, ymax), (xmin, ymin))
        if _xyi_ is not None: return 3, _xyi_
        # hacking the corner cases ... literally the corners
        if xy[0] >= xmin and xy[0] <= xmax and xy[1] >= ymin and xy[1] <= ymax:
            if uv == (0.0, 0.0):
                print(xy, uv, (xmin,ymin,xmax,ymax))
                raise Exception('No Intersection Found for sideAndXY() ... ray is (0,0)')
            else:
                xp, yp, up, vp = round(xy[0], 2), round(xy[1], 2), round(uv[0], 2), round(uv[1], 2)
                if abs(xp) == abs(yp) and abs(up) == abs(vp):
                    if   up < 0.0 and vp < 0.0: return 0, (xmin, ymin)
                    elif up < 0.0 and vp > 0.0: return 1, (xmax, ymin)
                    elif up > 0.0 and vp > 0.0: return 2, (xmax, ymax)
                    elif up > 0.0 and vp < 0.0: return 3, (xmin, ymax)
                print(xy, uv, (xmin,ymin,xmax,ymax))
                raise Exception('No Intersection Found for sideAndXY() ... xy or uv are not equal to one another')
        else:
            print(xy, uv, (xmin,ymin,xmax,ymax))
            raise Exception('No Intersection Found for sideAndXY() ... point not within bounds')
    # Calculate the sector angles
    _sector_angles_, _sector_anchor_ = [], []
    a, ainc = 0.0, 2*pi/sectors
    for s in range(sectors):
        _sector_angles_.append((a, a+ainc))
        _sector_anchor_.append(a + pi + ainc/2.0)
        a += ainc
    # Calculate the UV vector for a specific point
    def ptUVVec(x,y):
        _sector_sum_ = {}
        for s in range(sectors): _sector_sum_[s] = 0.0
        # Iterate over all points ... adding to the sector sum for the correct sector
        for i in range(len(xvals)):
            _x_, _y_, _w_ = xvals[i], yvals[i], weights[i]
            if _x_ == x and _y_ == y: continue
            _dx_, _dy_ = _x_ - x, _y_ - y
            a = atan2(_dy_, _dx_)
            if a < 0.0: a += 2*pi
            _sector_found_ = False
            for s in range(sectors):
                if a >= _sector_angles_[s][0] and a < _sector_angles_[s][1]:
                    _sector_sum_[s] += _w_
                    _sector_found_   = True
                    break
            if not _sector_found_: print('No sector found for point', _x_, _y_, a)
        # Determine the area for each sector (from this points perspective)
        _sector_area_, _poly_definition_ = {}, {}
        for s in range(sectors):
            uv          = (cos(_sector_angles_[s][0]), sin(_sector_angles_[s][0]))
            side_and_xy_a0 = sideAndXY((x,y), uv)
            uv = (cos(_sector_angles_[s][1]), sin(_sector_angles_[s][1]))
            side_and_xy_a1 = sideAndXY((x,y), uv)
            if side_and_xy_a0[0] == side_and_xy_a1[0]: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], side_and_xy_a1[1]] # same side
            else:
                if   side_and_xy_a0[0] == 0 and side_and_xy_a1[0] == 1: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmax,ymin), side_and_xy_a1[1]] # top 
                elif side_and_xy_a0[0] == 1 and side_and_xy_a1[0] == 2: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmax,ymax), side_and_xy_a1[1]] # right
                elif side_and_xy_a0[0] == 2 and side_and_xy_a1[0] == 3: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmin,ymax), side_and_xy_a1[1]] # bottom
                elif side_and_xy_a0[0] == 3 and side_and_xy_a1[0] == 0: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmin,ymin), side_and_xy_a1[1]] # left
            _poly_ = Polygon(_poly_definition_[s])
            _sector_area_[s] = _poly_.area
        # From the paper ... weight the anchor the difference between the expected and actual density
        _scalar_ = vector_scalar
        u, v = 0.0, 0.0
        for s in range(sectors):
            _diff_ = (_sector_sum_[s]/weight_sum) - (_sector_area_[s]/area_total)
            u, v   = u + _scalar_ * _diff_ * cos(_sector_anchor_[s]), v + _scalar_ * _diff_ * sin(_sector_anchor_[s])
        return u,v

    # Iterations...
    for iters in range(iterations):
        xvals_next, yvals_next = [], []
        for j in range(len(xvals)):
            _x_, _y_ = xvals[j], yvals[j]
            uv = ptUVVec(_x_, _y_)
            _x_next_, _y_next_ = _x_ + uv[0], _y_ + uv[1]
            xvals_next.append(_x_next_), yvals_next.append(_y_next_)
        xvals, yvals = xvals_next, yvals_next
        xvals, yvals = normalizeCoordinates(xvals, yvals)    
        xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0
        xperc, yperc = (xmax-xmin)*border_perc, (ymax-ymin)*border_perc
        xmin, ymin, xmax, ymax = xmin-xperc, ymin-yperc, xmax+xperc, ymax+yperc

    # Return
    return xvals, yvals

