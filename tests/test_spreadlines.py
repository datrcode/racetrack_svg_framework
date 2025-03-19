import polars as pl
import unittest
import rtsvg

class TestSCUEditorPanel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = rtsvg.RACETrack()
        _lu_ = {'sip':[], 'dip':[], 'timestamp':[]}
        def nodify(_fm_, _to_, _ts_):
            _fm_split_, _to_split_ = _fm_.split(), _to_.split()
            for i in range(len(_fm_split_)):
                _lu_['sip'].append(_fm_split_[i]), _lu_['dip'].append(_to_split_[i]), _lu_['timestamp'].append(_ts_)

        _fms_ = 'a    a    a     f0s0 f1s0 g0s0 g1s0 f0s0 f1s0 u0s0 u1s0'
        _tos_ = 't0s0 t1s0 t2s0  a    a    f0s0 f1s0 g2s0 g3s0 t0s0 t1s0'
        nodify(_fms_, _tos_, '2022-02-01')

        _fms_ = 'a    a    a    f0s0 f1s1  g0s0 t3s1'
        _tos_ = 't0s0 t1s0 t3s1 a    a     f1s1 u0s0'
        nodify(_fms_, _tos_, '2022-02-02')

        _fms_ = 'a    a    a    f0s0       g1s0 u0s0 u1s0'
        _tos_ = 't0s0 t1s0 t3s1 a          f0s0 t1s0 t3s1'
        nodify(_fms_, _tos_, '2022-02-03')

        _fms_ = 'a    a    f1s1'
        _tos_ = 't0s0 t1s0 a'
        nodify(_fms_, _tos_, '2022-02-04')

        self.df = pl.DataFrame(_lu_)
        self.df = self.rt_self.columnsAreTimestamps(self.df, 'timestamp')
        self.params = {'rt_self':self.rt_self, 'df':self.df, 'relationships':[('sip','dip')], 'node_color':'node', 'every':'1d', 'h':256}

    def test_1(self):
        sl = rtsvg.spreadLines(node_focus='a',                                           **self.params)
        sl._repr_svg_()

    def test_2(self):
        sl = rtsvg.spreadLines(node_focus='a', only_render_nodes=set(['t0s0', 'f0s0',]), **self.params)
        sl._repr_svg_()

    def test_3(self):
        sl = rtsvg.spreadLines(node_focus='a', only_render_nodes=set(),                  **self.params)
        sl._repr_svg_()
