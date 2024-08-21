import pandas as pd
import numpy  as np
import polars as pl
import unittest

from rtsvg import *

class TestRTSVG(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()
        lu = {'a':[10,   20,   12,   15,   18,   100,  101],
              'b':['a',  'b',  'c',  'a',  'b',  'c',  'a'],
              'c':[1,    2,    3,    1,    2,    3,    1]}
        self.df_pd   = pd.DataFrame(lu)
        self.df_pl   = pl.DataFrame(lu)

    def test_isPandas(self):
        self.assertTrue (self.rt_self.isPandas(self.df_pd))
        self.assertFalse(self.rt_self.isPandas(self.df_pl))
    
    def test_isPolars(self):
        self.assertFalse(self.rt_self.isPolars(self.df_pd))
        self.assertTrue (self.rt_self.isPolars(self.df_pl))

if __name__ == '__main__':
    unittest.main()
