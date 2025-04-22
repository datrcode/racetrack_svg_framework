import unittest
import rtsvg
import random

class TestCirclePacker(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = rtsvg.RACETrack()

    def test_1(self):
        #
        # 28.0s for 100_000 circles on M1 Pro (Unoptimized)
        #  2.9s for 100_000 circles on M1 Pro (w/ Arc Length Optimization)
        #  2.6s for 100_000 circles on M1 Pro (w/ Arc Length Optimization) // after a lot of debugging...
        #
        _circles_ = []
        for i in range(100_000): _circles_.append((0.0, 0.0, 1.0 + 3.0*random.random()))
        _cp_ = rtsvg.CirclePacker(self.rt_self, _circles_)

    def test_2(self):
        #
        # [1.0, 4.0] -- max difference in radius
        # ... seen breakage at [1.0, 4.25]
        # ... but with the sorting safeguard, it shouldn't matter anymore
        #
        for i in range(1_000):
            _circles_ = []
            for i in range(50): _circles_.append((0.0, 0.0, 1.0 + 3.0*random.random()))
            _cp_ = rtsvg.CirclePacker(self.rt_self, _circles_)
            if _cp_.__validateNoOverlaps__() == False: raise Exception('Overlaps Found')
            _packed_ = self.rt_self.packCircles(_circles_)


    def test_3(self):
        #
        # [0.1, 0.4125] -- max difference in radius
        # ... seen breakage at [0.1, 0.425]
        # ... but with the sorting safeguard, it shouldn't matter anymore
        #
        for i in range(1_000):
            _circles_ = []
            for i in range(50): _circles_.append((0.0, 0.0, 0.1 + 0.3125*random.random()))
            _cp_ = rtsvg.CirclePacker(self.rt_self, _circles_)
            if _cp_.__validateNoOverlaps__() == False: raise Exception('Overlaps Found')
            _packed_ = self.rt_self.packCircles(_circles_)


    def test_4(self):
        for i in range(1_000):
            _circles_ = []
            for i in range(50): _circles_.append((0.0, 0.0, 0.1 + 10.0*random.random()))
            _cp_ = rtsvg.CirclePacker(self.rt_self, _circles_)
            if _cp_.__validateNoOverlaps__() == False: raise Exception('Overlaps Found')
            _packed_ = self.rt_self.packCircles(_circles_)
