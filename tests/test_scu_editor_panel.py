import pandas as pd
import unittest
from rtsvg import *

class TestSCUEditorPanel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()
    
    def test_1(self):
        _lu_ = {'_id_':[],'q':[],'my_scu':[],'my_source':[],'my_summary':[],'my_excerpts':[]}
        _lu_['_id_']       .append(1)
        _lu_['q']          .append('What is your favorite color?')
        _lu_['my_scu']     .append('red')
        _lu_['my_source']  .append('wikipedia')
        _lu_['my_summary'] .append('Red is the color of blood, roses, and scarlet. It is also the color of many fruits, such as strawberries, raspberries, and blackberries.')
        _lu_['my_excerpts'].append('color of blood, roses ... strawberries, raspberries')
        _lu_['_id_']       .append(1)
        _lu_['q']          .append('What is your favorite color?')
        _lu_['my_scu']     .append('red')
        _lu_['my_source']  .append('otherpedia')
        _lu_['my_summary'] .append('Red is just another color.')
        _lu_['my_excerpts'].append('just another color')
        df = pd.DataFrame(_lu_)
        scuep = SCUEditorPanel(self.rt_self, df, 1, '_id_', 'q', 'my_scu', 'my_source', 'my_summary', 'my_excerpts')
        scuep.createHTMLForMissingSCUs()

    def test_2(self):
        _lu_ = {'_id_':[],'q':[],'my_scu':[],'my_source':[],'my_summary':[],'my_excerpts':[]}
        _lu_['_id_']       .append(1)
        _lu_['q']          .append('What is your favorite color?')
        _lu_['my_scu']     .append('red')
        _lu_['my_source']  .append(10)
        _lu_['my_summary'] .append('Red is the color of blood, roses, and scarlet. It is also the color of many fruits, such as strawberries, raspberries, and blackberries.')
        _lu_['my_excerpts'].append('color of blood, roses ... strawberries, raspberries')
        _lu_['_id_']       .append(1)
        _lu_['q']          .append('What is your favorite color?')
        _lu_['my_scu']     .append('red')
        _lu_['my_source']  .append(20)
        _lu_['my_summary'] .append('Red is just another color.')
        _lu_['my_excerpts'].append('just another color')
        df = pd.DataFrame(_lu_)
        scuep = SCUEditorPanel(self.rt_self, df, 1, '_id_', 'q', 'my_scu', 'my_source', 'my_summary', 'my_excerpts')
        scuep.createHTMLForMissingSCUs()

    def test_3(self):
        _lu_ = {'_id_':[],'q':[],'my_scu':[],'my_source':[],'my_summary':[],'my_excerpts':[]}
        _lu_['_id_']       .append('id_1')
        _lu_['q']          .append('What is your favorite color?')
        _lu_['my_scu']     .append('red')
        _lu_['my_source']  .append('wikipedia')
        _lu_['my_summary'] .append('Red is the color of blood, roses, and scarlet. It is also the color of many fruits, such as strawberries, raspberries, and blackberries.')
        _lu_['my_excerpts'].append('color of blood, roses ... strawberries, raspberries')
        _lu_['_id_']       .append('id_1')
        _lu_['q']          .append('What is your favorite color?')
        _lu_['my_scu']     .append('red')
        _lu_['my_source']  .append('otherpedia')
        _lu_['my_summary'] .append('Red is just another color.')
        _lu_['my_excerpts'].append('just another color')
        df = pd.DataFrame(_lu_)
        scuep = SCUEditorPanel(self.rt_self, df, 'id_1', '_id_', 'q', 'my_scu', 'my_source', 'my_summary', 'my_excerpts')
        scuep.createHTMLForMissingSCUs()

    def test_4(self):
        _lu_ = {'_id_':[],'q':[],'my_scu':[],'my_source':[],'my_summary':[],'my_excerpts':[]}
        _lu_['_id_']       .append('id_x')
        _lu_['q']          .append('What is your favorite color?')
        _lu_['my_scu']     .append('red')
        _lu_['my_source']  .append(10)
        _lu_['my_summary'] .append('Red is the color of blood, roses, and scarlet. It is also the color of many fruits, such as strawberries, raspberries, and blackberries.')
        _lu_['my_excerpts'].append('color of blood, roses ... strawberries, raspberries')
        _lu_['_id_']       .append('id_x')
        _lu_['q']          .append('What is your favorite color?')
        _lu_['my_scu']     .append('red')
        _lu_['my_source']  .append(20)
        _lu_['my_summary'] .append('Red is just another color.')
        _lu_['my_excerpts'].append('just another color')
        df = pd.DataFrame(_lu_)
        scuep = SCUEditorPanel(self.rt_self, df, 'id_x', '_id_', 'q', 'my_scu', 'my_source', 'my_summary', 'my_excerpts')
        scuep.createHTMLForMissingSCUs()

