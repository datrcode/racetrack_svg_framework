import polars as pl
import rtsvg
rt = rtsvg.RACETrack()
df = pl.DataFrame({'fm':['a','b','c'],'to':['b','c','a']})
_link_ = rt.link(df, [('fm','to')])
_link_._repr_svg_()
