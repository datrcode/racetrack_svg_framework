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
import unittest
import pandas as pd
import polars as pl
import numpy as np
import random
import copy
import string

from rtsvg import *

class Testrt_text_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()
        # Source:  Project Gutenberg, War and Peace [Leo Tolstoy]
        self._text = self.rt_self.textJoinNewLinesBetter("""It was in July, 1805, and the speaker was the well-known Anna Pávlovna
        Schérer, maid of honor and favorite of the Empress Márya Fëdorovna.
        With these words she greeted Prince Vasíli Kurágin, a man of high
        rank and importance, who was the first to arrive at her reception. Anna
        Pávlovna had had a cough for some days. She was, as she said, suffering
        from la grippe; grippe being then a new word in St. Petersburg, used
        only by the elite.""")


    def test_textCitationElements(self):
        answers = {'[1-3]':[1,2,3], 
                   '[2,3,4]':[2,3,4], 
                   '[1 - 6]':[1,2,3,4,5,6], 
                   '[ 1-6 ]':[1,2,3,4,5,6], 
                   '[2 , 5 , 6]':[2,5,6], 
                   '[1]':[1], 
                   '[1, 2-6 , 8]':[1,2,3,4,5,6,8], 
                   '[1, 2, 3, 5 ]':[1,2,3,5], 
                   '[3]':[3], 
                   '[2]':[2], 
                   '[4]':[4], 
                   '[1-3,4 , 8, 10]':[1,2,3,4,8,10], 
                   '[1,2,3]':[1,2,3], 
                   '[1,5,6]':[1,5,6],}
        for s in answers.keys():
            to_compare = self.rt_self.textCitationElements(s)
            self.assertEqual(sorted(answers[s]), sorted(to_compare))

    def test_textSubtractSpans(self):
        _answers_ = [
            ([(10,5), (15,5)],               [(12,5)],                 [(10, 2), (17, 3)]),
            ([(5,20)],                       [(10,2), (15,2), (25,5)], [(5, 5), (12, 3), (17, 8)]),
            ([(5,5), (15,5), (25,5)],        [(0,100)],                []),
            ([(0,2), (5,5), (15,5), (25,5)], [(1,28)],                 [(0, 1), (29, 1)]),
        ]
        for _spans_, _subtrahend_, _answer_ in _answers_:
            self.assertEqual(self.rt_self.textSubtractSpans(_spans_, _subtrahend_), _answer_)

    def test_textCitationSpans(self):
        examples = {
            'This is a basic citation. [1]':[(26,3)],
            'This is a basic citation. [1][1][1]':[(26,3),(29,3),(32,3)],
            '[1] This is a [1] basic citation. [1] [1] [1]':[(0,3),(14,3),(34,3),(38,3),(42,3)],
            'More [2] complicated one here [3].':[(5,3),(30,3)],
            'A comma listed [1,2,3] one.':[(15,7)],
            'At the end. [1,5,6]':[(12,7)],
            'With spaces [1, 2, 3, 5 ].':[(12,13)],
            'With [1-3] ranges in them [4].':[(5,5),(26,3)],
            'with spaces and commas and stuff [1-3,4 , 8, 10]':[(33,15)],
            '[1, 2-6 , 8] at the beginning and the ending. [ 1-6 ]':[(0,12),(46,7)],
            '[1 - 6][2,3,4][2 , 5 , 6]':[(0,7),(7,7),(14,11)],
            '[1 not 2] not this one.':[],
            '[-12] or this one either [-13] [14-]':[],
            '[1 -- 3] [1,-3], [-1], [-1 --,-- 3[]] several bad examples':[],
        }
        for s in examples.keys():
            _answer_ = sorted(examples[s])
            _spans_  = sorted(self.rt_self.textCitationSpans(s))
            self.assertEqual(_spans_, _answer_)

    def test_editDistance(self):
        _tuples_ = [('abc','abcd',1),
                    ("kitten", "sitting", 3), # https://en.wikipedia.org/wiki/Levenshtein_distance
                    ]
        for _tuple_ in _tuples_:
            self.assertEqual(self.rt_self.editDistance(_tuple_[0], _tuple_[1]), _tuple_[2])

    def test_textAggregateSpans(self):
        _spans_ = [(2,3), (4,10), (20,5),(40,5),(25,3)]
        self.assertEqual(self.rt_self.textAggregateSpans(_spans_), [(2, 12), (20, 8), (40, 5)])
        _spans_ = [(1,3), (10,5), (10,20), (35,5)]
        self.assertEqual(self.rt_self.textAggregateSpans(_spans_), [(1, 3), (10, 20), (35, 5)])
        _spans_ = [(1,3), (10,20),(10, 5), (35,5)]
        self.assertEqual(self.rt_self.textAggregateSpans(_spans_), [(1, 3), (10, 20), (35, 5)])

        def spansEquals(spans_1, spans_2):
            max_index = spans_1[0][0] + spans_1[0][1]
            for i in range(len(spans_1)):
                if spans_1[i][0] + spans_1[i][1] > max_index:
                    max_index = spans_1[i][0] + spans_1[i][1]
            for i in range(len(spans_2)):
                if spans_2[i][0] + spans_2[i][1] > max_index:
                    max_index = spans_2[i][0] + spans_2[i][1]
            bit_array_1 = [0] * (max_index+10)
            bit_array_2 = [0] * (max_index+10)
            for i in range(len(spans_1)):
                for j in range(spans_1[i][0],spans_1[i][0]+spans_1[i][1]):
                    bit_array_1[j] = 1
            for i in range(len(spans_2)):
                for j in range(spans_2[i][0],spans_2[i][0]+spans_2[i][1]):
                    bit_array_2[j] = 1
            for i in range(len(bit_array_1)):
                if bit_array_1[i] != bit_array_2[i]:
                    return False
            return True
        for i in range(100):
            num_spans = random.randint(1,100)
            spans_a   = []
            for j in range(num_spans): spans_a.append([random.randint(0,200),random.randint(1,10)])
            leftovers = copy.deepcopy(spans_a)
            spans_b   = []
            while len(leftovers) > 0:
                _choice_ = random.randint(0,len(leftovers)-1)
                spans_b.append(leftovers.pop(_choice_))
            self.assertTrue(spansEquals(spans_a,spans_b))

    def test_longestCommonSubstring(self):
        s1 = "once upon a time... there was a programmer... named john..."
        s2 = "there was a programmeronce upon a timenamed john"
        _len_, _i_, _j_ = self.rt_self.longestCommonSubstring(s1, s2)
        self.assertEqual(s1[_i_:_i_+_len_], s2[_j_:_j_+_len_])

        my_results, s1_leftovers, s2_leftovers = self.rt_self.iterativelyFindAllCommonSubstrings(s1, s2, min_length=4)
        for _tuple_ in my_results:
            _len_, _i_, _j_ = _tuple_
            self.assertEqual(s1[_i_:_i_+_len_], s2[_j_:_j_+_len_])
        self.assertEqual(s1_leftovers, '... ... ...') 
        self.assertEqual(s2_leftovers, '')

        other_my_results, other_s1_leftovers, other_s2_leftovers = self.rt_self.iterativeLongestCommonSubstrings(s1, s2, min_length=4)
        self.assertEqual(my_results,   other_my_results) 
        self.assertEqual(s1_leftovers, other_s1_leftovers) 
        self.assertEqual(s2_leftovers, other_s2_leftovers)

        my_longer  = 'abcdefxyzmno'
        my_shorter = 'abcdefghimno'
        my_results, my_longer_leftovers, my_shorter_leftovers = self.rt_self.iterativelyFindAllCommonSubstrings(my_longer, my_shorter, min_length=2)
        for _tuple_ in my_results:
            _len_, _i_, _j_ = _tuple_
            self.assertEqual(my_longer[_i_:_i_+_len_], my_shorter[_j_:_j_+_len_])
        self.assertEqual(my_longer_leftovers,  'xyz')
        self.assertEqual(my_shorter_leftovers, 'ghi')

        other_my_results, other_longer_leftovers, other_shorter_leftovers = self.rt_self.iterativeLongestCommonSubstrings(my_longer, my_shorter, min_length=2)
        self.assertEqual(my_results,   other_my_results) 
        self.assertEqual(my_longer_leftovers,  other_longer_leftovers) 
        self.assertEqual(my_shorter_leftovers, other_shorter_leftovers)

    def test_textExtractEntities(self):
        rttb        = self.rt_self.textBlock(self._text, word_wrap=True, txt_h=18, x_ins=3)
        _entities   = self.rt_self.textExtractEntities(self._text)
        _highlights = {}
        for _tup in _entities: _highlights[(_tup[2],_tup[3])] = _tup[1]
        rttb.highlights(_highlights, opacity=0.4)

    def test_textExtractSentences(self):
        rttb        = self.rt_self.textBlock(self._text, word_wrap=True, txt_h=18, x_ins=3)
        _sentences  = self.rt_self.textExtractSentences(self._text)
        _highlights = {}
        for _tup in _sentences: _highlights[(_tup[1],_tup[2])] = _tup[0]
        rttb.highlights(_highlights, opacity=0.4)

    def test_positionalDataFrame(self):
        rttb = self.rt_self.textBlock(self._text, w=512, word_wrap=True, txt_h=18)
        df  = rttb.positionalDataFrame()
        _df = df.query('type == "para" and num >= 2 and num <= 3')
        rttb.renderDataFrame(_df)

    def REMOVED_test_textCompareSummaries(self):
        import tensorflow as tf
        import tensorflow_hub as hub
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)
        def embed(input): return model(input)
        _text      = '''This is a test.  This is only a test.  Really, only a test.'''
        _summary_1 = '''Several test messages.'''
        _summary_2 = '''Three messages about a test.''' 
        self.rt_self.textCompareSummaries(_text, {'Summary 1':_summary_1, 'Summary 2':_summary_2}, embed_fn=embed, opacity=0.5)

    def test_pixelRepr(self):
        rttb = self.rt_self.textBlock(self._text,word_wrap=True,txt_h=12,w=384)
        lu   = {}
        lu[(   0,  31)]           = '#ff0000'
        lu[( 624, 841)]           = '#00ff00'
        lu[(3717,3838)]           = '#0000ff'
        lu[r'([Mm]arch(es|ers){0,1})']  = '#ff00ff'
        self.rt_self.tile([rttb.highlights(lu), '<svg x="0" y="0" width="10" height="10"></svg>', 
                           rttb.pixelRepr(lu),
                           rttb.pixelRepr(lu, draw_context=True),
                           rttb.pixelRepr(lu, draw_context=True, context_opacity=1.0)])._repr_svg_()

    def REMOVED_test_textRank(self):
        df_tr = self.rt_self.textRank(self._text)

    def REMOVED_test_textLexRank(self):
        self.rt_self.textLexRank(self._text, self.rt_self.textCreateEmbedder())

    def test_highlightComparisons(self):
        def junk(min_words=16, max_words=128):
            def makeWord(min_chars=3, max_chars=8):
                return ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(min_chars,max_chars)))
            words = []
            for i in range(random.randint(min_words, max_words)):
                words.append(makeWord())
            return ' '.join(words)
        my_markers = ['marker1', 'marker2', 'marker3', 'marker4', 'this is a longer marker that is meant to go across lines']
        my_colors  = ['#ff0000', '#00ff00', '#0000ff', '#ffff00', '#a0a0a0']
        all_lus    = {my_markers[0]: my_colors[0], my_markers[1]: my_colors[1], my_markers[2]: my_colors[2], my_markers[3]: my_colors[3], my_markers[4]: my_colors[4]}
        my_markups = {'Ex 1': {my_markers[0]: my_colors[0], my_markers[1]: my_colors[1]},
                    'Ex 2': {my_markers[2]: my_colors[2], my_markers[0]: my_colors[0]},
                    'Ex 3': {my_markers[0]: my_colors[0], my_markers[1]: my_colors[1], my_markers[3]: my_colors[3]},
                    'Ex 4': {my_markers[2]: my_colors[2], my_markers[3]: my_colors[3]},
                    'Ex 5': {my_markers[4]: my_colors[4]}}
        def makePassage(markers):
            _txt_= []
            _txt_.append(junk() + ' ' +random.choice(markers) + ' ' + junk())
            _txt_.append(random.choice(markers) + ' ' + junk(10, 12) + ' ' + random.choice(markers))
            _txt_.append(junk(32,64))
            _txt_.append(junk() + ' ' +random.choice(markers) + ' ' + junk(10, 12) + ' ' + random.choice(markers) + ' ' + junk(10, 12) + ' ' + random.choice(markers))
            _txt_.append(junk(32,256))
            _txt_.append(random.choice(markers) + ' ' + junk(10, 12) + ' ' + random.choice(markers))
            _txt_.append(junk(32,256))
            _txt_.append(random.choice(markers) + ' ' + junk(10, 12) + ' ' + random.choice(markers))
            _txt_.append(junk(32,40))
            _txt_.append(random.choice(markers) + ' ' + junk(10, 12) + ' ' + random.choice(markers))
            return '\n'.join(_txt_)

        passage = makePassage(my_markers)
        _tb_          = self.rt_self.textBlock(passage, word_wrap=True, w=300)
        my_svg_dict   = _tb_.highlightsComparison(my_markups, y_keep=4.0)
        _tb2_         = self.rt_self.textBlock(passage, word_wrap=True, w=700)
        my_svg_dict   = _tb2_.highlightsComparison(my_markups)

if __name__ == '__main__':
    unittest.main()

