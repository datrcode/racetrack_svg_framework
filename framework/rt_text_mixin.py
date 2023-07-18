# Copyright 2023 David Trimm
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

import pandas as pd
import numpy as np
import re

from rt_component import RTComponent

from shapely.geometry import Polygon, MultiPolygon

__name__ = 'rt_text_mixin'

#
# Abstraction for Text
#
class RTTextMixin(object):
    #
    # Constructor
    # 
    def __text_mixin_init__(self):
        self.spacy_loaded_flag = False

    #
    # textBlock() - render a textblock and track positional information of characters and words.
    #
    def textBlock(self,
                  txt,
                  txt_h=14,
                  line_space_px=3,
                  word_wrap=False,
                  w=512,
                  x_ins=5,
                  y_ins=3):
        
        svg,x,y,line = '',x_ins,y_ins+txt_h,''
        last_was_space = True
        line_lu = {}

        for i in range(0,len(txt)):
            c = txt[i]
            if c == '\n':
                # CODE BLOCK B
                line_lu[y]     = line + '\n'
                x              = x_ins
                y             += txt_h + line_space_px
                line           = ''
                last_was_space = True
                if word_wrap:
                    y += 3*line_space_px
            elif word_wrap == False:
                # CODE BLOCK A
                line          += c
                x             += self.textLength(c,txt_h)
            else:
                if last_was_space and self.__whitespace__(c) == False:
                    j,x_j = i+1,x+self.textLength(c,txt_h)
                    while j < len(txt) and self.__whitespace__(txt[j]) == False:
                        x_j += self.textLength(txt[j],txt_h)
                        j   += 1
                    if   x_j > (w-x_ins) and x != x_ins: # new word exceeds the maximum width / start new line
                        # CODE BLOCK B
                        line_lu[y]     = line
                        x              = x_ins
                        y             += txt_h + line_space_px
                        line           = ''
                        last_was_space = True
                        # CODE BLOCK A
                        line          += c
                        x             += self.textLength(c,txt_h)
                    elif (x_j > (w-x_ins) and x == x_ins) or ((x_j - x) >= (w-x_ins)): # a chunk of text is too long to fit on a line
                        # CODE BLOCK B-mod
                        line_lu[y]     = line
                        x              = x_ins
                        line           = ''
                        last_was_space = False # MOD HERE
                        # CODE BLOCK A
                        line          += c
                        x             += self.textLength(c,txt_h)                        
                    else:                        # fine to add the word
                        # CODE BLOCK A
                        line          += c
                        x             += self.textLength(c,txt_h)
                else:
                        # CODE BLOCK A
                        line          += c
                        x             += self.textLength(c,txt_h)
                if self.__whitespace__(c):
                    last_was_space = True

        # If there's a left over line, add it here...
        if len(line) > 0:
            line_lu[y]   = line
            y           += txt_h + line_space_px

        # Just render as is...
        _ignore_ = """
        orig_to_xy,i = {},0
        for k in line_lu:
            line = line_lu[k]
            svg += self.svgText(line_lu[k], x_ins, k, txt_h, color='#b0b0b0')
            x = x_ins
            for c in line:
                orig_to_xy[i] = (x,k)
                x += self.textLength(c, txt_h)
                i += 1
        """

        # Render each word individually -- w/ absolute coordinates
        orig_to_xy,i = {},0
        for y in line_lu:
            line = line_lu[y]
            x,j  = x_ins,0
            while j < len(line):
                c = line[j]
                svg += self.svgText(c, x, y, txt_h)
                orig_to_xy[i] = (x,y)
                j += 1
                i += 1
                x += self.textLength(c, txt_h)
        y += txt_h + line_space_px

        # Calculate geom_to_word
        geom_to_word = {}
        i,last_was_space = 0,True
        _dn = 4 # downward shift...
        while i < len(txt):
            if self.__whitespace__(txt[i]) or self.__punctuation__(txt[i]):
                last_was_space = True
                i += 1
            else:
                if last_was_space and self.__whitespace__(txt[i]) == False and self.__punctuation__(txt[i]) == False:
                    i0 = i
                    while i < len(txt)                           and \
                          self.__whitespace__ (txt[i])  == False and \
                          self.__punctuation__(txt[i])  == False:
                        i += 1
                        i1 = i
                    x0,y0 =  orig_to_xy[i0]
                    x1,y1 =  orig_to_xy[i1-1]
                    x1    += self.textLength(txt[i-1],txt_h)
                    _polygon = Polygon([[x0,y0+line_space_px+_dn], 
                                        [x1,y1+line_space_px+_dn], 
                                        [x1,y1-txt_h+_dn], 
                                        [x0,y1-txt_h+_dn]])
                    geom_to_word[_polygon] = txt[i0:i1]
                last_was_space = False
                i = i1

        bounds = (0,0,w,y-txt_h+y_ins)
        return RTTextBlock(self, txt, txt_h, line_space_px, word_wrap, w, x_ins, y_ins, svg, bounds, geom_to_word, orig_to_xy)

    # Is character whitespace?
    def __whitespace__ (self, c):
        return c == ' ' or c == '\t' or c == '\n'
    
    # Is character punctuation?
    def __punctuation__(self, c):
        _str = '''!.?,[]{}:;`~%^&*()-_+='"<>/\\'''
        return c in _str
    
    #
    # textJoinLines() - join lines together and remove extra spaces.
    # - expect that this is a utility to call before textBlock()
    #
    def textJoinNewLines(self, txt):
        joined = ' '.join(txt.split('\n'))
        while len(joined) > 0 and joined[0] == ' ':
            joined = joined[1:]
        while len(joined) > 0 and joined[-1] == ' ':
            joined = joined[:-1]
        words = joined.split(' ')
        wout_blanks = []
        for word in words:
            if len(word) > 0:
                wout_blanks.append(word)
        return ' '.join(wout_blanks)
    
    #
    # textJoinNewLinesBetter() - keep newlines (if single) intact...
    # - more closely mirrors the (de facto) standard of using double line returns 
    #   to separate paragraphs
    # - expect that this is a utility to call before textBlock()
    #
    def textJoinNewLinesBetter(self, txt):
        re_match = re.findall(r'([\n]{2,})',txt)
        if re_match is None:
            return self.joinNewLines(txt)
        else:
            i,_ret = 0,''
            for _match in re_match:
                j = txt.find(_match,i)
                _ret += self.textJoinNewLines(txt[i:j])
                for k in range(len(_match)-1):
                    _ret += '\n'
                i = j+len(_match)
            _ret += self.textJoinNewLines(txt[i:])
            return _ret
    
    #
    # maxLinePixels() - split a string by new line characters, then determine
    # the maximum line length (in pixels).
    #
    def textMaxLinePixels(self, txt, txt_h=14):
        _max = 0
        lines = txt.split('\n')
        for _line in lines:
            _len = self.textLength(_line, txt_h)
            _max = max(_len,_max)
        return _max + 6
    
    #
    # textExtractSentences() - extract sentences
    #
    # _tups = textExtractSentences(_str_)
    # _just_the_sentences_as_array = list(list(zip(*_tups))[0])
    #
    def textExtractSentences(self,
                             txt):
        import nltk
        tokens,sentences = nltk.sent_tokenize(txt),[]
        if len(tokens) > 0:
            i = txt.index(tokens[0])
        for _token in tokens:
            i = txt.index(_token,i)
            sentences.append((_token, i, i + len(_token)))
            i += len(_token)
        return sentences

    #
    # textExtractEntities() - extract entities.
    #
    def textExtractEntities(self, 
                            txt, 
                            algo='spacy'):
        if algo == 'spacy':
            return self.__textExtractEntitiesSpacy__(txt)
        else:
            raise Exception(f'RACETrack.textExtractEntities() - unknown algorithm "{algo}"')

    #
    # __extractEntitiesSpacy__() - extract entities using SpaCy.
    #
    def __textExtractEntitiesSpacy__(self,txt):
        if self.spacy_loaded_flag == False:
            import spacy
            self.nlp_spacy = spacy.load('en_core_web_sm')
            self.spacy_loaded_flag = True
        doc = self.nlp_spacy(txt)
        ret = []
        for entity in doc.ents:
            ret.append((entity.text, entity.label_, entity.end_char - len(entity.text), entity.end_char))
        return ret
    
    #
    # textCompareSummaries()
    #
    def textCompareSummaries(self, 
                             text_main,
                             text_summaries,
                             methodology      = "sentence_embeddings",
                             embed_fn         = None,                    # For "sentence_embeddings" methodology
                             main_txt_h       = 14,
                             summary_txt_h    = 16,
                             spacing          = 16,
                             opacity          = 0.8,
                             w                = 1280):
        if type(text_summaries) == str:
            text_summaries = {'Default':text_summaries}
        if   methodology == "sentence_embeddings":
            return self.__textCompareSummaries__sentence_embeddings__(text_main, text_summaries, embed_fn, main_txt_h, summary_txt_h, spacing, opacity, w)
        elif methodology == "bert_top_n":
            return self.__textCompareSummaries__bert_top_n__(text_main, text_summaries, main_txt_h, summary_txt_h, spacing, opacity, w)
        else:
            raise Exception(f'RACETrack.textCompareSummaries() - unknown methodology "{methodology}"')

    #
    # __textCompareSummaries__sentence_embeddings__()
    #
    def __textCompareSummaries__sentence_embeddings__(self,
                                                      text_main,
                                                      text_summaries,
                                                      embed_fn,
                                                      main_txt_h,
                                                      summary_txt_h,
                                                      spacing,
                                                      opacity,
                                                      w):
        import umap
        from sklearn.preprocessing import StandardScaler

        # Geometry
        main_w        = summary_w = (w - spacing)/2

        # Colors
        _colors   = self.co_mgr.brewerColors('qualitative', 12) # max available qualitative colors
        _colors_i = 0

        # Text Blocks
        main_rttb     = self.textBlock(text_main, txt_h=main_txt_h, w=main_w, word_wrap=True)
        summary_rttbs = []
        summary_rttb_to_desc = {}

        for _summary_desc in text_summaries:
            _summary = text_summaries[_summary_desc]
            _rttb    = self.textBlock(_summary, txt_h=summary_txt_h, w=summary_w, word_wrap=True)
            summary_rttbs.append(_rttb)
            summary_rttb_to_desc[_rttb] = _summary_desc
        
        # Embeddings
        main_sentences            = self.textExtractSentences(text_main)
        main_sentences_only       = list(list(zip(*main_sentences))[0])
        main_sentences_embeddings = embed_fn(main_sentences_only)
        main_sentence_colors      = {} # [sentence_index] = hex-color-string

        for_umap_source           = []
        for_umap_num              = []
        for_umap_sentence_num     = []
        for_umap_sentence         = []

        for_umap_embeddings       = []
        for_umap_embeddings.extend(main_sentences_embeddings)

        for i in range(0,len(main_sentences_only)):
            for_umap_source.      append('main')
            for_umap_num.         append(0)
            for_umap_sentence_num.append(i)
            for_umap_sentence.    append(main_sentences_only[i])

        # For every summary supplied...
        summary_dots_lu,summary_highlights,min_dot,max_dot,summary_num = {},{},None,None,0
        summary_highlights_lu = {} # [summary][summary_sentence_index] = best_found_main_sentence_index
        summary_num_to_desc   = {}
        for _summary_desc in text_summaries:
            _summary                      = text_summaries[_summary_desc]
            _summary_sentences            = self.textExtractSentences(_summary)
            _summary_sentences_only       = list(list(zip(*_summary_sentences))[0])
            _summary_sentences_embeddings = embed_fn(_summary_sentences_only)

            for_umap_embeddings.extend(_summary_sentences_embeddings)
            for i in range(0,len(_summary_sentences_only)):
                for_umap_source.       append('summary')
                for_umap_num.          append(summary_num)
                for_umap_sentence_num. append(i)
                for_umap_sentence.     append(_summary_sentences_only[i])

            summary_dots = []
            summary_highlights[_summary]    = {}
            summary_highlights_lu[_summary] = {}
            # For every sentence in this summary...
            for i in range(0,len(_summary_sentences)):
                _embedding = _summary_sentences_embeddings[i]
                dots = []
                best_dot,best_dot_main_sentence_index = None,None

                # Loop over the main sentence embeddings -- record both all the dot products as well as the best main sentence match
                for j in range(0,len(main_sentences)):
                    # Get the main sentence embedding
                    _main_embedding = main_sentences_embeddings[j]
                    # Compute the dot product between the main sentence and this specific summaries sentence
                    _dot = float(np.tensordot(_embedding, _main_embedding, axes=1)) # Works with Google's Universal Sentence Embedder...
                    if min_dot is None or min_dot > _dot:
                        min_dot = _dot
                    if max_dot is None or max_dot < _dot:
                        max_dot = _dot                        
                    dots.append(_dot)

                    # Record the best dot found so far (vs the main sentences)
                    if best_dot is None or best_dot < _dot: # Looking for the largest based on some testing...
                        best_dot                     = _dot
                        best_dot_main_sentence_index = j

                summary_dots.append(dots)

                # Try to highlight (if we found something -- how could we not? ... and if we have any colors left)
                if best_dot_main_sentence_index is not None:
                    summary_highlights_lu[_summary][i] = best_dot_main_sentence_index
                    if   best_dot_main_sentence_index in main_sentence_colors.keys(): # Already found!
                        beg_end = (_summary_sentences[i][-2], _summary_sentences[i][-1])
                        summary_highlights[_summary][beg_end] = main_sentence_colors[best_dot_main_sentence_index]
                    elif _colors_i < len(_colors):                                    # Still Have Colors Left!
                        main_sentence_colors[best_dot_main_sentence_index] = _colors[_colors_i]
                        _colors_i += 1
                        beg_end = (_summary_sentences[i][-2], _summary_sentences[i][-1])
                        summary_highlights[_summary][beg_end] = main_sentence_colors[best_dot_main_sentence_index]
                    else:                                                             # No Colors Left :(
                        pass

            summary_dots_lu     [_summary]      =  summary_dots
            summary_num_to_desc [summary_num]   =  _summary_desc
            summary_num                        +=  1

    
        # Create the main highlights
        main_highlights = {}
        for i in main_sentence_colors:
            _tup = main_sentences[i]
            main_highlights[(_tup[-2],_tup[-1])] = main_sentence_colors[i]

        # Renderings & Compositions
        summary_tiles = []
        for _rttb in summary_rttbs:
            _desc = summary_rttb_to_desc[_rttb]
            summary_tiles.append(f'<svg x="0" y="0" width="{summary_w}" height="{24}">' + \
                                 self.svgText(_desc, 3, 20, txt_h=19, color='#ffffff') + '</svg>')
            summary_tiles.append(_rttb.highlights(summary_highlights[_rttb.txt], opacity=opacity))
            summary_tiles.append(f'<svg x="0" y="0" width="{spacing}" height="{spacing}"> </svg>') # Spacers
            summary_tiles.append(self.__textDotProductHeatMap__(summary_dots_lu[_rttb.txt], min_dot, max_dot, 
                                                                summary_highlights_lu[_rttb.txt], main_sentence_colors))
            summary_tiles.append(f'<svg x="0" y="0" width="{spacing}" height="{spacing}"> </svg>') # Spacers
            summary_tiles.append(self.__textDotProductHistogram__(summary_dots_lu[_rttb.txt],
                                                                  summary_highlights_lu[_rttb.txt], main_sentence_colors, 
                                                                  summary_w))
            summary_tiles.append(f'<svg x="0" y="0" width="{spacing}" height="{spacing}"> </svg>') # Spacers

        # Create the UMAP
        umap_reducer               = umap.UMAP()
        scaled_for_umap_embeddings = StandardScaler().fit_transform(for_umap_embeddings) 
        umap_embedding             = umap_reducer.fit_transform(scaled_for_umap_embeddings)
        umap_xs,umap_ys            = [],[]
        for i in range(0,len(umap_embedding)):
            umap_xs.append(umap_embedding[i][0])
            umap_ys.append(umap_embedding[i][1])
        umap_color,umap_size = [],[]
        for i in range(0,len(umap_embedding)):
            sentence_num = for_umap_sentence_num[i]
            # Dot Color...
            if for_umap_source[i] == 'main':
                if sentence_num in main_sentence_colors.keys():
                    umap_color.append(main_sentence_colors[sentence_num])
                else:
                    umap_color.append('#808080')
            else:
                _summary_desc = summary_num_to_desc[for_umap_num[i]]
                _summary      = text_summaries[_summary_desc]
                if for_umap_sentence_num[i] in summary_highlights_lu[_summary].keys():
                    closest_main_i = summary_highlights_lu[_summary][for_umap_sentence_num[i]]
                    if closest_main_i in main_sentence_colors.keys():
                        umap_color.append(main_sentence_colors[closest_main_i])
                    else:
                        umap_color.append('#808080')    
                else:
                    umap_color.append('#808080')
            # Dot Size...
            if   sentence_num == 0:
                umap_size.append(2.0)
            elif sentence_num == 1:
                umap_size.append(1.8)
            elif sentence_num == 2:
                umap_size.append(1.5)
            elif sentence_num == 3:
                umap_size.append(1.3)
            else:
                umap_size.append(1.0)

        df_umap = pd.DataFrame({'sentence':    for_umap_sentence,
                                'setence_num': for_umap_sentence_num,
                                'source_type': for_umap_source,
                                'source_num':  for_umap_num,
                                'color':       umap_color,
                                'size':        umap_size,
                                'x_umap':      umap_xs,
                                'y_umap':      umap_ys})
        
        def _mydotshape_(_df, _k, _x, _y, _local_dot_w, _color, _opacity):
            source_type = _df['source_type'].iloc[0]
            source_num  = _df['source_num'] .iloc[0]
            if source_type == 'main':
                return 'x'
            else:
                if   source_num == 0:
                    return 'square'
                elif source_num == 1:
                    return 'ellipse'
                else:
                    return 'triangle'

        # Skip the umap for now... doesn't add anything...
        # summary_tiles.append(self.xy(df_umap, x_field='x_umap', y_field='y_umap', color_by='color', count_by='size', dot_size='vary', dot_shape=_mydotshape_, draw_labels=False))

        # Compose the summary side
        tile_composition = self.tile(summary_tiles, horz=False)

        # Compose the total
        composition = [tile_composition,
                       f'<svg x="0" y="0" width="{spacing}" height="{spacing}"> </svg>',
                       main_rttb.highlights(main_highlights, opacity=opacity)]
        
        return self.tile(composition)
    
    #
    # __textDotProductXYDataFrame__
    #
    def __textDotProductHistogram__(self, arr, _sentence_index_to_main_index, _main_index_colors, _w):
        _xs,_ys,_colors,_groups = [],[],[],[]
        for _group in range(0,len(arr)):
            _copy = sorted(np.array(arr[_group]),reverse=True)
            for x in range(0,len(_copy)):
                y = _copy[x]
                _xs.     append(x)
                _ys.     append(y)
                _groups. append(_group)
                _color = '#000000'
                if _group in _sentence_index_to_main_index.keys():
                    main_index = _sentence_index_to_main_index[_group]
                    if main_index in _main_index_colors.keys():
                        _color = _main_index_colors[main_index]
                _colors.append(_color)
        _df = pd.DataFrame({'x':_xs,'y':_ys,'color':_colors,'group':_groups})
        return self.xy(_df, x_field='x', y_field='y', color_by='color', line_groupby_field='group', line_groupby_w=2.0, dot_size=None, w=_w, h=128)

    #
    # _textDotProductHeatMap__():  Make a simplified heatmap
    #
    def __textDotProductHeatMap__(self, arr, _min, _max, _sentence_index_to_main_index, _main_index_colors):
        if _min == _max:
            _max = _min + 1
        x_tiles,y_tiles = len(arr[0]),len(arr)
        tile_w, tile_h  = 12,12
        svg = f'<svg x="0" y="0" width="{x_tiles*tile_w + 3*tile_w}" height="{y_tiles*tile_h}">'
        for y in range(0,len(arr)):
            for x in range(0,len(arr[y])):
                _value = arr[y][x]
                # _color = self.co_mgr.spectrumAbridged(_value, _min, _max)
                _gray    = min(255, int(255 * (_value - _min)/(_max - _min)))
                _color   = f'#{_gray:02x}{_gray:02x}{_gray:02x}'
                svg += f'<rect x="{x*tile_w}" y="{y*tile_h}" width = "{tile_w}" height="{tile_h}" fill="{_color}" />'
            if y in _sentence_index_to_main_index.keys():
                main_index = _sentence_index_to_main_index[y]
                if main_index in _main_index_colors.keys():
                    _color = _main_index_colors[main_index]
                    svg += f'<rect x="{x_tiles*tile_w + tile_w}" y="{y*tile_h}" width = "{2*tile_w}" height="{tile_h}" fill="{_color}" />'
        svg += '</svg>'
        return svg
    
    #
    # __textCompareSummaries__bert_top_n__():  Compare via top-n bert placements
    #
    def __textCompareSummaries__bert_top_n__(self,
                                             text_main, 
                                             text_summaries, 
                                             main_txt_h, 
                                             summary_txt_h, 
                                             spacing, 
                                             opacity, 
                                             w):
        # Geometry & Parameter Evaluation
        main_w = summary_w = (w - spacing)/2
        if type(text_summaries) == str:
            text_summaries = {'Default':text_summaries}

        # From the throwaway file "bert_mlm_example.ipynb"
        #
        # Modified From https://towardsdatascience.com/masked-language-modelling-with-bert-7d49793e5d2c
        #
        from transformers import BertTokenizer, BertForMaskedLM, TFBertForMaskedLM, AdamW
        import tensorflow as tf
        import torch

        mask_perc            = 0.75 # tutorial was 0.15
        epochs               = 100  # tutorial was less...
        tokenizer            = BertTokenizer.  from_pretrained('bert-base-cased')
        model                = BertForMaskedLM.from_pretrained('bert-base-cased')
        text_main_as_tokens  = tokenizer.tokenize(text_main, return_tensors='pt') # Newer version is just tokenizer(text) ... and returns the inputs structure below
        as_encoded           = tokenizer.encode(text_main)
        inputs = {'input_ids':     torch.Tensor([as_encoded]).long(),
                  'token_type_ids':torch.Tensor([np.zeros(len(as_encoded))]).long(),
                  'attention_mask':torch.Tensor([np.ones (len(as_encoded))]).long()}
        inputs['lm_labels'] = inputs['input_ids'].detach().clone()                                  # labels are just the original text...
        rand = torch.rand(inputs['input_ids'].shape)                                                # create random array of floats in equal dimension to input_ids
        mask_arr = (rand < mask_perc) * (inputs['input_ids'] != 101) * (inputs['input_ids'] != 102) # As an example of how to separate out those two token types
        selection = torch.flatten((mask_arr[0]).nonzero()).tolist()                                 # create selection from mask_arr
        inputs['input_ids'][0, selection] = 103                                                     # apply selection index to inputs.input_ids, adding MASK tokens
        outputs = model(**inputs)                                                                   # pass inputs as kwarg to model
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)                                                                            # and move our model over to the selected device
        model.train()                                                                               # activate training mode
        optim = AdamW(model.parameters(), lr=5e-5)                                                  # initialize optimizer
        # In the example, the input is broken into multiple versions -- based on each paragraph... not doing that here... but maybe we should
        for epoch in range(epochs):
            optim.zero_grad()                                                                       # initialize calculated gradients (from prev step)
            input_ids      = inputs['input_ids'].     to(device)                                    # move to gpu
            attention_mask = inputs['attention_mask'].to(device)
            lm_labels      = inputs['lm_labels'].     to(device)    
            outputs = model(input_ids, attention_mask=attention_mask, lm_labels=lm_labels)          # process
            loss = outputs[0]                                                                       # extract loss
            loss.backward()                                                                         # calculate loss for every parameter that needs grad update    
            optim.step()                                                                            # update parameters    
            if (epoch%10) == 0:                                                                     # print updated information
                print(f'Epoch {epoch:3}\t{loss.item()}')
        model.eval()                                                                                # Deactivate training mode

        # From https://mattmckenna.io/bert-off-the-shelf/#:~:text=BERT%20works%20by%20masking%20certain,ate%20the%20%5BMASK%5D%E2%80%9D.
        # - with a lot of modifications ... (e.g., for the GPU version...)
        def estimateKthPrediction(tokenized_inputs,
                                  k_max=20):
            # For every token, replace it with the mask... and then determine what kth it would have been... 
            _return        = []
            _mask_token    = tokenizer.encode(tokenizer.mask_token)[1]
            for i in range(len(tokenized_inputs)):
                tokenized_inputs_ith   = tokenized_inputs[:i] + [_mask_token] + tokenized_inputs[i+1:]
                outputs                = model(torch.Tensor([tokenized_inputs_ith]).long().to(device))
                top_k_indices          = tf.math.top_k(outputs[0].cpu().detach().numpy(), k_max).indices[0].numpy()
                jth_found              = None
                for j in range(len(top_k_indices[i])):
                    if top_k_indices[i][j] == tokenized_inputs[i]:
                        if jth_found is None:
                            jth_found = j
                if jth_found is None:
                    _return.append(k_max)
                else:
                    _return.append(jth_found)
            return _return

        # Align the tokenized version with original text -- list of tuples [(index, length), ...]
        def alignTokensWithTextAsTuples(_txt, _tokens):
            _return = []
            txt_i,token_i = 0,0
            while token_i < len(_tokens):
                _token = _tokens[token_i]
                if _token.startswith('##'):
                    _token = _token[2:]
                txt_i = _txt.index(_token,txt_i)
                _return.append((txt_i,len(_token)))
                token_i += 1
            return _return

        # Put the two last functions together for input highlights text input...
        def highlightsForText(_txt):
            _tokens      = tokenizer.tokenize(_txt, return_tensors='pt')
            _alignment   = alignTokensWithTextAsTuples(_txt, _tokens)
            _kth_predict = estimateKthPrediction(tokenizer.encode(_txt))[1:-1] # Crop out the begin/end tokens
            _highlights  = {}
            for i in range(len(_kth_predict)):
                if   _kth_predict[i] <= 1:
                    _co = None
                elif _kth_predict[i] <= 5:
                    _co = 'blue'
                elif _kth_predict[i] <= 10:
                    _co = 'yellow'
                elif _kth_predict[i] <  20:
                    _co = 'orange'
                else:
                    _co = 'red'
                if _co is not None:
                    i0_to_i1 = (_alignment[i][0], _alignment[i][0] + _alignment[i][1])
                    _highlights[i0_to_i1] = _co
            return _highlights

        rttb_main = self.textBlock(text_main, txt_h=main_txt_h, word_wrap=True, w=main_w)
        summary_tiles = []
        for summary_desc in text_summaries:
            _summary = text_summaries[summary_desc]
            rttb_summary = self.textBlock(_summary, txt_h=summary_txt_h, word_wrap=True, w=summary_w)
            summary_tiles.append(f'<svg x="0" y="0" width="{summary_w}" height="{24}">' + \
                                 self.svgText(summary_desc, 3, 20, txt_h=19, color='#ffffff') + '</svg>')
            summary_tiles.append(rttb_summary.highlights(highlightsForText(_summary), opacity=opacity))
            summary_tiles.append(f'<svg x="0" y="0" width="{spacing}" height="{spacing}"> </svg>') # Spacers

        return self.tile([self.tile(summary_tiles, horz=False),
                          f'<svg x="0" y="0" width="{spacing}" height="{spacing}"> </svg>',
                          rttb_main.highlights(highlightsForText(text_main), opacity=opacity)])

#
# RTTextBlock - instance of rendered text block
#
class RTTextBlock(object):
    #
    # Constructor
    #
    def __init__(self,
                 rt_self,           # Reference to parent class instance
                 txt,               # Original text string
                 txt_h,             # Text height in pixels
                 line_space_px,     # Pixel space between paragraphs
                 word_wrap,         # Word wrap flag
                 w,                 # Width of SVG results
                 x_ins,             # x insert left & right
                 y_ins,             # y insert top & bottom
                 svg,               # rendered svg (w/out svg begin/end wrapper)
                 bounds,            # Four tuple of x,y,w,h
                 geom_to_word,      # Shapely polygon to word
                 orig_to_xy):       # Original text index to xy-tuple
        self.rt_self        = rt_self
        self.txt            = txt
        self.txt_h          = txt_h
        self.line_space_px  = line_space_px
        self.word_wrap      = word_wrap
        self.w              = w
        self.x_ins          = x_ins
        self.y_ins          = y_ins
        self.svg            = svg
        self.bounds         = bounds
        self.geom_to_word   = geom_to_word
        self.orig_to_xy     = orig_to_xy
        
    #
    # spanGeometry() - return a polygon that covers a specified text span.
    #
    def spanGeometry(self, i, j):
        last_c = ' '
        if len(self.txt) > 0:
            last_c = self.txt[-1]

        if i >= len(self.txt):
            xy0    = self.orig_to_xy[len(self.txt)-1]
            xy0    = (xy0[0] + self.rt_self.textLength(last_c,self.txt_h),xy0[1])
        else:
            xy0    = self.orig_to_xy[i]
        if j >= len(self.txt):
            xy1    = self.orig_to_xy[len(self.txt)-1]
            xy1    = (xy1[0] + self.rt_self.textLength(last_c,self.txt_h),xy1[1])
        else:
            xy1    = self.orig_to_xy[j]

        _dn = 4 # downward shift...
        if     xy0[1]                                    == xy1[1]: # On same line...
            return Polygon([[xy0[0],xy0[1]+_dn],
                            [xy1[0],xy1[1]+_dn],
                            [xy1[0],xy1[1]-self.txt_h+_dn],
                            [xy0[0],xy0[1]-self.txt_h+_dn]
                            ])
        elif  (xy0[1] + self.txt_h + self.line_space_px) == xy1[1] and (xy1[0] < xy0[0]):
            _poly0 = Polygon([[xy0[0],              xy0[1]+_dn],
                              [self.w - self.x_ins, xy0[1]+_dn],
                              [self.w - self.x_ins, xy0[1]-self.txt_h+_dn],
                              [xy0[0],              xy0[1]-self.txt_h+_dn]])
            _poly1 = Polygon([[xy1[0],              xy1[1]+_dn],
                              [xy1[0],              xy1[1]-self.txt_h+_dn],
                              [self.x_ins,          xy1[1]-self.txt_h+_dn],
                              [self.x_ins,          xy1[1]+_dn]])
            return MultiPolygon([_poly0,_poly1])
        else: # Multiple lines...
            return Polygon([[xy0[0],              xy0[1]+_dn],
                            [self.x_ins,          xy0[1]+_dn],
                            [self.x_ins,          xy1[1]+_dn],
                            [xy1[0],              xy1[1]+_dn],
                            [xy1[0],              xy1[1]-self.txt_h+_dn],
                            [self.w - self.x_ins, xy1[1]-self.txt_h+_dn],
                            [self.w - self.x_ins, xy0[1]-self.txt_h+_dn],
                            [xy0[0],              xy0[1]-self.txt_h+_dn]
                            ])

    #
    # highlights() - highlight user-specified text.
    # - lu: either a [1] (i,j) tuple or [2] a regex string to either a [A] seven-character hex color string or a [B] string color
    #   - lu[(0,10)]            = '#ff0000'
    #   - lu['regex substring'] = '#000000'
    #   - lu['many']            = 'whatever' # any 'many' substrings will get colored with 'whatever' color lookup
    #
    def highlights(self, lu, opacity=1.0):
        svg_underlay = ''
        for k in lu:
            _co = lu[k]
            if _co.startswith('#') == False or len(_co) != 7: # If it's not a hex hash color string... then look it up...
                _co = self.rt_self.co_mgr.getColor(_co)
            if   type(k) == tuple:
                _poly = self.spanGeometry(k[0],k[1])
                svg_underlay += f'<path d="{self.rt_self.shapelyPolygonToSVGPathDescription(_poly)}" fill="{_co}" fill-opacity="{opacity}" />'
            elif type(k) == str:
                i = 0
                re_match = re.findall(k,self.txt)
                if re_match is not None:
                    i = 0
                    for _match in re_match:
                        i = self.txt.index(k,i)
                        j = i + len(_match)
                        _poly = self.spanGeometry(i,j)
                        svg_underlay += f'<path d="{self.rt_self.shapelyPolygonToSVGPathDescription(_poly)}" fill="{_co}" fill-opacity="{opacity}" />'
                        i += len(_match)
            else:
                raise Exception(f'RTTextBlock.highlights() - do not understand key value type {type(k)}')

        x,y,w,h = self.bounds
        _co     = self.rt_self.co_mgr.getTVColor('background','default')
        return f'<svg x="0" y="0" width="{w}" height="{h}">' + \
               f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" />' + \
               svg_underlay + \
               self.svg + \
               '</svg>'


    #
    # unwrappedSVG() - return the unwrapped version of the SVG.
    #
    def unwrappedSVG(self):
        return self.svg

    #
    # SVG Representation -- adds the svg begin/end markup...
    #
    def _repr_svg_(self):
        x,y,w,h = self.bounds
        _co     = self.rt_self.co_mgr.getTVColor('background','default')
        return f'<svg x="0" y="0" width="{w}" height="{h}">' + \
               f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" />' + \
               self.svg + \
               '</svg>'
    
    #
    # Debugging Original Indices
    #
    def __debug_svgOfOverlayOriginalIndices__(self):
        svg_overlay = ''
        _co = self.rt_self.co_mgr.getTVColor('data','default')
        for i in self.orig_to_xy:
            x,y = self.orig_to_xy[i]
            svg_overlay += f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y-self.txt_h}" stroke="{_co}" />'

        x,y,w,h = self.bounds
        _co     = self.rt_self.co_mgr.getTVColor('background','default')
        return f'<svg width="{w}" height="{h}">' + \
               f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" />' + \
               self.svg + \
               svg_overlay + \
               '</svg>'
    
    #
    # Debugging Word Geometries
    #
    def __debug_svgOfWordColors__(self):
        svg_underlay = ''

        for _poly in self.geom_to_word:
            _word = self.geom_to_word[_poly]
            _co   = self.rt_self.co_mgr.getColor(_word)
            svg_underlay += f'<path d="{self.rt_self.shapelyPolygonToSVGPathDescription(_poly)}" fill="{_co}" />'

        x,y,w,h = self.bounds
        _co     = self.rt_self.co_mgr.getTVColor('background','default')
        return f'<svg width="{w}" height="{h}">' + \
               f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" />' + \
               svg_underlay + \
               self.svg     + \
               '</svg>'

    #
    # Create a positional dataframe of words / sentences / paragraphs.
    #
    def positionalDataFrame(self):
        _txt = []
        _typ = []
        _num = [] # number within that type
        _beg = []
        _end = []
        # Paragraphs
        _parts = self.txt.split('\n')
        i,para_num = 0,0
        for _part in _parts:
            if len(_part) > 0:
                _txt.append(_part)
                _typ.append('para')
                _num.append(para_num)
                _beg.append(i)
                _end.append(i+len(_part))
                para_num += 1
                i += len(_part)
            i += 1 # for the '\n'

        # Sentences
        _sents = self.rt_self.textExtractSentences(self.txt)
        i,sent_num = 0,0
        for _sent in _sents:
            _txt.append(_sent[0])
            _typ.append('sent')
            _num.append(sent_num)
            _beg.append(_sent[-2])
            _end.append(_sent[-1])
            sent_num += 1
            i += len(_sent)

        # Words
        _word = ''
        i,i0,word_num = 0,-1,0
        while i < len(self.txt):
            c = self.txt[i]
            if self.rt_self.__whitespace__(c) or self.rt_self.__punctuation__(c):
                if len(_word) > 0:
                    _txt.append(_word)
                    _typ.append('word')
                    _num.append(word_num)
                    _beg.append(i0)
                    _end.append(i)
                    word_num += 1
                    _word = ''
                    i0 = -1
            elif i0 != -1:
                _word += c
            else:
                i0    = i
                _word = str(c)
            i += 1
        if len(_word) > 0:
            _txt.append(_word)
            _typ.append('word')
            _num.append(word_num)
            _beg.append(i0)
            _end.append(i)
            word_num += 1
        
        return pd.DataFrame({
            'text': _txt,
            'type':_typ,
            'num':_num,
            'beg':_beg,
            'end':_end
        })
    
    #
    # renderDataFrame() - render a position dataframe (assumes some level of filtering)
    # ... i.e., one can filter the pandas dataframe and then re-render to highlight text/etc.
    #
    def renderDataFrame(self, 
                        df,                               # Positional Dataframe from positionalDataFrame() method...
                        color_by          = None,         # Field in the dataframe
                        color_by_style    = 'highlight',  # 'highlight' (like a highlighter), 'underline', or 'text'
                        highlight_opacity = 0.6,          # Opacity of highlight
                        context_opacity   = 0.7,          # Context opacity
                        render_context    = True):        # Render all of the text as a background
        x,y,w,h = self.bounds
        _co     = self.rt_self.co_mgr.getTVColor('background','default')
        my_svg  = f'<svg x="0" y="0" width="{w}" height="{h}">' + \
                  f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" />'
        
        # Render the context for the highlights
        if render_context:
            my_svg += self.svg
            my_svg += f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" fill-opacity="{context_opacity}" />'
        
        # First pass... really only for highlights or for underlines
        _co = '#404040'
        if   color_by_style == 'highlight':
            for row_i,row in df.iterrows():
                if color_by is not None:
                    _co = self.rt_self.co_mgr.getColor(row[color_by])
                _poly = self.spanGeometry(row['beg'],row['end'])
                my_svg += f'<path d="{self.rt_self.shapelyPolygonToSVGPathDescription(_poly)}" fill="{_co}" fill-opacity="{highlight_opacity}" />'
                pass
        elif color_by_style == 'underline':
            _stroke_w = min(0.5 + self.txt_h/14, 2.5)
            for row_i,row in df.iterrows():
                if color_by is not None:
                    _co = self.rt_self.co_mgr.getColor(row[color_by])
                x0, x1, y = None,None,None
                for i in range(row['beg'],row['end']):
                    c, xy = self.txt[i], self.orig_to_xy[i]
                    if self.rt_self.__whitespace__(c) or self.rt_self.__punctuation__(c):
                        if x0 is not None:
                            my_svg += f'<line x1="{x0}" y1="{y + 1 + _stroke_w}" x2="{x1}" y2="{y + 1 + _stroke_w}" stroke="{_co}" stroke-width="{_stroke_w}" />'
                            x0, x1, y = None,None,None
                    elif x0 is not None and y != xy[1]:
                        my_svg += f'<line x1="{x0}" y1="{y + 1 + _stroke_w}" x2="{x1}" y2="{y + 1 + _stroke_w}" stroke="{_co}"  stroke-width="{_stroke_w}" />'
                        x0, x1, y = None,None,None
                    elif x0 is None:
                        x0, x1, y = xy[0], xy[0] + self.rt_self.textLength(c, self.txt_h), xy[1]
                    else:
                        x1 = xy[0] + self.rt_self.textLength(c, self.txt_h)
                if x0 is not None:
                    my_svg += f'<line x1="{x0}" y1="{y + 1 + _stroke_w}" x2="{x1}" y2="{y + 1 + _stroke_w}" stroke="{_co}"  stroke-width="{_stroke_w}" />'
        elif color_by_style == 'text':
            pass
        else:
            raise Exception('RTTextBlock.renderDataFrame() - color_by_style "{highlight_style}" unknown')

        # Second pass...
        for row_i, row in df.iterrows():
            for i in range(row['beg'],row['end']):
                xy = self.orig_to_xy[i]
                if color_by is None or color_by_style != 'text':
                    my_svg += self.rt_self.svgText(self.txt[i], xy[0], xy[1], self.txt_h)
                else:
                    _co = self.rt_self.co_mgr.getColor(row[color_by])
                    my_svg += self.rt_self.svgText(self.txt[i], xy[0], xy[1], self.txt_h, color=_co)

        my_svg += '</svg>'
        return my_svg