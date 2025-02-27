import pandas as pd
import panel as pn
import html
import param

__name__ = 'scu_editor_panel'

#
# SCUEditorPanel() - edit the excerpts for a specific summary source and question id
#
class SCUEditorPanel(param.Parameterized):
    """ Don't forget the following extension initialization.
    
pn.extension(design="material", sizing_mode="stretch_width")
"""
    #
    # __init__() - constructor
    #
    def __init__(self,
                 rt_self,
                 df,
                 q_id,
                 q_id_field      = 'question_id',
                 question_field  = 'question',
                 scu_field       = 'summary_content_unit',
                 source_field    = 'model',
                 summary_field   = 'summary',
                 excerpt_field   = 'excerpt',
                 n_cols          = 3, 
                 w               = 1600,
                 edit_dataframe  = True,
                 **params):
        super().__init__(**params)
        self.rt_self           = rt_self
        self.df                = df
        self.q_id              = q_id
        self.q_id_field        = q_id_field
        self.source_field      = source_field
        self.source_list       = list(self.df.query(f'`{self.q_id_field}` == @self.q_id')[self.source_field].unique())
        self.source            = self.source_list[0]
        self.question_field    = question_field
        self.scu_field         = scu_field
        self.summary_field     = summary_field
        self.excerpt_field     = excerpt_field
        self.n_cols            = n_cols
        self.w                 = w
        self.edit_dataframe    = edit_dataframe
        self.summary           = self.df.query(f'`{self.q_id_field}` == @self.q_id and `{self.source_field}` == @self.source')[self.summary_field].unique()[0]
        self.scus              = sorted(self.df.query(f'`{self.q_id_field}` == @self.q_id')[self.scu_field].unique()) # all scu's identified for this question
        self.scu_to_text_input = {}
        self.text_input_to_scu = {}

        # ensure there's a maximum of one excerpt per scu/question_id/source
        _df_ = self.df.groupby([self.q_id_field, self.source_field, self.scu_field]).count().reset_index()
        if max(_df_[self.excerpt_field]) > 1: raise Exception(f'Multiple excerpts for {self.q_id_field} {self.source_field} {self.scu_field}')

        # make widgets
        self.text_inputs = []
        for scu in self.scus:
            _df_ = df.query(f'`{self.q_id_field}` == @self.q_id and `{self.scu_field}` == @scu and `{self.source_field}` == @self.source')
            if len(_df_) == 0: _str_ = ''
            else:              _str_ = _df_.iloc[0][self.excerpt_field]
            text_input               = pn.widgets.TextInput(name=scu, 
                                                            value=_str_,
                                                            stylesheets=[self.validationColor(_str_)])
            text_input.param.watch(self.inputTextChanged, ['value_input','value'], onlychanged=False)
            self.scu_to_text_input[scu], self.text_input_to_scu[text_input] = text_input, scu
            self.text_inputs.append(text_input)
        self.summary_widget      = pn.pane.HTML(self.markupHighlights())
        self.scu_examples_widget = pn.pane.HTML('<h3>Examples...</h3>')

        self.source_select = pn.widgets.Select(name='Source', options=self.source_list)
        self.source_select.param.watch(self.sourceSelectChanged, ['value'])

        # make layout
        self._column_ = pn.Column(self.source_select,
                                  self.summary_widget, 
                                  pn.GridBox(*self.text_inputs, ncols=self.n_cols, sizing_mode="fixed", width=self.w),
                                  self.scu_examples_widget)

    def _update_panel(self):
        pass
        
    def panel(self):
        return self._column_

    #
    # sourceSelectChanged() - change the source
    # - disable editing of the dataframe
    #
    def sourceSelectChanged(self, *events):
        self.source = self.source_select.value
        _df_ = self.df.query(f'`{self.q_id_field}` == @self.q_id and `{self.source_field}` == @self.source').reset_index()
        original_edit_dataframe = self.edit_dataframe
        self.edit_dataframe = False
        self.summary        = _df_.iloc[0][self.summary_field]
        for _text_input_ in self.text_inputs: _text_input_.value = '' # clear all values
        for i in range(len(_df_)):
            _scu_              = _df_.iloc[i][self.scu_field]
            _text_input_       = self.scu_to_text_input[_scu_]
            possible_txt       = _df_.iloc[i][self.excerpt_field]
            if possible_txt is None: possible_txt = ''
            _text_input_.value = possible_txt
            _text_input_.stylesheets = [self.validationColor(possible_txt)]
        self.summary_widget.object = self.markupHighlights()
        self.edit_dataframe = original_edit_dataframe

    #
    # validationColor() - set the color of the text based on whether all parts of the excerpt are in the summary
    #
    # https://panel.holoviz.org/how_to/styling/design_variables.html
    def validationColor(self, txt): # color of the text itself (if the page is dark, this is better)
        if txt is None or len(txt) == 0: return ':host { --design-secondary-text-color: #ffd7b5; }'
        _parts_ = txt.split('...')
        for _part_ in _parts_:
            _part_ = _part_.lower().strip()
            if _part_ not in self.summary.lower(): return ':host { --design-secondary-text-color: #ff0000; }'
        return ':host { --design-secondary-text-color: #00008b; }'

    #
    # exampleSCUs() - give examples from other sources as HTML markup
    #
    def OLD_exampleSCUs(self, scu):
        _htmls_ = [f'<h3> "{html.escape(str(scu))}" Examples </h3>']
        _df_ = self.df.query(f'`{self.q_id_field}` == @self.q_id and `{self.scu_field}` == @scu and `{self.source_field}` != @self.source').reset_index()
        for i in range(len(_df_)):
            _excerpt_ = _df_.iloc[i][self.excerpt_field]
            _model_   = _df_.iloc[i][self.source_field]
            _htmls_.append(f'<p><b>{html.escape(str(_model_))}</b><br>{html.escape(str(_excerpt_))}</p>')
        return ''.join(_htmls_)

    #
    # exampleSCU() - give examples from other sources as HTML markup
    # - this one uses an html table
    #
    def exampleSCUs(self, scu):
        _htmls_ = [f'<h3> "{html.escape(str(scu))}" Examples </h3>']
        _df_ = self.df.query(f'`{self.q_id_field}` == @self.q_id and `{self.scu_field}` == @scu and `{self.source_field}` != @self.source').reset_index()
        _htmls_.append('<table>')
        for i in range(len(_df_)):
            _excerpt_ = _df_.iloc[i][self.excerpt_field]
            if _excerpt_ is None or len(_excerpt_) == 0: continue
            _model_   = _df_.iloc[i][self.source_field]
            _htmls_.append(f'<tr><td><b>{html.escape(str(_model_))}</b></td><td>{html.escape(str(_excerpt_))}</td></tr>')
        _htmls_.append('</table>')
        return ''.join(_htmls_)

    #
    # inputTextChanged() - update the style based on the text so far
    # ... update the summary with new highlights
    # ... give examples from other sources
    #
    def inputTextChanged(self, *events):
        for event in events:
            txt = event.obj.value_input
            event.obj.stylesheets = [self.validationColor(txt)]
            if event.obj in self.text_input_to_scu: 
                self.scu_examples_widget.object = self.exampleSCUs(self.text_input_to_scu[event.obj])        
            if self.edit_dataframe:
                txt_to_add_to_df = txt.strip()
                txt_is_empty     = True
                for _part_ in txt_to_add_to_df.split('...'):
                    if len(_part_.strip()) > 0: txt_is_empty = False
                if txt_is_empty: txt_to_add_to_df = None
                _location_ = (self.df[self.q_id_field]   == self.q_id)   & \
                             (self.df[self.source_field] == self.source) & \
                             (self.df[self.scu_field]    == self.text_input_to_scu[event.obj])
                if len(self.df[_location_]) == 0:
                    _list_ = []
                    for _col_ in self.df.columns:
                        if   _col_ == self.q_id_field:     _list_.append(self.q_id)
                        elif _col_ == self.question_field: _list_.append(self.df.query(f'`{self.q_id_field}` == @self.q_id')[self.question_field].unique()[0])
                        elif _col_ == self.scu_field:      _list_.append(self.text_input_to_scu[event.obj])
                        elif _col_ == self.source_field:   _list_.append(self.source)
                        elif _col_ == self.summary_field:  _list_.append(self.summary)
                        elif _col_ == self.excerpt_field:  _list_.append(txt_to_add_to_df)
                        else:                              _list_.append(None)
                    self.df.loc[len(self.df)] = _list_
                else:
                    self.df.loc[_location_, self.excerpt_field] = txt_to_add_to_df
        self.summary_widget.object = self.markupHighlights()

    #
    # markupHighlights() - markup the summary based on the excerpts
    #
    def markupHighlights(self):
        tuples = []
        # Identify the tuples (indices and lengths) based on the excerpt parts
        for scu in self.scus:
            _excerpt_ = self.scu_to_text_input[scu].value
            if _excerpt_ is None or len(_excerpt_) == 0: continue
            _parts_   = _excerpt_.split('...')
            for _part_ in _parts_:
                _part_ = _part_.strip().lower()
                if len(_part_) == 0: continue
                i0 = 0
                i0 = self.summary.lower().index(_part_, i0) if _part_ in self.summary.lower()[i0:] else None
                while i0 is not None:
                    i1 = i0 + len(_part_)
                    tuples.append((i0, len(_part_)))
                    i0 = self.summary.lower().index(_part_, i1) if _part_ in self.summary.lower()[i1:] else None
        # Aggregate the tuples
        tuples = sorted(tuples)
        i = 0
        while i < len(tuples):
            if i < len(tuples)-1 and tuples[i+1][0] <= tuples[i][0] + tuples[i][1]:
                tuples[i] = (tuples[i][0], (tuples[i+1][0] + tuples[i+1][1]) - tuples[i][0])
                tuples.pop(i+1)
            else: i += 1
        # Markup the HTML
        with_marks = []
        i, j = 0, 0
        while i < len(self.summary):
            if j < len(tuples):
                if i < tuples[j][0]:
                    with_marks.append(html.escape(self.summary[i:tuples[j][0]]))
                _safe_ = html.escape(self.summary[tuples[j][0]:tuples[j][0]+tuples[j][1]])
                with_marks.append(f'<mark>{_safe_}</mark>')
                i, j = tuples[j][0]+tuples[j][1], j+1
            else:
                with_marks.append(html.escape(self.summary[i:]))
                i = len(self.summary)
        return ''.join(with_marks)

    #
    # createDataFrame() - create a dataframe of the currently filled in values
    #
    def createDataFrame(self):
        _lu_     = {self.q_id_field:[], self.question_field:[], self.source_field:[], self.scu_field:[], self.summary_field:[], self.excerpt_field:[]}
        for scu in self.scu_to_text_input:
            _excerpt_ = self.scu_to_text_input[scu].value
            if len(_excerpt_) == 0: continue
            _lu_[self.q_id_field].append(self.q_id)
            _lu_[self.question_field].append(self.df.query(f'`{self.q_id_field}` == @self.q_id')[self.question_field].unique()[0])
            _lu_[self.source_field].append(self.source)
            _lu_[self.scu_field].append(scu)
            _lu_[self.summary_field].append(self.summary)
            _lu_[self.excerpt_field].append(_excerpt_)
        return pd.DataFrame(_lu_)

    #
    # excerptCoverage()
    # - calculate the coverage of the excerpts
    # - returns (percentage of summary covered, length of the excerpts, length of the summary, aggregated spans)
    # - if exclude_citation_spans is True, then exclude citation spans
    #   citation spans will look as follows:  [1], [2], [1,2,4], [1-5,8]
    #
    def excerptCoverage(self, q_id, source, exclude_citation_spans=True):
        _df_      = self.df.query(f'`{self.q_id_field}` == @q_id and `{self.source_field}` == @source').reset_index()
        _summary_ = _df_.iloc[0][self.summary_field].lower()
        _spans_ = []
        for i in range(len(_df_)):
            _excerpt_ = _df_.iloc[i][self.excerpt_field]
            if _excerpt_ is None: continue
            for _part_ in _excerpt_.split('...'):
                _part_ = _part_.strip().lower()
                if len(_part_) == 0: continue
                if _part_ not in _summary_: raise Exception(f'{q_id=} {source=} "{_part_}" from "{_excerpt_}" not in "{_summary_}"')
                j = 0
                while _part_ in _summary_[j:]:
                    j = _summary_.index(_part_, j)
                    _spans_.append((j, len(_part_)))
                    j += len(_part_)

        # Aggregate the spans if they overlap
        _spans_ = self.rt_self.textAggregateSpans(_spans_)
        
        # Exclude citations
        if exclude_citation_spans:
            _citation_spans_       = self.rt_self.textCitationSpans(_summary_)
            _spans_wout_citations_ = self.rt_self.textSubtractSpans(_spans_, _citation_spans_)
            len_of_covered_text = 0
            for i in range(len(_spans_wout_citations_)): len_of_covered_text += _spans_wout_citations_[i][1]
            len_summary_wout_citations = len(_summary_)
            for i in range(len(_citation_spans_)): len_summary_wout_citations -= _citation_spans_[i][1]
            return len_of_covered_text/len_summary_wout_citations, len_of_covered_text, len_summary_wout_citations, _spans_
        else:
            len_of_covered_text = 0
            for i in range(len(_spans_)): len_of_covered_text += _spans_[i][1]
            return len_of_covered_text/len(_summary_), len_of_covered_text, len(_summary_), _spans_

    #
    # excerptsDensity() - determine the average density of the excerpts per character
    #
    def excerptsDensity(self, q_id, source, ignore_uncovered_chars=True):
        _df_      = self.df.query(f'`{self.q_id_field}` == @q_id and `{self.source_field}` == @source').reset_index()
        _summary_ = _df_.iloc[0][self.summary_field].lower()
        _counts_  = [0] * len(_summary_)
        for i in range(len(_df_)):
            _excerpt_ = _df_.iloc[i][self.excerpt_field]
            _parts_   = _excerpt_.split('...')
            for _part_ in _parts_:
                _part_ = _part_.strip().lower()
                if len(_part_) == 0: continue
                i0 = 0
                i0 = _summary_.index(_part_, i0) if _part_ in _summary_[i0:] else None
                while i0 is not None:
                    i1 = i0 + len(_part_)
                    for j in range(i0, i1): _counts_[j] += 1
                    i0 = _summary_.index(_part_, i1) if _part_ in _summary_[i1:] else None
        _sum_, _samples_ = 0, 0
        if ignore_uncovered_chars:
            for i in range(len(_counts_)):
                if _counts_[i] > 0: _sum_, _samples_ = _sum_ + _counts_[i], _samples_ + 1
        else:
            _sum_     = sum(_counts_)
            _samples_ = len(_counts_)
        
        if _samples_ == 0: _samples_ = 1
        return _sum_ / _samples_

    #
    # __applyUnderlines__()
    #
    def __applyUnderlines__(self, text, spans, uncovered_color='#a00000', covered_color='#a0a0a0'):
        # Make alternating text blocks ... every other block is underlined
        sorted_spans  = sorted(spans, key=lambda x: x[0])
        _alternating_ = [] # odds require underline
        i, i0 = 0, 0
        while i < len(sorted_spans):
            _alternating_.append((i0, sorted_spans[i][0]))
            _alternating_.append((sorted_spans[i][0], sorted_spans[i][0] + sorted_spans[i][1]))
            i0 = sorted_spans[i][0] + sorted_spans[i][1]
            i += 1
        _alternating_.append((i0, len(text)))

        # Add the underlines
        _txt_ = []
        for i in range(len(_alternating_)):
            i0, i1 = _alternating_[i][0], _alternating_[i][1]
            if   i%2 == 0: _txt_.append(f'<e style="color: {uncovered_color}">' + html.escape(text[i0:i1]) + '</e>')
            else:          _txt_.append(f'<u style="color: {covered_color}">' + html.escape(text[i0:i1]) + '</u>')
        return ''.join(_txt_)

    def createHTMLForMissingSCUs(self, qids=None, uncovered_color='#a00000', covered_color='#a0a0a0', background_color='white'):
        # Sort the Question ID's by coverage (lowest coverage to highest coverage)
        _lu_ = {self.q_id_field:[], self.source_field:[], 'coverage':[], 'len_sum':[], 'len_summary':[], 'spans':[]}
        qid_to_source_to_coverage = {}
        for k, k_df in self.df.groupby([self.q_id_field, self.source_field]):
            _percent_, _len_, _len_summary_, _spans_ = self.excerptCoverage(k[0], k[1])
            _lu_[self.q_id_field].append(k[0]), _lu_[self.source_field].append(k[1]),      _lu_['coverage'].append(_percent_)
            _lu_['len_sum'].append(_len_),      _lu_['len_summary'].append(_len_summary_), _lu_['spans'].append(_spans_)
            if k[0] not in qid_to_source_to_coverage: qid_to_source_to_coverage[k[0]] = {}
            qid_to_source_to_coverage[k[0]][k[1]] = _percent_
        df_coverage         = pd.DataFrame(_lu_)
        df_coverage_average = df_coverage.groupby(self.q_id_field).agg({'coverage': 'mean'}).sort_values('coverage').reset_index()
        # Create the HTML
        _htmls_ = []
        _htmls_.append('''<style>
.tooltip { position: relative; display: inline-block; border-bottom: 1px dotted black; }
.tooltip .tooltiptext {
  visibility: hidden; width: 800px; background-color: #555; color: #fff; text-align: center;
  border-radius: 6px; padding: 5px 0; position: absolute; z-index: 1; bottom: 125%;
  left: 50%; margin-left: -60px; opacity: 0; transition: opacity 0.3s; }
.tooltip .tooltiptext::after { content: ""; position: absolute; top: 100%; left: 50%; margin-left: -5px;
  border-width: 5px; border-style: solid; border-color: #555 transparent transparent transparent; }
.tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
</style>
''')

        for q_id in df_coverage_average[self.q_id_field]:
            if qids is not None and q_id not in qids: continue
            question = self.df.query(f'`{self.q_id_field}` == @q_id').iloc[0][self.question_field]
            _htmls_.append(f'<h3> ({html.escape(str(q_id))}) {html.escape(str(question))} </h3>')

            # For each source w/in the specific question id, sort from least to highest coverage
            source_ordering = []
            for source in df_coverage.query(f'`{self.q_id_field}` == @q_id').sort_values('coverage').reset_index()[self.source_field]: source_ordering.append(source)

            # Table Header
            _htmls_.append('<table>')
            _htmls_.append('<tr align="center">')
            for source in source_ordering:
                _coverage_ = qid_to_source_to_coverage[q_id][source]
                _htmls_.append(f'<td align="center"> <b> {html.escape(str(source))} </b> ({_coverage_:0.2f}) </td>')
            _htmls_.append('</tr>')

            # Summaries w/ Underlines
            _htmls_.append('<tr>')
            for source in source_ordering:
                _summary_ = self.df.query(f'`{self.q_id_field}` == @q_id and `{self.source_field}` == @source')[self.summary_field].unique()[0]
                _summary_ = self.__applyUnderlines__(_summary_, df_coverage.query(f'`{self.q_id_field}` == @q_id and `{self.source_field}` == @source').iloc[0]['spans'], uncovered_color, covered_color)
                _htmls_.append(f'<td align="left" valign="top"> {_summary_} </td>')
            _htmls_.append('</tr>')

            # Figure out the order of the SCU's
            _df_ = self.df.query(f'`{self.q_id_field}` == @q_id').dropna(subset=[self.excerpt_field]).reset_index().drop('index', axis=1)
            _df_ = _df_[_df_[self.excerpt_field] != ""].reset_index()
            _df_ = _df_.groupby(self.scu_field).size().reset_index().rename({0:'__count__'},axis=1).sort_values('__count__', ascending=False)
            scu_ordering = []
            scu_to_count = {}
            for i in range(len(_df_)):
                scu = _df_.iloc[i][self.scu_field]
                scu_to_count[scu] = _df_.iloc[i]['__count__']
                scu_ordering.append(scu)

            _htmls_.append('<tr>')
            for source in source_ordering: _htmls_.append('<td align="center"> Missing SCU\'s </td>')
            _htmls_.append('</tr>')

            def examples(q_id, scu, source):
                _df_examples_ = self.df.query(f'`{self.q_id_field}` == @q_id and `{self.source_field}` != @source and `{self.scu_field}` == @scu').reset_index()
                _txts_ = []
                for i in range(len(_df_examples_)):
                    _possible_txt_ = _df_examples_.iloc[i][self.excerpt_field]
                    if _possible_txt_ is not None:
                        _possible_txt_ = str(_possible_txt_).strip()
                        if _possible_txt_ == '': continue
                        _txts_.append(html.escape(_possible_txt_))
                return ' | '.join(_txts_)
            
            # List the missing SCU's -- in order of most occuring scu to least occuring scu
            _htmls_.append('<tr>')
            for source in source_ordering:
                _htmls_.append('<td align="left" valign="top">')
                for scu in scu_ordering:
                    _df_ = self.df.dropna(subset=[self.excerpt_field]).query(f'`{self.q_id_field}` == @q_id and `{self.source_field}` == @source and `{self.scu_field}` == @scu').reset_index()
                    _df_ = _df_[_df_[self.excerpt_field] != ""].reset_index()
                    if len(_df_) == 0: 
                        _htmls_.append(f'<li>')
                        _htmls_.append(f'[{scu_to_count[scu]}] ')
                        _htmls_.append('<div class="tooltip">')
                        _htmls_.append(f'{html.escape(str(scu))}')
                        _htmls_.append('<span class="tooltiptext">')
                        _htmls_.append(examples(q_id, scu, source))
                        _htmls_.append('</span>')
                        _htmls_.append('</div>')
                        _htmls_.append(f'</li>')
                _htmls_.append('</td>')
            _htmls_.append('</tr>')

            _htmls_.append('</table><hr>')

        _html_header_ = f'<!DOCTYPE html><html><body style="background-color:{background_color};">'
        _html_footer_ = '</body></html>'
        return _html_header_ + ''.join(_htmls_) + _html_footer_

