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
                 df,
                 q_id,
                 source,
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
        self.df                = df
        self.q_id              = q_id
        self.source            = source
        self.q_id_field        = q_id_field
        self.question_field    = question_field
        self.scu_field         = scu_field
        self.source_field      = source_field
        self.summary_field     = summary_field
        self.excerpt_field     = excerpt_field
        self.n_cols            = n_cols
        self.w                 = w
        self.edit_dataframe    = edit_dataframe
        self.summary           = self.df.query(f'`{self.q_id_field}` == @self.q_id and `{self.source_field}` == @self.source')[self.summary_field].unique()[0]
        self.scus              = sorted(self.df.query(f'`{self.q_id_field}` == @self.q_id')[self.scu_field].unique()) # all scu's identified for this question
        self.scu_to_text_input = {}
        self.text_input_to_scu = {}

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

        # make layout
        self._column_ = pn.Column(self.summary_widget, 
                                  pn.GridBox(*self.text_inputs, ncols=self.n_cols, sizing_mode="fixed", width=self.w),
                                  self.scu_examples_widget)

    def _update_panel(self):
        pass
        
    def panel(self):
        return self._column_

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
    def exampleSCUs(self, scu):
        _htmls_ = [f'<h3> "{html.escape(scu)}" Examples </h3>']
        _df_ = self.df.query(f'`{self.q_id_field}` == @self.q_id and `{self.scu_field}` == @scu and `{self.source_field}` != @self.source').reset_index()
        for i in range(len(_df_)):
            _excerpt_ = _df_.iloc[i][self.excerpt_field]
            _model_   = _df_.iloc[i][self.source_field]
            _htmls_.append(f'<p><b>{html.escape(_model_)}</b><br>{html.escape(_excerpt_)}</p>')
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
