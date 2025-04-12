__name__ = 'xwords'

import time
import ast
import networkx as nx
#import rtsvg
#rt = rtsvg.RACETrack()

class XWordsSolver(object):
    def __init__(self, xwords, **kwargs):
        self.xwords = xwords
    def solve(self):
        raise Exception('XWordsSolver.solve() -- not implemented')

class XWords(object):
    #
    # Constructor
    #
    # entries_file      {(cluenum, orientation): clue, ...}
    # geometries_file   [(cluenum, xi, yi), ...]
    # blockers_file     [(xi,yi), ...]
    #
    def __init__(self, rt_self, entries_file, geometries_file, blockers_file, answers_file, cell_w=32, cell_h=32, txt_h=22, cluenum_txt_h=10):
        self.rt_self         = rt_self
        self.entries_file    = entries_file
        self.geometries_file = geometries_file
        self.blockers_file   = blockers_file
        self.answers_file    = answers_file
        self.cell_w          = cell_w
        self.cell_h          = cell_h
        self.txt_h           = txt_h
        self.cluenum_txt_h   = cluenum_txt_h    

        # Read the files
        with open(self.entries_file) as f:
            self.entries    = ast.literal_eval(f.read())
        with open(self.geometries_file) as f:
            self.geometries = ast.literal_eval(f.read())
        with open(self.blockers_file) as f:
            self.blockers   = ast.literal_eval(f.read())
        with open(self.answers_file) as f:
            self.answers    = ast.literal_eval(f.read())
        
        # Determine the number of tiles
        self.x_tiles, self.y_tiles = 1, 1
        for blocker in self.blockers:
            self.x_tiles = max(self.x_tiles, blocker[0] + 1)
            self.y_tiles = max(self.y_tiles, blocker[1] + 1)
        for geometry in self.geometries:
            self.x_tiles = max(self.x_tiles, geometry[1] + 1)
            self.y_tiles = max(self.y_tiles, geometry[2] + 1)
        self.w = self.x_tiles * cell_w
        self.h = self.y_tiles * cell_h

        class XWCell(object):
            def __init__(self, xi, yi):
                self.xi, self.yi    = xi, yi
                self.__cluenum__    = None
                self.__is_blocker__ = False
                self.__clues__      = {}
                self.__guesses__    = {}
                self.__answer__     = None
            def addGuess(self, cluenum, orientation, guess):
                self.__guesses__[(cluenum, orientation)] = guess
            def setAnswer(self, answer):
                self.__answer__ = answer
            def setClueNumber(self, cluenum): 
                self.__cluenum__ = cluenum
                if self.__is_blocker__:
                    raise Exception(f'XWCell.setCluenNumber() -- blocker already set {self.__is_blocker__} ({self.xi},{self.yi})')
            def isBlocker(self): 
                return self.__is_blocker__
            def setBlocker(self): 
                self.__is_blocker__ = True
                if self.__cluenum__ is not None:
                    raise Exception(f'XWCell.setBlocker() -- cluenum already set {self.__cluenum__} ({self.xi},{self.yi})')
            def addClue(self, cluenum, clue, orientation):
                if self.__cluenum__ != cluenum:
                    raise Exception(f'XWCell.addClue() -- cluenum mismatch {self.__cluenum__} != {cluenum} ({self.xi},{self.yi})')
                if orientation in self.__clues__:
                    raise Exception(f'XWCell.addClue() -- orientation already set {orientation} ({self.xi},{self.yi})')
                self.__clues__[orientation] = clue
            def clearGuess(self, cluenum, orientation):
                if (cluenum, orientation) in self.__guesses__:
                    del self.__guesses__[(cluenum, orientation)]
            def clear(self):
                self.__guesses__ = {}

        # Create a two dimensions structure that captures the state of each cell
        self.cells = [] # self.cells[yi][xi]
        for yi in range(self.y_tiles):
            _row_ = []
            for xi in range(self.x_tiles): _row_.append(XWCell(xi, yi))
            self.cells.append(_row_)

        # ... fill the cells in with information from the other structures
        self.cluenum_to_cell = {}
        for geometry in self.geometries:
            cluenum, xi, yi               = geometry
            _cell_                        = self.cells[yi][xi]
            _cell_.setClueNumber(cluenum)
            self.cluenum_to_cell[cluenum] = _cell_
        for cluenum_orientation in self.entries:
            cluenum, orientation          = cluenum_orientation
            clue                          = self.entries[cluenum_orientation]
            _cell_                        = self.cluenum_to_cell[cluenum]
            _cell_.addClue(cluenum, clue, orientation)
        for blocker in self.blockers:
            xi, yi                        = blocker
            _cell_                        = self.cells[yi][xi]
            _cell_.setBlocker()
        for cluenum_orientation in self.answers:
            cluenum, orientation = cluenum_orientation
            _num_of_letters_     = self.numberOfLetters(cluenum, orientation)
            if len(self.answers[cluenum_orientation]) != _num_of_letters_:
                raise Exception(f'XWords.__init__() -- number of letters mismatch {len(self.answers[cluenum_orientation])} != {_num_of_letters_} for {cluenum_orientation}')
            _cell_               = self.cluenum_to_cell[cluenum]
            if orientation == 'across':
                for i in range(_num_of_letters_):
                    _cell_across_ = self.cells[_cell_.yi][_cell_.xi + i]
                    _cell_across_.setAnswer(self.answers[cluenum_orientation][i])
            else:
                for i in range(_num_of_letters_):
                    _cell_down_ = self.cells[_cell_.yi + i][_cell_.xi]
                    _cell_down_.setAnswer(self.answers[cluenum_orientation][i])

    #
    # _repr_svg_() - return an svg representation of the crossword puzzle
    #
    def _repr_svg_(self):
        svg = [f'<svg x="0" y="0" width="{self.w}" height="{self.h}" viewBox="-5 -5 {self.w+10} {self.h+10}">']
        svg.append(f'<rect x="-5" y="-5" width="{self.w+10}" height="{self.h+10}" fill="#ffffff" />')
        for x_tile in range(self.x_tiles+1):
            svg.append(f'<line x1="{x_tile*self.cell_w}" y1="0" x2="{x_tile*self.cell_w}" y2="{self.h}" stroke="black" stroke-width="0.25" />')
        for y_tile in range(self.x_tiles+1):
            svg.append(f'<line x1="0" y1="{y_tile*self.cell_h}" x2="{self.w}" y2="{y_tile*self.cell_h}" stroke="black" stroke-width="0.25" />')
        for blocker in self.blockers:
            svg.append(f'<rect x="{blocker[0]*self.cell_w}" y="{blocker[1]*self.cell_h}" width="{self.cell_w}" height="{self.cell_h}" fill="black" />')
        for geometry in self.geometries:
            cluenum, xi, yi = geometry
            _cell_          = self.cluenum_to_cell[cluenum]
            x, y            = xi*self.cell_w, yi*self.cell_h
            svg.append(self.rt_self.svgText(str(cluenum), x + 2, y + 2 + self.cluenum_txt_h, txt_h=self.cluenum_txt_h, color='gray'))
        for yi in range(len(self.cells)):
            for xi in range(len(self.cells[yi])):
                _cell_ = self.cells[yi][xi]
                if _cell_.isBlocker(): continue
                x, y = xi*self.cell_w, yi*self.cell_h
                # Draw in the answer / faded ... only if there are no guesses
                _answer_ = _cell_.__answer__
                if _answer_ is not None: _answer_ = _answer_.upper()
                if _cell_.__answer__ is not None and len(_cell_.__guesses__) == 0:
                    svg.append(self.rt_self.svgText(_cell_.__answer__.upper(), x + self.cell_w/2.0, y + self.cell_h - 4, txt_h=self.txt_h, color='#d8d8d8', anchor='middle'))
                # Draw in the guesses
                _guesses_ = []
                for k in _cell_.__guesses__.keys(): _guesses_.append(f'{_cell_.__guesses__[k]}')
                _guesses_ = list(set(_guesses_)) # this is to remove duplicates ... in case it doesn't make sense the next time I see it...
                if   len(_guesses_) == 1:
                    _color_ = 'red' if _answer_ != _guesses_[0].upper() else 'black'
                    svg.append(self.rt_self.svgText(_guesses_[0], x + self.cell_w/2.0, y + self.cell_h - 4, txt_h=self.txt_h, color=_color_, anchor='middle'))
                elif len(_cell_.__guesses__) == 2:
                    _color_ = 'red' if _answer_ != _guesses_[0].upper() else 'black'
                    svg.append(self.rt_self.svgText(_guesses_[0], x + 1*self.cell_w/4.0, y + self.cell_h     - 2, txt_h=self.txt_h//2, color=_color_, anchor='middle'))
                    _color_ = 'red' if _answer_ != _guesses_[1].upper() else 'black'
                    svg.append(self.rt_self.svgText(_guesses_[1], x + 3*self.cell_w/4.0, y + self.cell_h/2.0 - 2, txt_h=self.txt_h//2, color=_color_, anchor='middle'))
                elif len(_cell_.__guesses__) >  2:
                    svg.append(self.rt_self.svgText('?', x + self.cell_w/2.0, y + self.cell_h - 2, txt_h=self.txt_h, color='red', anchor='middle'))
                    print('should we really be here? xwords._repr_svg_()')                
        svg.append('</svg>')
        return ''.join(svg)

    #
    # smallMultipleSVG() - return an svg representation for a small multiple
    #
    def smallMultipleSVG(self, s=5):
        w_sm = self.x_tiles * s
        h_sm = self.y_tiles * s
        svg = [f'<svg x="0" y="0" width="{w_sm}" height="{h_sm}" >']
        svg.append(f'<rect x="0" y="0" width="{w_sm}" height="{h_sm}" fill="#ffffff" />')
        for x_tile in range(self.x_tiles+1):
            svg.append(f'<line x1="{x_tile*s}" y1="0" x2="{x_tile*s}" y2="{h_sm}" stroke="black" stroke-width="0.1" />')
        for y_tile in range(self.x_tiles+1):
            svg.append(f'<line x1="0" y1="{y_tile*s}" x2="{w_sm}" y2="{y_tile*s}" stroke="black" stroke-width="0.1" />')
        for blocker in self.blockers:
            svg.append(f'<rect x="{blocker[0]*s}" y="{blocker[1]*s}" width="{s}" height="{s}" fill="black" />')
        for yi in range(len(self.cells)):
            for xi in range(len(self.cells[yi])):
                _cell_ = self.cells[yi][xi]
                if _cell_.isBlocker(): continue
                x, y = xi*s, yi*s
                # Draw in the answer / faded ... only if there are no guesses
                _answer_ = _cell_.__answer__
                if _answer_ is not None: _answer_ = _answer_.upper()
                # Draw in the guesses
                _guesses_ = []
                for k in _cell_.__guesses__.keys(): _guesses_.append(f'{_cell_.__guesses__[k]}')
                _guesses_ = list(set(_guesses_)) # this is to remove duplicates ... in case it doesn't make sense the next time I see it...
                _color_   = None
                if   len(_guesses_) == 1:
                    _color_ = 'red' if _answer_ != _guesses_[0].upper() else 'green'
                elif len(_cell_.__guesses__) == 2:
                    if   _answer_ != _guesses_[0].upper() and _answer_ != _guesses_[1].upper(): _color_ = '#818603'
                    elif _answer_ == _guesses_[0].upper() or  _answer_ != _guesses_[1].upper(): _color_ = 'yellow'
                    elif _answer_ != _guesses_[0].upper() or  _answer_ == _guesses_[1].upper(): _color_ = 'yellow'
                    else:                                                                       _color_ = 'green'
                elif len(_cell_.__guesses__) >  2:
                    _color_ = '#404040' # gray / shouldn't happen based on the current implementation
                if _color_ is not None: svg.append(f'<rect x="{x}" y="{y}" width="{s}" height="{s}" fill="{_color_}" stroke="#ffffff" stroke-width="0.2" />')
        svg.append('</svg>')
        return ''.join(svg)

    #
    # numberOfLetters() - return the number of letters for a (cluenum, orientation)
    #
    def numberOfLetters(self, cluenum, orientation):
        _cell_ = self.cluenum_to_cell[cluenum]
        xi, yi = _cell_.xi, _cell_.yi
        if orientation == 'across':
            while self.cells[yi][xi].isBlocker() == False: 
                xi += 1
                if xi == self.x_tiles: break
            return xi - _cell_.xi
        else:
            while self.cells[yi][xi].isBlocker() == False: 
                yi += 1
                if yi == self.y_tiles: break
            return yi - _cell_.yi
    
    #
    # clue() - return the clue for a (cluenum, orientation)
    #
    def clue(self, cluenum, orientation):
        return self.entries[(cluenum, orientation)]
    
    #
    # answer()
    #
    def answer(self, cluenum, orientation):
        return self.answers[(cluenum, orientation)]
    
    #
    # crossCluesAtCellCoordinates()
    # - returns returned as a two tuples
    #   [(cluenum, orientation, character_index), (cluenum, orientation, character_index)]
    #   ... character_index starts at 1
    #
    def crossCluesAtCellCoordinates(self, xi, yi):
        _cell_ = self.cells[yi][xi]
        if _cell_.isBlocker(): return None
        dx, dy = 0, 0
        while yi+dy > 0 and self.cells[yi+dy-1][xi].isBlocker() == False: dy -= 1 
        while xi+dx > 0 and self.cells[yi][xi+dx-1].isBlocker() == False: dx -= 1
        _cell_down_   = self.cells[yi+dy][xi]
        _cell_across_ = self.cells[yi][xi+dx]
        return (_cell_across_.__cluenum__, 'across', abs(dx)+1), (_cell_down_.__cluenum__, 'down', abs(dy)+1)

    #
    # allClueNumbersAndOrientations() - return all the clue numbers and orientations
    #
    def allClueNumbersAndOrientations(self):
        return set(self.entries.keys())
    
    #
    # guess()
    #
    def guess(self, cluenum, orientation, guess):
        guess               = guess.upper()
        _cell_              = self.cluenum_to_cell[cluenum]
        _xi_, _yi_          = _cell_.xi, _cell_.yi
        _number_of_letters_ = self.numberOfLetters(cluenum, orientation)
        if _number_of_letters_ != len(guess):
            raise Exception(f'XWords.guess() -- number of letters mismatch {_number_of_letters_} != {len(guess)}')
        if orientation == 'across':
            for i in range(_number_of_letters_):
                _cell_ = self.cells[_yi_][_xi_ + i]
                _cell_.addGuess(cluenum, orientation, guess[i])
        else:
            for i in range(_number_of_letters_):
                _cell_ = self.cells[_yi_ + i][_xi_]
                _cell_.addGuess(cluenum, orientation, guess[i])

    #
    # clearGuess()
    #
    def clearGuess(self, cluenum, orientation):
        _cell_              = self.cluenum_to_cell[cluenum]
        _number_of_letters_ = self.numberOfLetters(cluenum, orientation)
        if orientation == 'across':
            for i in range(_number_of_letters_):
                _cell_to_mod_ = self.cells[_cell_.yi][_cell_.xi + i]
                _cell_to_mod_.clearGuess(cluenum, orientation)
        else:
            for i in range(_number_of_letters_):
                _cell_to_mod_ = self.cells[_cell_.yi + i][_cell_.xi]
                _cell_to_mod_.clearGuess(cluenum, orientation)

    #
    # clearAll()
    #   
    def clearAll(self):
        for yi in range(self.y_tiles):
            for xi in range(self.x_tiles):
                self.cells[yi][xi].clear()

    #
    # describeMissingLetters()
    #
    def describeMissingLetters(self, cluenum, orientation):
        def suffix(n): return 'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'
        _cell_              = self.cluenum_to_cell[cluenum]
        _number_of_letters_ = self.numberOfLetters(cluenum, orientation)
        _strs_              = []
        for i in range(_number_of_letters_):
            if   orientation == 'across': _cell_i_ = self.cells[_cell_.yi][_cell_.xi + i]
            elif orientation == 'down':   _cell_i_ = self.cells[_cell_.yi + i][_cell_.xi]
            else:                         raise Exception(f'XWords.describeMissingLetters() -- unknown orientation: {orientation}')
            _guesses_ = []
            for k in _cell_i_.__guesses__.keys(): _guesses_.append(f'{_cell_i_.__guesses__[k]}')
            _guesses_ = list(set(_guesses_))
            if   len(_guesses_) == 0:
                ...
            elif len(_guesses_) == 1: _strs_.append(f'"{_guesses_[0]}" at the {i+1}{suffix(i+1)} letter')
            elif len(_guesses_) == 2: _strs_.append(f'"{_guesses_[0]}" or "{_guesses_[1]}" at the {i+1}{suffix(i+1)} letter')
            else:                     raise Exception(f'XWords.describeMissingLetters() -- unknown number of guesses: {len(_guesses_)}')
        return _strs_

    def createOverSpecifiedPrompt(self, cluenum, orientation):
        def suffix(n): return 'st' if n == 1 else 'nd' if n == 2 else 'rd' if n == 3 else 'th'
        clue           = self.clue(cluenum, orientation)
        num_of_letters = self.numberOfLetters(cluenum, orientation)
        s = f'What is the answer to the crossword puzzle clue "{clue}" which is {num_of_letters} letters long?'
        other_orientation = 'across' if orientation == 'down' else 'down'
        _cell_     = self.cluenum_to_cell[cluenum]
        _xi_, _yi_ = _cell_.xi, _cell_.yi
        for i in range(num_of_letters):
            if orientation == 'across':
                dy = 0
                while _yi_ + dy > 0 and self.cells[_yi_+dy-1][_xi_].isBlocker() == False: dy -= 1
                other_cluenum = self.cells[_yi_+dy][_xi_].__cluenum__
                s += f'\n- The {i+1}{suffix(i+1)} letter is the {-dy+1}{suffix(-dy+1)} letter of the clue "{self.clue(other_cluenum, other_orientation)}".'
                _xi_ += 1
            else:
                dx = 0
                while _xi_ + dx > 0 and self.cells[_yi_][_xi_+dx-1].isBlocker() == False: dx -= 1
                other_cluenum = self.cells[_yi_][_xi_+dx].__cluenum__
                s += f'\n- The {i+1}{suffix(i+1)} letter is the {-dx+1}{suffix(-dx+1)} letter of the clue "{self.clue(other_cluenum, other_orientation)}".'
                _yi_ += 1
        s += f'\nreturn the answer as a JSON object.'
        return s

    def characterLevelAccuracy(self):
        _correct_, _incorrect_, _total_ = 0, 0, 0
        for yi in range(self.y_tiles):
            for xi in range(self.x_tiles):
                _cell_ = self.cells[yi][xi]
                if _cell_.__answer__ is None: continue
                if _cell_.isBlocker(): continue
                _guesses_ = []
                for k in _cell_.__guesses__.keys(): _guesses_.append(f'{_cell_.__guesses__[k]}')
                _guesses_ = list(set(_guesses_))
                if   len(_guesses_) == 1 and _guesses_[0].upper() == _cell_.__answer__.upper(): _correct_   += 1
                else:                                                                           _incorrect_ += 1
                _total_ += 1
        return _correct_/_total_

    #
    # sweetClipGraphs() - construct two graphs per the following paper:
    #
    #  Language Models are Crossword Solvers
    #  Soumadeep Saha,  Sutanoya Chakraborty, Saptarshi Saha, Utpal Garain 
    #  Indian Statistical Institute Kolkata, India 
    #  arXiv:2406.09043v1 [cs.CL] 13 Jun 2024
    #
    def sweepClipGraphs(self):
        g_p, g_n = nx.Graph(), nx.Graph()
        for yi in range(self.y_tiles):
            for xi in range(self.x_tiles):
                _cell_ = self.cells[yi][xi]
                if _cell_.isBlocker() == False and len(_cell_.__guesses__) >= 2:
                    _as_list_ = list(_cell_.__guesses__.keys())
                    a_i, a_j     = _as_list_[0], _as_list_[1]
                    a_i_u, a_j_v = _cell_.__guesses__[a_i], _cell_.__guesses__[a_j]
                    if a_i_u == a_j_v: g_p.add_edge(a_i, a_j)
                    else:              g_n.add_edge(a_i, a_j)
        return g_p, g_n
