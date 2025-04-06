from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel
import time

__name__ = 'sweep_across_context_down_solver'

from xwords import XWordsSolver, XWords

#
# SweepAcrossSweepDownSolver()
# ... go through all the clues and solve them (w/out regard to other solved clues)
#
class SweepAcrossContextDownSolver(XWordsSolver):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'model' not in kwargs: self.model = 'gemma3:27b'
        else:                     self.model = kwargs['model']

    def solve(self):
        def promptModel(prompt):
            response: ChatResponse = chat(model=self.model, messages=[{ 'role': 'user', 'content': prompt,},],)
            return response['message']['content']
        promptModel('What is 55*3?  Return a single number.') # force model to load so as not to mess up the timing
        class Guess(BaseModel):
            guess: str
        answer_lu, _num_of_llm_requests_, _request_stats_ = {}, 0, []
        for cluenum, orientation in self.xwords.allClueNumbersAndOrientations():
            _tuple_ = (cluenum, orientation)
            if orientation != 'across': continue
            clue    = self.xwords.clue(cluenum, orientation)
            prompt  = f'Solve the crossword puzzle clue "{clue}" that is {self.xwords.numberOfLetters(cluenum, orientation)} letters long.  Return the characters as a JSON object.'
            t0 = time.time()
            response: ChatResponse = chat(model=self.model, messages=[{ 'role': 'user', 'content':  prompt,},], format=Guess.model_json_schema())
            t1 = time.time()
            _num_of_llm_requests_ += 1
            _request_stats_.append((prompt, response['message']['content'], t1-t0, response.prompt_eval_count, response.eval_count))
            guess = Guess.model_validate_json(response['message']['content'])
            if len(guess.guess) != self.xwords.numberOfLetters(cluenum, orientation):
                if ' ' in guess.guess: guess.guess = guess.guess.replace(' ', '')         # maybe there's spaces for multi-word answers?
                if len(guess.guess) != self.xwords.numberOfLetters(cluenum, orientation):
                    print('!',end='')
                else:
                    self.xwords.guess(cluenum, orientation, guess.guess)
                    answer_lu[_tuple_] = guess.guess
                    print('+',end='')
            else:
                self.xwords.guess(cluenum, orientation, guess.guess)
                answer_lu[_tuple_] = guess.guess
                print('.',end='')

        for cluenum, orientation in self.xwords.allClueNumbersAndOrientations():
            _tuple_ = (cluenum, orientation)
            if orientation != 'down': continue
            clue    = self.xwords.clue(cluenum, orientation)

            _contexts_ = self.xwords.describeMissingLetters(cluenum, orientation)
            if len(_contexts_) == 0:
                prompt  = f'Solve the crossword puzzle clue "{clue}" that is {self.xwords.numberOfLetters(cluenum, orientation)} characters long.  Return the characters as a JSON object.'
            else:
                prompt  = f'Solve the crossword puzzle clue "{clue}" that is {self.xwords.numberOfLetters(cluenum, orientation)} characters long.  \n' + \
                          f'Context: {", ".join(_contexts_)}\n\n Return the characters as a JSON object.'
            t0 = time.time()
            response: ChatResponse = chat(model=self.model, messages=[{ 'role': 'user', 'content':  prompt,},], format=Guess.model_json_schema())
            t1 = time.time()
            _num_of_llm_requests_ += 1
            _request_stats_.append((prompt, response['message']['content'], t1-t0, response.prompt_eval_count, response.eval_count))
            guess = Guess.model_validate_json(response['message']['content'])
            if len(guess.guess) != self.xwords.numberOfLetters(cluenum, orientation):
                if ' ' in guess.guess: guess.guess = guess.guess.replace(' ', '')         # maybe there's spaces for multi-word answers?
                if len(guess.guess) != self.xwords.numberOfLetters(cluenum, orientation):
                    print('!',end='')
                else:
                    self.xwords.guess(cluenum, orientation, guess.guess)
                    answer_lu[_tuple_] = guess.guess
                    print('+',end='')
            else:
                self.xwords.guess(cluenum, orientation, guess.guess)
                answer_lu[_tuple_] = guess.guess
                print('.',end='')

        return answer_lu, _request_stats_, _num_of_llm_requests_
