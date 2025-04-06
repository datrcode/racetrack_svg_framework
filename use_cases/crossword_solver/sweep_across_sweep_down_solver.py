from ollama import chat
from ollama import ChatResponse
from pydantic import BaseModel
import time

__name__ = 'sweep_across_sweep_down_solver'

from xwords import XWordsSolver, XWords

class SweepAcrossSweepDownSolver(XWordsSolver):
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
        response_lu = {}
        for cluenum, orientation in self.xwords.allClueNumbersAndOrientations():
            _tuple_ = (cluenum, orientation)
            if _tuple_ not in response_lu: response_lu[_tuple_] = []
            clue    = self.xwords.clue(cluenum, orientation)
            prompt  = f'Solve the crossword puzzle clue "{clue}" that is {self.xwords.numberOfLetters(cluenum, orientation)} letters long.  Return the characters as a JSON object.'
            t0 = time.time()
            response: ChatResponse = chat(model=self.model, messages=[{ 'role': 'user', 'content':  prompt,},], format=Guess.model_json_schema())
            t1 = time.time()
            response_lu[_tuple_].append((t1-t0, response))
            guess = Guess.model_validate_json(response['message']['content'])
            if len(guess.guess) != self.xwords.numberOfLetters(cluenum, orientation):
                if ' ' in guess.guess: guess.guess = guess.guess.replace(' ', '')
                if len(guess.guess) != self.xwords.numberOfLetters(cluenum, orientation):
                    print('!',end='')
                else:
                    self.xwords.guess(cluenum, orientation, guess.guess)
                    print('+',end='')
            else:
                self.xwords.guess(cluenum, orientation, guess.guess)
                print('.',end='')
