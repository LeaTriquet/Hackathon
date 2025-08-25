import os
import sys
from os.path import dirname as up

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.models.summarizer import Summarizer
from src.dataloader import load_dataset


class Baseline(Summarizer):
    def __init__(self) -> None:
        super().__init__(local=True, loaded=False, name='Baseline')
    
    def summary(self, original_text: str) -> str:
        return original_text
    

if __name__ == '__main__':
    baseline = Baseline()
    test_data = load_dataset(filepath=os.path.join('data', 'data_json', 'test.json'))
    print(test_data)
    baseline.generate_summaries(test_data)