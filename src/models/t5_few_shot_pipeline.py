import os
import sys
from os.path import dirname as up
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, logging

import torch
from transformers import pipeline

sys.path.append(up(up(up(os.path.abspath(__file__)))))

from src.hyperparameter import HyperParameter
from src.models.summarizer import Summarizer
from src.find_best_summary import SummariesAnalyser
from src.hyperparameter import HyperParameter

class T5FewShotPipeline(Summarizer):
    def __init__(self,
                 folder: str,
                 num_summary: int,
                 length_penalty: HyperParameter,
                 num_beams: HyperParameter,
                 min_length: int=0,
                 max_length: int=1000,
                 batch_size: int=1,
                 xgboost_weight_path: str=os.path.join('src', 'xgboost', 'xgweigth.model'),
                 ) -> None:
        super().__init__(local=False,
                         loaded=True,
                         name=f"t5_fewshot-sm-{num_summary}-{num_beams}-{length_penalty}")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU : ", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("Using CPU")

        self.length_penalty = length_penalty
        self.num_beams = num_beams
        self.num_summary = num_summary
        self.batch_size = batch_size
        
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(folder)
        
        self.summary_parameter = {
            'model': self.model,
            'tokenizer': self.tokenizer,
            'device': device,
            'min_length': min_length,
            'max_length': max_length,
            'do_sample': True
        }
        
        self.summaries_analyser = SummariesAnalyser(xgboost_weight_path)

        
    def summary(self, original_text: str) -> str:
        summaries = self.multiple_summary(original_text)
        unique_summaries = list(set(summaries))
        # print(f'number unique summaries: {len(unique_summaries)}')
        _, _, best_summary = self.summaries_analyser.find_best(original_text, unique_summaries)
        return best_summary
    
        
    def summary_scored(self, original_text: str) -> str:
        summaries = self.multiple_summary(original_text)
        unique_summaries = list(set(summaries))
        # print(f'number unique summaries: {len(unique_summaries)}')
        _, score, best_summary = self.summaries_analyser.find_best(original_text, unique_summaries)
        return score, best_summary
    
    
    def multiple_summary(self, original_text: str) -> list[str]:
        summary: list[str] = []
        for _ in range(self.num_summary):
            result = self.run_one_summary(original_text)
            summary.append(result)
        return summary
    
    def run_one_summary(self, original_text: str) -> str:
        """Make one summary."""
        self.summary_parameter['length_penalty'] = self.length_penalty()
        self.summary_parameter['num_beams'] = int(self.num_beams())

        logging.set_verbosity(logging.ERROR)
        pipe = pipeline("summarization", **self.summary_parameter)
        response = pipe(original_text)[0]["summary_text"]
        logging.set_verbosity(logging.DEBUG)
        return self.clean_string(response)



if __name__ == '__main__':
    bart = T5FewShotPipeline(num_summary=20,
                     length_penalty=HyperParameter(name='lp', _min=0.5, _max=2.0),
                     num_beams=HyperParameter(name='b', _min=2, _max=15))
    
    original_text = "If some Parts are declared as non-airworthy by the Seller and returned to the Customer, the Seller waives all liability on said Parts which shall be scrapped under Customer\u2019s responsibility and expenses. In such case and without any formal request from the Customer in the repair Order or any other documents considered as contractual, said Parts will be recorded and identified as unserviceable by the Seller according to the Seller\u2019s applicable procedures (record of the scrapped Part in the Seller\u2019s database, identification of the Parts through \u201cunserviceable\u201d tag and identification of the Parts with a triangle scrapping mark when possible)."

    bart.summary(original_text)

    # test_data = load_dataset(filepath=os.path.join('data', 'data_json', 'test.json'))
    # bart.generate_summaries(test_data)