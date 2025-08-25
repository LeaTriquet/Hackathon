import torch
import numpy as np
import xgboost as xgb
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


class SummariesAnalyser:
    def __init__(self, xgboost_weight_path: str=None) -> None:
        self.semantic_model = SentenceTransformer("./evaluation_toolkit/all-MiniLM-L6-v2/")

        rouge_metrics_name = ["rouge1", "rouge2", "rougeL"]
        self.scorer = rouge_scorer.RougeScorer(rouge_metrics_name, use_stemmer=True)

        if xgboost_weight_path is not None:
            self.xg_reg = xgb.XGBRegressor()
            self.xg_reg.load_model(xgboost_weight_path)
        else:
            self.xg_reg = None

        if torch.cuda.is_available():
            self.semantic_model = self.semantic_model.cuda()    
    
    def compute_metrics(self,
                        original_text: str,
                        summaries: list[str]
                        ) -> dict[str, list[float]]:
        """
        renvoie le meilleur résumé et son indice dans la liste des summaries
        le meilleur résumé est calculé selon la similarité avec le texte originale
        """
        results: dict[str, list[float]] = {}
        results.update(self.compute_similarity(original_text, summaries))
        results.update(self.compute_rouge(original_text, summaries))
        
        return results
    
    def compute_similarity(self,
                           original_text: str,
                           summaries: list[str]
                           ) -> dict[str, list[float]]:
        """ compute cos similarity metrics on all diferent summary for the original_text """
        similarity_score: list[float] = []
        original_text_embeddings = self.semantic_model.encode(original_text, convert_to_tensor=True)

        for summary in summaries:
            generated_embeddings = self.semantic_model.encode(summary, convert_to_tensor=True)
            sim_score: float = util.cos_sim(original_text_embeddings, generated_embeddings).cpu().item()
            similarity_score.append(sim_score)
            
        return {'similarity': similarity_score}
    
    def compute_rouge(self,
                      original_text: str,
                      summaries: list[str]
                      ) -> dict[str, list[float]]:
        """ compute rouge metrics on all diferent summary for the original_text """
        rouge_score: dict[str, list[float]] = {"rouge1": [], "rouge2": [], "rougeL": []}

        for summary in summaries:
            r_score = self.scorer.score(target=original_text, prediction=summary)
            for rouge_metrics in rouge_score.keys():
                rouge_score[rouge_metrics].append(float(r_score[rouge_metrics].fmeasure))
        
        return rouge_score
    
    def get_keys(self) -> list[str]:
        """ get all keys of compute_metrics output """
        return ['similarity', 'rouge1', 'rouge2', 'rougeL']
        
    def find_best(self,
                  original_text: str,
                  summaries: list[str],
                  xgboost_weight_path: str=None
                  ) -> tuple[int, str]:
        """ returns the index and the best summary found by xbgoost among all summaries
        specify xgboost_weight_path if you didn't do so when initializing the class
        """
        if self.xg_reg is None:
            self.xg_reg = xgb.XGBRegressor()
            if xgboost_weight_path is None:
                raise ValueError(f'specify the path of the xgboost weights')
            self.xg_reg.load_model(xgboost_weight_path)
        
        results = self.compute_metrics(original_text, summaries)
        # ic(results)
        results: np.ndarray[float] = np.array(list(results.values())).T
        y_score = np.array([result[0]/2+np.sum(result[1:])/6 for result in results])
        y_pred = np.array(self.xg_reg.predict(results))
        # Perform weighted mean
        weights = [0.5, 0.5]  # Adjust the weights as needed
        scores = np.average([y_score, y_pred], axis=0, weights=weights)
        # ic(y_pred)
        best_sum_index = np.argmax(scores)
        # print("Summary found scores: ", scores)
        return best_sum_index, scores[best_sum_index], summaries[best_sum_index]
        

if __name__ == '__main__':
    import os
    summaries_analyser = SummariesAnalyser(xgboost_weight_path=os.path.join('src', 'xgboost', 'xgweigth.model'))

    original_text = "If some Parts are declared as non-airworthy by the Seller and returned to the Customer, the Seller waives all liability on said Parts which shall be scrapped under Customer\u2019s responsibility and expenses. In such case and without any formal request from the Customer in the repair Order or any other documents considered as contractual, said Parts will be recorded and identified as unserviceable by the Seller according to the Seller\u2019s applicable procedures (record of the scrapped Part in the Seller\u2019s database, identification of the Parts through \u201cunserviceable\u201d tag and identification of the Parts with a triangle scrapping mark when possible)."
    summaries = ['Seller waives all liability on said Parts which shall be scrapped under Customer’s responsibility and expenses. In such case and without any formal request from the Customer in the repair Order or any other documents considered as contractual, said Parts will be recorded and identified as unserviceable.', 'If some Parts are declared as non-airworthy by the Seller and returned to the Customer, the Seller waives all liability on said Parts which shall be scrapped under Customer’s responsibility and expenses. In such case and without any formal request from the Customer in the repair Order or any other documents considered as contractual, said Parts will be recorded and identified as unserviceable.', 'If some Parts are declared as non-airworthy by the Seller and returned to the Customer, the Seller waives all liability on said Parts which shall be scrapped under Customer’s responsibility and expenses. In such case and without any formal request from the Customer in the repair Order or any other documents considered as contractual, said Parts will be recorded and identified as unserviceable.']
    # summaries_analyser.compute_rouge(original_text, summaries)
    
    print(summaries_analyser.compute_metrics(original_text, summaries))
    print(summaries_analyser.get_keys())

    print(summaries_analyser.find_best(original_text, summaries))
