def get_main_dir(depth: int = 0):
    """Get the main directory of the project."""
    import os
    import sys
    from os.path import dirname as up


    main_dir = os.path.dirname(os.path.abspath(__file__))
    for _ in range(depth):
        sys.path.append(up(main_dir))
        main_dir = up(main_dir)
    return main_dir


MAIN_DIR = get_main_dir(depth=2)

import os
import chromadb
from tqdm import tqdm
import json
import time
from src.models.summarizer import Summarizer

from src.dataloader import load_dataset
from data.analyse_grammaticale.fonction_prompt import creation_elements_prompt
from src.utils import group_list
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from src.find_best_summary import SummariesAnalyser



class SummaryDecisionmakerAnalyse(Summarizer):
    def __init__(
        self,
        llm: Summarizer,
        small_summarizer: Summarizer,
        n_results: int = 1,
        prompt_id:int = 0,
        use_llm: bool = False
    ) -> None:
        self.name = "decimake-analyse-" + str(n_results) + "_" + llm.name + f"-temp{llm.temperature:.2f}" + "_" + small_summarizer.name + f"({prompt_id})"
        super().__init__(local=False, loaded=True, name=self.name)
        self.llm = llm
        self.small_summarizer = small_summarizer
        self.n_results = n_results
        self.embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2", device="cuda"
        )
        self.use_llm = use_llm
        self.chroma_client = chromadb.PersistentClient(MAIN_DIR + "/src/collection")
        self.prompt_id = prompt_id
        self.lengths = [40, 60, 100, 150]
        # print("Lengths intervals:", f"{self.lengths[0]};{self.lengths[1]};{self.lengths[2]};{self.lengths[3]};")
        self.prompt_ids = [None, 3, 0, 1, 2]
        # print("Prompt ids:", f"{self.prompt_ids[0]};{self.prompt_ids[1]};{self.prompt_ids[2]};{self.prompt_ids[3]};{self.prompt_ids[4]};")

        promt_path = MAIN_DIR + "/src/prompt_templates/decimake_analyse.json"
        if not os.path.exists(promt_path):
            raise FileExistsError(promt_path)
        self.prompt_templates = json.load(open(promt_path))[f"{self.prompt_id}"]


    def load_collection(self, collection_name: str) -> None:
        """Load the collection with the given name. If it does not exist, create it."""
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function
        )

    def group_list(self, list: list, group_size: int) -> list[list]:
        """Creates a list of lists of group_size"""
        k = len(list)
        grouped_list = []
        for k in range(0, k, group_size):
            grouped_list.append(list[k : k + group_size])
        return grouped_list

    def ingest(self, dataset: dict[str, dict[str, str]], batch_size: int = 16) -> None:
        """Ingest dataset of examples into the collection."""
        batchs = group_list(list(dataset.values()), batch_size)
        for batch in tqdm(batchs, desc="Ingesting documents"):
            self.collection.add(
                documents=[document["original_text"] for document in batch],
                metadatas=[
                    {"reference_summary": document["reference_summary"]}
                    for document in batch
                ],
                ids=[document["uid"] for document in batch],
            )

    def retrieve(self, original_text: str, n_results: int = 2) -> str:
        """Retrieves relevant context for the given original text."""
        results = self.collection.query(
            query_texts=[original_text], n_results=n_results
        )
        return results
    
    def summary(self, original_text: str) -> str:
        """Make a summary."""
        
        # prompt_templates = json.load(open(MAIN_DIR + "/src/prompt_templates/decimake_analyse.json"))[f"{self.prompt_id}"] 
        
        element_prompt = creation_elements_prompt(original_text.replace(".", " ").replace(",", " ").replace(";", " ").replace(":", " ").replace("(", " ").replace(")", " ").replace("  ", " "))

        analyse = "Here are the sens group of the text:"
        analyse += "\nThe head and most important sens group is: " + element_prompt[0] + "\n"
        for element in element_prompt[1:]:
            analyse += "Sens group: " + element + "\n"
        
        small_summarizer_score, small_summarizer_summary = self.small_summarizer.summary_scored(original_text)
        small_summarizer_summary = self.clean_string(small_summarizer_summary)

        
        if not self.use_llm or small_summarizer_score > 0.82:
            return small_summarizer_summary
        
        if len(original_text.split()) > self.lengths[3]:
            self.prompt_template = self.prompt_templates[f"{self.prompt_ids[4]}"]  
            # print(f"Template {self.prompt_ids[4]}") 
        elif len(original_text.split()) > self.lengths[2]:
            self.prompt_template = self.prompt_templates[f"{self.prompt_ids[3]}"]  
            # print(f"Template {self.prompt_ids[2]}") 
        elif len(original_text.split()) < self.lengths[0]:
            # print(f"Template {self.prompt_ids[0]}") 
            return small_summarizer_summary
        elif len(original_text.split()) < self.lengths[1]:
            self.prompt_template = self.prompt_templates[f"{self.prompt_ids[2]}"] 
            # print(f"Template {self.prompt_ids[1]}") 
        else:
            self.prompt_template = self.prompt_templates[f"{self.prompt_ids[1]}"]
            # print(f"Template {self.prompt_ids[2]}") 
            
        rules = self.prompt_template["rules"]
            
        small_summarizer_prompt = self.prompt_template["small_summarizer"].format(small_summary=small_summarizer_summary)

        results = self.retrieve(original_text, n_results=self.n_results+2)
        results_text = [text for text in results["documents"][0]]
        results_summary = [results["metadatas"][0][k]["reference_summary"] for k in range(len(results["metadatas"][0]))] 
        text_summary_pairs = list(zip(results_text, results_summary))
        sorted_pairs = sorted(text_summary_pairs, key=lambda pair: len(pair[0].split()))
        closest_texts = sorted_pairs[:self.n_results] 
        example_prompt = ""
        for text, summary in closest_texts:
            example_prompt += self.prompt_template["example"].format(
                original_text=text,
                reference_summary=summary
            )
                
        prompt = self.prompt_template["main"].format(
            example_prompt=example_prompt,
            original_text=original_text,
            small_summarizer=small_summarizer_prompt,
            analyse=analyse,
            rules=rules
        )
       
        response = self.llm.ask(prompt)
                
        llm_summary = response.split("\n")
        llm_summary = [self.clean_string(summary) for summary in llm_summary if len(summary) > 0 and self.clean_string(summary) != original_text]
        llm_summary.append(small_summarizer_summary)
        summaries_analyser = SummariesAnalyser(xgboost_weight_path=os.path.join('src', 'xgboost', 'xgweigth.model'))
        _, score, summary =  summaries_analyser.find_best(original_text, llm_summary)
        print("Length of original text:", len(original_text.split()))
        print("Length ratio:", len(original_text.split())/len(summary.split()))
        print("Score predicted:", score)
                 
        return summary