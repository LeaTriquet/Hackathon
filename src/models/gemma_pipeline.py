from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


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

from src.models.summarizer import Summarizer


class GemmaPipeline(Summarizer):
    def __init__(self, folder: str, temperature: float = 0.7) -> None:
        """Initializes the LocalLLM object with the given model path."""
        super().__init__(local=True, loaded=False, name='gemma')
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.prompt_template = "Summarize : {orignal_text}:"
            
        self.model = AutoModelForCausalLM.from_pretrained(folder)
        self.tokenizer = AutoTokenizer.from_pretrained(folder)
        self.pipeline = pipeline("text-generation", 
                                model=self.model, 
                                tokenizer=self.tokenizer, 
                                device=device, 
                                temperature=temperature, 
                                max_length=8192,
                                do_sample=True)
        self.loaded = True
        self.temperature = temperature

    def ask(self, prompt: str) -> str:
        """Ask the model a question."""
        reponse = self.pipeline(prompt)[0]['generated_text']
        return reponse

    def summary(self, original_text: str) -> str:
        """Make a summary."""
        prompt = self.prompt_template.format(orignal_text=original_text)
        response = self.ask(prompt)
        print("Response: ", response)
        return self.clean_string(response)
