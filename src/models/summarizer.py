import json
from tqdm import tqdm


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


class Summarizer:
    def __init__(self, local: bool, loaded: bool, name: str | None) -> None:
        """Initialize the Summarizer class."""
        self.local = local
        self.loaded = loaded
        self.name = name
        
    def clean_string(self, generated_summary: str) -> str:
        generated_summary = generated_summary.split("\n")[-1]
        generated_summary = generated_summary.replace("\"", "")
        generated_summary = generated_summary.replace('"', "")
        generated_summary = generated_summary.replace("**", "")
        generated_summary = generated_summary.replace("'", "\u2019")
        generated_summary = generated_summary.replace("Text:", "")
        generated_summary = generated_summary.replace("Summary:", "")
        generated_summary = generated_summary.replace("text:", "")
        generated_summary = generated_summary.replace("summary:", "")
        generated_summary = generated_summary.strip()
        return generated_summary
    
    def summary(self, original_text: str) -> str:
        """Make a summary."""
        pass  # This method will be implemented in subclasses

    def generate_summaries(self,
                           dataset: dict[str, dict[str, str]],
                           output_file: str=None
                           ) -> None:
        """Generate summaries for a dataset."""
        json_dict = {}
        
        for keys, data in tqdm(dataset.items(), desc="Generating summaries"):
            json_dict[keys] = {
                    "generated_summary": self.summary(data["original_text"]),
                    "uid": data["uid"],
                }
            
        if output_file is not None:
            dst_path = output_file
        else:
            dst_path = MAIN_DIR + f"/results/generated_summaries/{self.name}.json"

        with open(dst_path, "w") as f:
            json.dump(json_dict, f, indent=2)
            f.close()
            
        print(f"Saving the summaries to {dst_path}")