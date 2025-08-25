import os
import json
# import sys
# import numpy as np


# def get_main_dir(depth: int = 0):
#     """Get the main directory of the project."""
#     from os.path import dirname as up

#     main_dir = os.path.dirname(os.path.abspath(__file__))
#     for _ in range(depth):
#         main_dir = up(main_dir)
#     return main_dir


def get_all_paths_of_extension(directory: str, extension: str = "json") -> list[str]:
    """Get paths of all files with '.extension' extension in the specified directory and its subdirectories."""
    paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(directory)
        for name in files
        if name.endswith(f".{extension}")
    ]
    return paths


def load_all_data(directory: str='data//original_data') -> dict[str, dict[str, str]]:
    datas_path = [os.path.join(directory, 'airbus_helicopters_train_set.json'),
                  os.path.join(directory, 'open_source_dataset.json')]
    
    dataset : dict[str, dict[str, str]] = {}
    
    for data_path in datas_path:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Error when loading dataset {data_path}")

        with open(data_path, "r", encoding="utf8") as json_file:
            dataset.update(json.load(json_file))
    
    return dataset


def load_dataset(filepath: str) -> dict:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")

    with open(filepath, "r", encoding="utf8") as json_file:
        data = json.load(json_file)
    
    return data


# # Une autre version que je propose, qui est plus développée, et qui affiche des stats dessus
# def load_dataset_bis(directory: str, show: bool = False) -> dict[str, dict[str, str]]:
#     """Load all JSON files in the specified directory and its subdirectories."""
#     paths = get_all_paths_of_extension(directory, "json")
#     if paths == []:
#         raise FileNotFoundError(f"No JSON files found in {directory}")

#     data = {}
#     for path in paths:
#         with open(path, encoding='utf8') as file:
#             print(path)
#             data.update(json.load(file))

#     if show:
#         print(f"Found {len(paths)} JSON files.")
#         print("Files:")
#         for path in paths:
#             print("-", os.path.basename(path))
#         print(f"Total data length: ", len(data))
#         print(f"Size in memory: {sys.getsizeof(data) / 1024:.2f} KB")
#         print(
#             f"Average length of the original text: {np.mean([len(example['original_text']) for example in data.values()]):.2f}"
#         )
#         print(
#             f"Average length of the reference summary: {np.mean([len(example['reference_summary']) for example in data.values()]):.2f}"
#         )
#         print(
#             "Longest original text:",
#             max([len(example["original_text"]) for example in data.values()]),
#         )
#         print(
#             "Longest reference summary:",
#             max([len(example["reference_summary"]) for example in data.values()]),
#         )
#         print("\n" + "Example of the first element:")
#         print(json.dumps(list(data.values())[0], indent=4))

#     return data


if __name__ == "__main__":
    print(len(load_all_data()))
    print(load_dataset(filepath=os.path.join('data', 'data_json', 'train.json')))