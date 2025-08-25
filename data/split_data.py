import os
import sys
import json
import numpy as np
from os.path import dirname as up
np.random.seed(0)

sys.path.append(up(up(os.path.abspath(__file__))))

from src.dataloader import load_dataset, load_all_data


def get_and_split_data(data: dict[str, dict[str, str]],
                       data_name: str='Airbus'
                       ) -> None:
    """ split data into train and test """
    
    keys_train = choise(data=list(data.keys()), k=int(0.9 * len(data)))

    data_train : dict[str, dict[str, str]] = {}
    data_test :  dict[str, dict[str, str]] = {}

    data_items = np.array(list(data.items()))
    np.random.shuffle(data_items)

    for key, value in data_items:
        if key in keys_train:
            data_train[key] = value
        else:
            data_test[key] = value
    
    open_source_data = '_opensource' if data_name != 'Airbus' else ''
    train = f'train{open_source_data}'
    test = f'test{open_source_data}'

    save_into_csv(data=list(data_train.values()),
                  filename=os.path.join('data', 'data_csv', f'{train}.csv'))
    save_into_csv(data=list(data_test.values()),
                  filename=os.path.join('data', 'data_csv', f'{test}.csv'))

    save_into_json_with_promt3(data=data_train,
                               filename=os.path.join('data', 'data_json_instruct', f'{train}.json'))
    save_into_json_with_promt3(data=data_test,
                               filename=os.path.join('data', 'data_json_instruct', f'{test}.json'))

    save_into_json(data=data_train, filename=os.path.join('data', 'data_json', f'{train}.json'))
    save_into_json(data=data_test, filename=os.path.join('data', 'data_json', f'{test}.json'))


def make_prompt(original_text: str, reference_summary: str, **kwars) -> str:
    return f'human: {original_text} \\n bot: {reference_summary}'

def make_prompt2(original_text: str, reference_summary: str, **kwars) -> str:
    return f"### Human: {original_text} ### Assistant: {reference_summary.replace(',', '')}"

def make_prompt3(original_text: str, reference_summary: str, **kwars) -> dict[str, str]:
    promt = 'Please provide a concise summary of the main points covered in the text.'
    return {"instruction": promt, "input": original_text, "output": reference_summary}


def save_into_csv(data: list[dict[str, str]], filename: str) -> None:
    with open(filename, mode='w', encoding='utf8') as f:
        f.write('text\n')
        for dico in data:
            f.write(f'{make_prompt2(**dico)}\n')
        f.close()


def save_into_json(data: dict[str, str], filename: str) -> None:
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(data, f, indent=2)
        f.close()


def save_into_json_with_promt3(data: dict[str, str], filename: str) -> None:
    new_data:list[dict[str, str]] = list(map(lambda value: make_prompt3(**value), data.values()))
    with open(filename, 'w', encoding='utf8') as f:
        json.dump(new_data, f, indent=2)
        f.close()


def choise(data: list[str], k: int) -> list[str]:
    index = np.arange(len(data))
    np.random.shuffle(index)
    data = np.array(data)
    return list(data[index[:k]])


if __name__ == '__main__':
    airbus_data = False
    if airbus_data:
        data = load_dataset(os.path.join('data', 'original_data', 'airbus_helicopters_train_set.json'))
        get_and_split_data(data, data_name='Airbus')
    else:
        data = load_dataset(os.path.join('data', 'original_data', 'open_source_dataset.json'))
        get_and_split_data(data, data_name='open_source')
    
    