# AH Hackathon AI 2024 - Evaluation Toolkit

## Getting started

This toolkit is an evaluation script which produces a report in json format (which is the standard output format) of the scores of the different metrics evaluated as part of the hackathon.

## Requirements

This script was developped and tested on python 3.9.18, you can install all the dependencies using requirements.txt file:pip install -r requirements.txt
## Input format

To facilitate the use of this evalutation toolkit, you must format your generated summary like the standard defined in the corpus:


```
{
    "test_sum01": {
        "generated_summary": "...",
        "uid": "test_sum01"
    },
    ...
}
```
## Usage

```
$ python src/evaluate.py -r <path/reference_dataset.json> -g <path/generated_dataset.json>
```

For us :

```
python .\evaluation-toolkit\src\evaluate.py -r data/test.json -g .\src\generated_summaries\zephyr-7b-beta.json
```

## Authors
Airbus Helicopters