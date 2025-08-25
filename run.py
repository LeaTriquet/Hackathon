import time
import argparse

from src.dataloader import load_dataset
from src.hyperparameter import HyperParameter

from src.models.gemma_pipeline import GemmaPipeline
from src.models.t5_few_shot_pipeline import T5FewShotPipeline
from src.models.decimake_analyse import SummaryDecisionmakerAnalyse



def check_options(options: dict[str, str]) -> dict[str, str | bool]:
    if options['use_llm'].lower() == "true":
        options['use_llm'] = True
    elif options['use_llm'].lower() == "false":
        options['use_llm'] = False
    else:
        raise ValueError(f'Expected use_llm is true or false but found {options["use_llm"]}')
    
    if options['t5_path'] is None:
        raise ValueError(f'T5 path is required')
    
    if options['use_llm'] and options['gemma_path'] is None:
        raise ValueError(f'Gemma path is required if use_llm=True')
    
    if options['data_path'] is None:
        raise ValueError(f'Data path is required')
    
    if options['output_path'] is None:
        raise ValueError(f'Output path is required')
    
    for key, value in options.items():
        print(f'{key:<15} : {value}')

    return options



def infer_t5(options: dict[str, str | bool]) -> None:

    if options['use_llm']:
        raise ValueError(f'Error function only for inference with only t5 (not using llm)')

    data = load_dataset(options['data_path'])
    
    t5 = T5FewShotPipeline(options['t5_path'], 
                           num_summary=10, 
                           length_penalty=HyperParameter(name='lp', _min=-0, _max=1),
                           num_beams=HyperParameter(name='b', _min=5, _max=15),
                           batch_size=1)
    
    start_time = time.time()
    
    t5.generate_summaries(dataset=data,
                          output_file=options['output_path'])
    
    end_time = time.time()
    print(f"temps d'inference: {end_time - start_time}s")
    

def infer_t5_gemma(options: dict[str, str | bool]) -> None:

    if not options['use_llm']:
        raise ValueError(f'Error function must use llm')

    data = load_dataset(options['data_path'])

    t5 = T5FewShotPipeline(options['t5_path'], 
                           num_summary=10, 
                           length_penalty=HyperParameter(name='lp', _min=-0, _max=1),
                           num_beams=HyperParameter(name='b', _min=5, _max=15),
                           batch_size=10)

    llm = GemmaPipeline(options['gemma_path'],
                        temperature=0.7)

    summarizer = SummaryDecisionmakerAnalyse(llm=llm,
                                             small_summarizer=t5,
                                             n_results=1,
                                             prompt_id=1,
                                             use_llm=options['use_llm'])
    
    
    summarizer.load_collection("full_collection")

    start_time = time.time()
    summarizer.generate_summaries(dataset=data,
                                  output_file=options['output_path'])
    end_time = time.time()
    print(f"temps d'inference: {end_time - start_time}s")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gemma_path', '-g', type=str, help='path to the model config and safetensors')
    parser.add_argument('--t5_path', '-t', type=str, help='path to the model config and safetensors')
    parser.add_argument('--data_path', '-d', type=str, help='path to the data set')
    parser.add_argument('--output_path', '-o', type=str, help='path to the output file')
    parser.add_argument('--use_llm', '-u', type=str, help='use the language model to generate the output file')
    args = parser.parse_args()

    options = check_options(options=vars(args))

    if not options['use_llm']:
        infer_t5(options)
    else:
        infer_t5_gemma(options)
