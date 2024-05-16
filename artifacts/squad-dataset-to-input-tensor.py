#!/usr/bin/env python3
"""
This file contains the script to generate QA input tensors using SQUAD dataset
"""

import json
import logging
import os
import struct
import argparse
import numpy
import sys
from pathlib import Path


############ PARSE ARGUMENTS ###################
def parse_arguments():
    """
    Generate a QA dataset from SQUAD with the specifications introduced as flags
    """
    parser = argparse.ArgumentParser(
        description="Generate a QA dataset from SQuad with the specifications introduced as flags")
    parser.add_argument('-m', '--model',
                        choices=['bert', 'albert', 'distilbert'],
                        type=str, default='albert',
                        help='Model we want to generate the dataset for')
    parser.add_argument("-s", '--seq-length',
                        metavar="N", type=int, default=128,
                        help='Maximum sequential length we want the dataset to be')
    parser.add_argument("-d", '--dataset-length',
                        metavar="N", type=int, default=128,
                        help='Length of the dataset we want to generate')
    parser.add_argument("-b", '--batch-size',
                        metavar="N", type=int, default=1,
                        help='Batch size (default=1)')
    parser.add_argument("-o", '--output-dir',
                        metavar="DIR", type=str, default='.',
                        help='Directory of the output dataset')
    parser.add_argument("-f", '--file', metavar="FILE",
                        type=str, default='squad-v1.1-dev.json',
                        help='Filename of the squad JSON dataset')
    args = parser.parse_args()
    return args


###################################################################
def gen_context_and_question(texts):
    """
    Generate context and ask question
    Args:
        texts: list of text
    """
    for text in texts:
        for paragraph in text['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                if not 'is_impossible' in qa or not qa['is_impossible']:
                     if len(qa['answers']) > 1 and qa['answers'][0]['text'] == qa['answers'][1]['text']: 
                        if len(qa['answers']) == 2 or (len(qa['answers']) > 2 and qa['answers'][0]['text'] == qa['answers'][2]['text']): 
                             yield context, question, qa['answers'][0]['text']


def main():
    args = parse_arguments()
    MODEL = args.model
    max_seq_length = args.seq_length
    dataset_length = args.dataset_length
    output_data_abs_path = args.output_dir
    input_file = Path(args.file)
    batch_size = args.batch_size


    if MODEL == 'bert':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
    elif MODEL == 'albert':
        from transformers import AlbertTokenizer
        tokenizer = AlbertTokenizer.from_pretrained("twmkn9/albert-base-v2-squad2")
    elif MODEL == 'roberta':
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("deepset/roberta-base-squad2")
    elif MODEL == 'distilbert':
        from transformers import DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    else:
        print("Model not recognized")
        return 1
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

    try:
        with open(input_file, 'r') as file:
            dataset = json.load(file)
    except FileNotFoundError:
        print("Cannot find the SQuad Dataset")
        return 1

    # Counter
    count = 0
    # List of dicts. Every single dict will contain all the information needed and then
    # according to the batch size, the output will be created.
    data_per_inference = []
    for context, question, answer in gen_context_and_question(dataset['data']):
        # Dictionary for every context, question and answer
        data_per_inference_sub = {"first_token": None,
                                  "last_token": None,
                                  "dict_context_pads": None,
                                  "inputs_ids_packed": None,
                                  "input_mask_packed": None,
                                  "segment_ids_packed": None,
                                  "data": None
                                  }
        
        if count >= dataset_length:
            print('Dataset generated!')
            break

        dict_answer = tokenizer(answer)
        dict_context = tokenizer(question, context)
        if len(dict_context['input_ids']) <= max_seq_length:
            count += 1
            dict_context_pads = tokenizer(question, context,
                                          padding="max_length",
                                          max_length=max_seq_length,
                                          truncation=True,
                                          return_tensors='pt')
        else:
            continue

        first_token = dict_answer['input_ids'][1]
        last_token = dict_answer['input_ids'][len(dict_answer['input_ids']) - 2]
        if max_seq_length > 0:
            struct_type = 'q'
        else:
            print("Invalid sequence length")
            return 1
        
        inputs_ids_packed = None
        for t in numpy.array(dict_context_pads['input_ids']).flatten():
            if inputs_ids_packed is None:
                inputs_ids_packed = struct.pack(struct_type, t)
            else:
                inputs_ids_packed = inputs_ids_packed + struct.pack(struct_type, t)
        input_mask_packed = None
        for t in numpy.array(dict_context_pads['attention_mask']).flatten():
            if input_mask_packed is None:
                input_mask_packed = struct.pack(struct_type, t)
            else:
                input_mask_packed = input_mask_packed + struct.pack(struct_type, t)
        segment_ids_packed = None
        if MODEL == 'bert' or MODEL == 'albert':
            for t in numpy.array(dict_context_pads['token_type_ids']).flatten():
                if segment_ids_packed is None:
                    segment_ids_packed = struct.pack(struct_type, t)
                else:
                    segment_ids_packed = segment_ids_packed + struct.pack(struct_type, t)

        pos = 0
        answer_start = 0
        answer_end = 0
        sentence_match = False
        for token in dict_context['input_ids']:
            if token == first_token:
                answer_start = pos
                # Same word start and end
                if first_token == last_token:
                    answer_end = pos
                    break
                # Different word between start and end
                sentence_match = True
                context_pos = pos + 1
                pred_pos = 2
                # Iterate to see if prediction matches in the context
                while sentence_match:
                    if dict_context['input_ids'][context_pos] == last_token:
                        answer_end = context_pos
                        break
                    if dict_context['input_ids'][context_pos] != dict_answer['input_ids'][pred_pos]:
                        sentence_match = False
                        break
                    pred_pos = pred_pos + 1
                    context_pos = context_pos + 1
            if sentence_match:
                break
            pos = pos + 1
        data = {'answer': answer, 'answer_start': answer_start, 'answer_end': answer_end, 'context': context}
        # Save data in dict
        data_per_inference_sub['first_token'] = first_token
        data_per_inference_sub['last_token'] = last_token
        data_per_inference_sub['dict_context_pads'] = dict_context_pads
        data_per_inference_sub['inputs_ids_packed'] = inputs_ids_packed
        data_per_inference_sub['input_mask_packed'] = input_mask_packed
        data_per_inference_sub['segment_ids_packed'] = segment_ids_packed
        data_per_inference_sub['data'] = data
        # Append dictionary to list of dictionaries
        data_per_inference.append(data_per_inference_sub)

    # Keep each inference in a separate directory
    inference_list = []
    inference_count = 0
    # Inference loop that determines the total amount of inference by batch size and count tokens
    for inference in range(int(count / batch_size)):
        relative_path = f'data/inference-{inference_count}'
        path = os.path.join(output_data_abs_path, relative_path)
        os.makedirs(path)
        # Prepare dictionaries used on the data descriptor
        inference_dict = {'name': 'inference-' + str(inference_count), 'tensors': {}}
        # Prepare data for inputs_ids, input_mask, segment_ids
        input_ids_total = None
        input_mask_total = None
        segment_ids_total = None
        data_prediction = []
        for batch_iter in range(batch_size):
            if input_ids_total is None:
                input_ids_total = data_per_inference[inference]['inputs_ids_packed']
            else:
                input_ids_total = input_ids_total + data_per_inference[inference]['inputs_ids_packed']
            if input_mask_total is None:
                input_mask_total = data_per_inference[inference]['input_mask_packed']
            else:
                input_mask_total = input_mask_total + data_per_inference[inference]['input_mask_packed']
            if MODEL == 'bert' or MODEL == 'albert':
                if segment_ids_total is None:
                    segment_ids_total = data_per_inference[inference]['segment_ids_packed']
                else:
                    segment_ids_total = segment_ids_total + data_per_inference[inference]['segment_ids_packed']
            data_prediction.append(data_per_inference[inference]['data'])
        # Write data into inference data files
        with open(os.path.join(path, f'input_ids.bin'), 'wb') as f:
            inference_dict['tensors']['input_ids'] = os.path.join(relative_path, f'input_ids.bin')
            f.write(input_ids_total)
        with open(os.path.join(path, f'input_mask.bin'), 'wb') as f:
            inference_dict['tensors']['input_mask'] = os.path.join(relative_path, f'input_mask.bin')
            f.write(input_mask_total)
        if MODEL == 'bert' or MODEL == 'albert':
            with open(os.path.join(path, f'segment_ids.bin'), 'wb') as f:
                inference_dict['tensors']['segment_ids'] = os.path.join(relative_path, f'segment_ids.bin')
                f.write(segment_ids_total)
        inference_list.append(inference_dict)
        # Write prediction
        with open(os.path.join(path, 'prediction.json'), 'w') as file:
            json.dump(data_prediction, file)
        inference_count += 1
    desc = {'name': MODEL + ' ' + Path(input_file).stem, 'model': 'Generic', 'inputs': []}
    desc['inputs'].append({"name": "input_ids", "type": f'index64<{batch_size}x{max_seq_length}>'})
    desc['inputs'].append({"name": "input_mask", "type": f'index64<{batch_size}x{max_seq_length}>'})
    desc['inputs'].append({"name": "segment_ids", "type": f'index64<{batch_size}x{max_seq_length}>'})
    desc['inferences'] = inference_list

    with open(os.path.join(output_data_abs_path, 'data-desc.json'), 'w') as file:
        json.dump(desc, file, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
