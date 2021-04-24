#create_npy_data.py
import logging
import math
import os
import sys
import pandas as pd
import copy
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from datasets import Dataset

import numpy as np

import transformers
from transformers import GPT2Tokenizer
from transformers.utils import check_min_version

import argparse

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--preprocessing_num_workers", type=int,  default=1)
parser.add_argument("--max_seq_length",type=int, default=1024, help="sequence length")
parser.add_argument("--n_lines_per_chunk",type=int, default=1_000_000, help="loading n_lines at a time from the file (lazy loading).")
parser.add_argument("--n_samples_per_npy",type=int, default=100_000, help="saving max n_samples to single  npy file")
parser.add_argument("--model_tokenizer", required=True, 
                    help="path to the saved model tokenizer or the name of the model ['EleutherAI/gpt-neo-1.3B']")
parser.add_argument("--filename", default= "thepile", 
                    help="filename for numpy saved file.")
parser.add_argument("--train_files_path", help="folder path contaning train files", required=True)
parser.add_argument("--test_files_path", help="folder path contaning train files")
parser.add_argument("--overwrite_cache", help="overwrite cache of the tokenized dataset",action="store_true")
parser.add_argument("--pad_to_max_length", help="overwrite cache of the tokenized dataset",action="store_true")

data_args = parser.parse_args()

def read_chunked_file(file_name, n_lines):
    return pd.read_csv(file_name, 
                        sep = r"\n+", 
                        chunksize=n_lines, 
                        skip_blank_lines=True, 
                        names=["text"],
                        engine='python')

def get_chunked_data(files, data_args):   
    large_data_df = pd.DataFrame()
    n_data_files = len(files)
    
    for file_num, file_path in enumerate(files):
        if os.path.isdir(file_path): continue #ignore folders
        n_lines = data_args.n_lines_per_chunk # a million lines by default

        chunk_with_million_lines = read_chunked_file(file_path, n_lines)    
        for i, data_df in enumerate(chunk_with_million_lines):
            if len(data_df) < n_lines:
                large_data_df = large_data_df.append(data_df)
                if file_num != n_data_files-1 : continue
            else:
                large_data_df = large_data_df.append(data_df)
                
            temp = large_data_df
            large_data_df = pd.DataFrame()
            yield file_num, temp.reset_index(drop=True)

def tokenize_dataset(datasets, tokenizer, data_args):
    
    column_names = ['text']
    tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = data_args.max_seq_length

    # we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_special_tokens_mask=False)

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // max_seq_length) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
    )
        
    return copy.deepcopy(tokenized_datasets)

          
def save_as_npy(serialized_dataset, folder_name, filename, counter, n_samples_per_tfrecord):
    for i in range(0, len(serialized_dataset), n_samples_per_tfrecord):
        if len(serialized_dataset[i:i+n_samples_per_tfrecord]) < n_samples_per_tfrecord:
            return counter, serialized_dataset[i:] #returning the remaininf sample to append to next file

        npy_filepath = os.path.join(folder_name, f"{filename}_{counter}" + ".npy")
        counter += 1
        np.save(npy_filepath, np.asarray(serialized_dataset[i:i+n_samples_per_tfrecord], dtype="uint8"))
    return counter, []

def main():

    data_files = {}
    if data_args.train_files_path is not None:
        if os.path.isfile(data_args.train_files_path):
            train_files = [data_args.train_files_path]
        else:
            train_files = [data_args.train_files_path+"/"+train_file
                          for train_file in os.listdir(data_args.train_files_path)]
        data_files["train"] = train_files
        
    if data_args.test_files_path is not None:
        if os.path.isfile(data_args.test_files_path):
            test_files = [data_args.train_files_path]
        else:
            test_files = [data_args.test_files_path+"/"+test_file
                         for test_file in os.listdir(data_args.test_files_path)]
        data_files["test"] = test_files
        
    
    tokenizer = GPT2Tokenizer.from_pretrained(data_args.model_tokenizer , 
                                              max_position_embeddings=data_args.max_seq_length)

    for file_type in data_files: 
        # here file_type is train or test files
        counter = 0
        remaining_samples = []
        folder_name = f"npy_{file_type}"
        if not os.path.isdir(folder_name): 
            os.mkdir(folder_name)
        n_files = len(data_files[file_type])
        for file_num, large_data_df in get_chunked_data(data_files[file_type], data_args):
            datasets = Dataset.from_pandas(large_data_df)
            tokenized_dataset = tokenize_dataset(datasets, tokenizer, data_args)["input_ids"] + remaining_samples
            counter, remaining_samples = save_as_npy(tokenized_dataset, 
                                                      folder_name, 
                                                      data_args.filename, 
                                                      counter, data_args.n_samples_per_npy)
        if len(remaining_samples)>0:
            counter, _ = save_as_npy(remaining_samples, folder_name, data_args.filename, 
                                     counter, len(remaining_samples))
    
if __name__ == "__main__":
    main()