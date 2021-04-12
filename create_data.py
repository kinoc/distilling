import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset

import transformers
from transformers import GPT2Tokenizer
from transformers.utils import check_min_version

import argparse
import pickle

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0.dev0")

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--preprocessing_num_workers", default=1)
parser.add_argument("--max_seq_length", default=1024, help="sequence length")
parser.add_argument("--train_files_path", help="folder path contaning train files", required=True)
parser.add_argument("--test_files_path", help="folder path contaning train files")
parser.add_argument("--overwrite_cache", help="overwrite cache of the tokenized dataset",action="store_true")
parser.add_argument("--line_by_line", help="overwrite cache of the tokenized dataset",action="store_true")
parser.add_argument("--pad_to_max_length", help="overwrite cache of the tokenized dataset",action="store_true")



data_args = parser.parse_args()


def main():

#     datasets = load_dataset(extension, data_files=data_files)
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

    
    datasets = load_dataset('text', data_files=data_files)
    column_names = ['text']
    tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B', 
                                              max_position_embeddings=data_args.max_seq_length)
    tokenizer.pad_token = tokenizer.eos_token

    max_seq_length = data_args.max_seq_length
    text_column_name = "text"
    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
                # receives the `special_tokens_mask`.
                return_special_tokens_mask=False,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=False)

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

    for key in tokenized_datasets:
        file_type = key #train ot test file
        with open(f"{file_type}_data.pkl", "wb") as f:
            pickle.dump(tokenized_datasets[file_type]['input_ids'], f)


if __name__ == "__main__":
    main()
