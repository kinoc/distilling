# coding=utf-8
# Copyright 2019-present, the HuggingFace Inc. team and Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Dataset to distilled models
    adapted in part from Facebook, Inc XLM model (https://github.com/facebookresearch/XLM)
"""
import numpy as np
import torch
import os
from torch.utils.data import Dataset

from utils import logger
import json


class LazyLmSeqsDataset(Dataset):
    """Custom Dataset wrapping language modeling sequences.

    Each sample will be retrieved by indexing the list of token_ids and their corresponding lengths.

    Input:
    ------
        params: `NameSpace` parameters
        data: `List[np.array[int]]
    """

    def __init__(self, params):
        self.params = params
        self.train_data_info = {}
        self.sampleid2fileid = []
        self.create_dataset_metadata()
        self.total_samples = len(self.sampleid2fileid)
        self.print_statistics()

    def __getitem__(self, index):
        id_info = self.sampleid2fileid[index]
        file_id = id_info["file_id"]
        local_sample_id = id_info["local_sample_id"]
        
        file_name = self.train_data_info["info"][file_id]["file_name"]
        data = self.memmap_read_npy(os.path.join(self.params.data_file, file_name))
        sample = data[local_sample_id]
        return (sample, sample.shape[0])

    def __len__(self):
        return self.total_samples

    def print_statistics(self):
        """
        Print some statistics on the corpus. Only the master process.
        """
        if not self.params.is_master:
            return
        logger.info(f"{len(self)} sequences")
        # data_len = sum(self.lengths)
        # nb_unique_tokens = len(Counter(list(chain(*self.token_ids))))
        # logger.info(f'{data_len} tokens ({nb_unique_tokens} unique)')

        # unk_idx = self.params.special_tok_ids['unk_token']
        # nb_unknown = sum([(t==unk_idx).sum() for t in self.token_ids])
        # logger.info(f'{nb_unknown} unknown tokens (covering {100*nb_unknown/data_len:.2f}% of the data)')
        
    def memmap_read_npy(self, npy_file):
        return np.lib.format.open_memmap(npy_file, 
                          mode='r+', dtype=np.uint16, 
                          shape=None, fortran_order=False, version=None)
        
    def create_dataset_metadata(self):
        train_npy_path = self.params.data_file
        if not os.path.isdir(train_npy_path): raise IOError("Please provide the folder path not file.")
        train_npy_files = os.listdir(train_npy_path)
        self.train_data_info["info"] = []
        file_id = len(self.train_data_info["info"])
        for npy_file in train_npy_files:
            if npy_file.endswith("npy"):
                data = self.memmap_read_npy(os.path.join(train_npy_path, npy_file))
                data_shape = data.shape
                
                info = {
                    "file_id" : file_id,
                    "file_name" : npy_file,
                    "n_samples" : data_shape[0], 
                    "sample_size" : data_shape[1],
                    "dtype" : data.dtype
                }
                self.sampleid2fileid += [{"file_id":file_id, "local_sample_id":local_sample_id} 
                                             for local_sample_id in range(data_shape[0])]
                self.train_data_info["info"].append(info)
        
#         with open("train_data_info.json", "w") as f:
#             f.write(json.dumps(self.train_data_info))
            
    
    def batch_sequences(self, batch):
        """
        Do the padding and transform into torch.tensor.
        """
        token_ids = [t[0] for t in batch]
        lengths = [t[1] for t in batch]
        assert len(token_ids) == len(lengths)

        # Max for paddings
        max_seq_len_ = max(lengths)

        # Pad token ids
        if self.params.mlm:
            pad_idx = self.params.special_tok_ids["pad_token"]
        else:
            pad_idx = self.params.special_tok_ids["unk_token"]
        tk_ = [list(t.astype(int)) + [pad_idx] * (max_seq_len_ - len(t)) for t in token_ids]
        assert len(tk_) == len(token_ids)
        assert all(len(t) == max_seq_len_ for t in tk_)

        tk_t = torch.tensor(tk_)  # (bs, max_seq_len_)
        lg_t = torch.tensor(lengths)  # (bs)
        return tk_t, lg_t