#  -*- encoding: utf-8 -*-
#  Copyright (c) 2023 Teaho Sagong
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import torch
import numpy as np
from torch.utils.data import Dataset


class BertDataset(Dataset):
    def __init__(self, texts, labels, max_seq_length, vocab):
        self.texts = texts
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer.tokenize(text)[:self.max_seq_length-2]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = np.zeros(self.max_seq_length, dtype=np.int64)
        input_mask = np.zeros(self.max_seq_length, dtype=np.int64)
        segment_ids = np.zeros(self.max_seq_length, dtype=np.int64)
        for i, token in enumerate(tokens):
            input_ids[i] = self.vocab[token]
            input_mask[i] = 1
            segment_ids[i] = 0
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(input_mask, dtype=torch.long),
            'token_type_ids': torch.tensor(segment_ids, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }
