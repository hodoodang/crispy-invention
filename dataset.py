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

import json
import os
from glob import glob

import torch
import numpy as np
import pandas as pd


def get_contents(file_path: str):
    with open(file_path, encoding="utf-8") as f:
        file = json.load(f)

    for data in file.values():
        if 'text' in data:
            for news in data['text']:
                yield news['content']
        elif isinstance(data, list):
            for news_list in data:
                document = ""
                for sent in news_list['content']:
                    document += sent['sentence'] + " "
                yield document

    if "사회일발" in file_path:
        for data in file['named_entity']:
            document = ""
            for sent in data['content']:
                document += sent['sentence'] + " "
            yield document


if __name__ == "__main__":
    dir_list = glob(".\\data/*/*")
    label = [cnt for cnt in range(0, len(dir_list))]
    file_list = []

    for dir in dir_list:
        file_path_list = os.path.join(dir + "/*.json")
        file_list.append(glob(file_path_list))

    dataset_list = []
    for label, category in zip(label, file_list):
        for filename in category:
            for content in get_contents(filename):
                dataset_list.append([label, content])

    dataset = pd.DataFrame(dataset_list, columns=['label', 'content'])

    dataset_dict = dataset.to_dict(orient="records")
    sample_dict = dataset[:10].to_dict(orient="records")

    with open('.\\data\\web\\web_dataset.json', 'w', encoding="utf-8") as outfile:
        json.dump(dataset_dict, outfile, indent=4, ensure_ascii=False)

    with open('.\\data\\web\\sample_dataset.json', 'w', encoding="utf-8") as outfile:
        json.dump(sample_dict, outfile, indent=4, ensure_ascii=False)
