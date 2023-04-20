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
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(f"used: {device}")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b")
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/polyglot-ko-1.3b", torch_dtype='auto'
    ).to(device=device, non_blocking=True)

# 12.8B
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-12.8b")
# model = AutoModelForCausalLM.from_pretrained(
#     "EleutherAI/polyglot-ko-12.8b", torch_dtype='auto'
#     ).to(device=device, non_blocking=True)

prompt = """
MBTI는 성격 유형 검사로 개인의 성격에 대해 16가지 유형으로 표현한다. 각 유형에 대한 특징은 다음과 같다.  
"""

with torch.no_grad():
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device=device, non_blocking=True)
    gen_tokens = model.generate(tokens, do_sample=True, temperature=0.8, max_length=256)
    generated = tokenizer.batch_decode(gen_tokens)

print(generated[0])

