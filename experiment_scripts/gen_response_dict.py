"""
This script is used to generate a dictionary of embeddings of generated responses of a model
in the gen bias task. Which will be used to calculate the GAS score.
"""

import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='The model whose responses to consider.')
parser.add_argument('--device', type=str, required=True, help='The device to use for the sentence transformer.')
parser.add_argument('--response_number', type=int, required=True, help='The response number to consider.')
args = parser.parse_args()

response_number  = args.response_number
model = args.model
device = args.device

gen_resp_path = f'response_{response_number}/gen_bias/{model}'
files = os.listdir(gen_resp_path)
regions = [file.split('.csv')[0].split('_')[-1].upper() for file in files]

sim_model = SentenceTransformer("all-mpnet-base-v2",device = device)
gen_response_dict = {r:{reg:[] for reg in regions} for r in regions}

for file in files:
    df = pd.read_csv(os.path.join(gen_resp_path,file))
    file_reg = file.split('.csv')[0].split('_')[-1].upper()
    for r in regions:
        if r ==file_reg:
            continue
        gen_response_dict[file_reg][r].extend(sim_model.encode(df[r].values))
    
torch.save(gen_response_dict,f'{model}_gen_response_{response_number}_dict.pt')