import pandas as pd
import torch
import os
from sentence_transformers import util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model name')
parser.add_argument('--device', type=str, required=True, help='Device')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--score_threshold', type=float, default=0.9, help='Score threshold')
parser.add_argument('--response_number', type=int, required=True, help='The number of the response to process.')
args = parser.parse_args()

model = args.model
response_number = args.response_number
gen_resp_path = f'response_{response_number}/gen_bias/{model}'

files = os.listdir(gen_resp_path)
regions = [f.split('.csv')[0].split('_')[-1].upper() for f in files]

gen_response_dict = torch.load(f'{model}_gen_response_{response_number}_dict.pt')

def get_similars(emb1,emb2,threshold=0.55):
    """
    Get the indexes of the similar embeddings in emb2 for the given embedding emb1.
    Args:
        emb1: The embedding to compare with.
        emb2: The list of embeddings to compare against.
        threshold: The similarity threshold.
    Returns:
        A list of indexes of the similar embeddings in emb2.
    """
    sim = util.cos_sim(emb1,np.array(emb2))
    idx = torch.where(sim[0]>threshold)
    return idx[0].tolist()

model_response_data_dict = {r:{reg:[] for reg in regions} for r in regions}
for file in files:
    df = pd.read_csv(os.path.join(gen_resp_path,file))
    file_reg = file.split('.csv')[0].split('_')[-1].upper()
    for r in regions:
        if r ==file_reg:
            continue
        model_response_data_dict[file_reg][r].extend(df[r].values)

device_nli = args.device
nli_model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large-mnli',device = device_nli)
nli_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-mnli',device = device_nli)
nli_model.to(device_nli)

nli_model_dict = {r:{'entail':0,'contradict':0} for r in regions}
batch_size = args.batch_size
score_threshold = args.score_threshold
for r in regions:
    prem_hypo_pairs = []
    for i,reg1 in enumerate(regions):
        if r == reg1:
            continue
        for j,emb1 in enumerate(gen_response_dict[reg1][r]):
            for k,reg2 in enumerate(regions):
                if(k<=i or reg2==r):
                    continue
                sim_indexes = get_similars(emb1,gen_response_dict[reg2][r])
                for idx in sim_indexes:
                    # Here we are trying to get pairs of sentences that are about the same theme.
                    # We create a combined sentence with the premise and hypothesis for nli
                    prem_hypo_pairs.append(f'{model_response_data_dict[reg1][r][j]}</s>{model_response_data_dict[reg2][r][idx]}')

    for i in tqdm(range(0,len(prem_hypo_pairs),batch_size)):
        with torch.no_grad():
            batch = prem_hypo_pairs[i:i+batch_size]
            input = nli_tokenizer.batch_encode_plus(batch, return_tensors='pt',padding=True,truncation=True).to(device_nli)
            logits = nli_model(**input)[0]
            probs = logits.softmax(dim=1)
            nli_model_dict[r]['entail']+= len(torch.where(probs[:,2]>score_threshold)[0])
            nli_model_dict[r]['contradict']+= len(torch.where(probs[:,0]>score_threshold)[0])
            torch.cuda.empty_cache()
        
torch.save(nli_model_dict,f'{model}_nli_response_{response_number}_dict.pt')
