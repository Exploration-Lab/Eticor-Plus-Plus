import torch
import os
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline,BitsAndBytesConfig
import logging
import json
import csv
import argparse
import pandas as pd
from transformers import logging
logging.set_verbosity_error()

torch.manual_seed(42)
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-70B-Instruct")
device = "cuda:1"
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    device_map=device,
    torch_dtype=torch.bfloat16,
    attn_implementation="eager"
)
model.eval()

def generate_prompt(prompt: str) -> str:
    chat = [{"role": "user", "content": f"{prompt}"}]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

def generate_region_combinations(regions, target_region):
    result = [target_region]
    x = target_region
    for region in regions:
        if region != target_region:
            x+=' , '+region
            result.append(x)

    # Add the final correct combination with the target region
    return result[1:]
  

def generate_input(etiqutte,options):
  p1 = "Out of the options: {} Select only one option without any explanation. Which region does the etiquette '{}' belong to?"
  inputs= []
  for option in options[0:]:
    inputs.append(p1.format(option,etiqutte))
  return inputs

def final_fun_cal(df,target_region,options):
  sentences1 = list(df['data'])
  group1 =list(df['group'])
  final_output_list=[]
  for i,j in enumerate(sentences1):
    if i%10==0:
      print(i)
    out_dic={
        "etiquette":j,
        "target_region":target_region,
        "options":options,
        "group":group1[i],
        "output":[]
    }
    for input in generate_input(j,options):
      tokenized_prompt = generate_prompt(input)
      input_ids = tokenizer(tokenized_prompt, return_tensors="pt").input_ids
      outputs = model.generate(input_ids.to(device), max_new_tokens=100)
      text = tokenizer.decode(outputs[0], skip_special_tokens=True)
      lines = [line.strip() for line in text.splitlines() if line.strip()]
      generated_text = lines[-1]
      out_dic["output"].append(generated_text)
    final_output_list.append(out_dic)


  df = pd.DataFrame(final_output_list)
  f_name= target_region.lower() + "_pref_bias.csv"
  df.to_csv( f_name, index=False)


dic_region={
    "Latin america":'LA',
    "Middle-east-Africa":'MEA',
    "East-Asia":'EA',
    "North America Europe":'NE',
    "INDIA":'INDIA'
}

regions = ["INDIA","Middle-east-Africa", "East-Asia", "North America Europe","Latin america"]


for region in regions:
  print(region)
  target_region = region
  options= generate_region_combinations(regions, target_region)
  #x_name = 'final_data_grouped/final_merged_data/GROUPED/Positive/'+ dic_region[region] +'.json'
  x_name = '~/final_data_grouped/final_merged_data/GROUPED/Positive/'+ dic_region[region] +'.json'
  df1 = pd.read_json(x_name)
  final_fun_cal(df1,target_region,options)

