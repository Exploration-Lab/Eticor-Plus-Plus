import argparse
from tqdm import tqdm
import torch
import os
from torch.utils.data import Dataset
import logging
import pandas as pd
import json
from transformers import logging
import requests

torch.manual_seed(42)
# Set the logging level to suppress warnings
logging.set_verbosity_error()

def main(args):

    region_mapper = {'EA':'East Asia','MEA':'Middle East','INDIA':'India','LA':'South America','NE':'Europe'}
    def generate_prompt(prompt: str,group,region1,region2):
        """
        Generates a prompt for the model based on the input sentence.
        Args:
            prompt (str): The input sentence.
            group (str): The group associated with the input sentence.
            region1 (str): The first region for comparison.
            region2 (str): The second region for comparison.
        Returns:
            str: The formatted prompt for the model.
        """
        return f"""You are an expert in cultural etiquettes across various countries. I will provide you with an etiquette from one region, and you need to provide, concisely in one sentence, the main corresponding etiquette from the other specified region.
        In the context of {group}, Etiquette in {region1} is: '{prompt}'
        Corresponding Etiquette in {region2} is: """.strip()
    
    def get_answer(sentence):
        f = False
        api_key = ""
        api_endpoint = "https://api.openai.com/v1/chat/completions"
        data = {
            "model":"gpt-4o",
            "messages":[
                {"role":"user","content":sentence}
            ]
        }
        response = requests.post(
            api_endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json=data
        )

        # Process the response
        if response.status_code == 200:
            result = response.json()
            assistant_response = result["choices"][0]["message"]["content"]
            return assistant_response
        else:
            return "NULL"
        

    input_folder = 'final_data_grouped/final_merged_data/GROUPED/Positive'
    output_folder = 'response_1/gen_bias/chatgpt'
    os.makedirs(output_folder, exist_ok=True)
    # Regions and labels
    max_length_prompt = args.max_length_prompt
    regions = ['EA', 'MEA', 'INDIA', 'LA', 'NE']

    # Processing each region and predicting
    for region in regions:
        if os.path.exists(f'response_1/gen_bias/chatgpt/chatgpt_gen_bias_{region.lower()}.csv'):
            df = pd.read_csv(f'response_1/gen_bias/chatgpt/chatgpt_gen_bias_{region.lower()}.csv')
        else:
            df = pd.DataFrame(columns=['sentence', 'group', 'true_label', 'EA', 'MEA', 'INDIA', 'LA', 'NE'])
            
        l = len(df)
        print(region,l)

        json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
        # Process each json file in the folder
        for json_file in json_files:
            with open(os.path.join(input_folder, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Loop through each dict entry
                if data[0]['region'] != region:
                    continue
                print(region)
                sentences = []
                groups =[]
                for entry in data:
                    if len(entry['data']) <=max_length_prompt:
                        sentences.append(entry['data'])
                        groups.append(entry['group'])
                
                for i,sent in enumerate(tqdm(sentences)):
                    if(i<600+l):
                        continue
                    elif(i>700):
                        break
                    response_sentences = [None]*len(regions)
                    for j,r in enumerate(regions):
                        if r == region:
                            continue
                        batch = generate_prompt(sent,groups[i],region_mapper[region],region_mapper[r])

                        response = get_answer(batch)
                        response_sentences[j] = response
                    df = pd.concat([df,pd.DataFrame([[sent, data[i]['group'], data[i]['label'], response_sentences[0],response_sentences[1],response_sentences[2],response_sentences[3],response_sentences[4]]],columns=['sentence', 'group', 'true_label', 'EA', 'MEA', 'INDIA', 'LA', 'NE'])])
                    df.to_csv(f'response_1/gen_bias/chatgpt/chatgpt_gen_bias_{region.lower()}.csv',index=False)
                    

    print("Prediction and CSV generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating response from chatgpt for the gen bias task.')
    parser.add_argument('--max_length_prompt', type=int, default=100, help='Maximum length of prompt')
    args = parser.parse_args()
    main(args)