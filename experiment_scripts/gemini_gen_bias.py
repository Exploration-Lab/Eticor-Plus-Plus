import argparse
from tqdm import tqdm
import torch
import os
from torch.utils.data import Dataset
import logging
import json
import csv
from transformers import logging
import google.generativeai as genai
genai.configure(api_key='')

torch.manual_seed(42)
# Set the logging level to suppress warnings
logging.set_verbosity_error()

def main(args):
    # Select the model to use
    model = genai.GenerativeModel("gemini-1.5-flash")

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
    

    class CustomDataset(Dataset):
        def __init__(self, dataframe,groups,region1,region2):
            self.type = type(dataframe)
            self.dataframe = [generate_prompt(data,group,region1,region2) for data,group in zip(dataframe,groups)]

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            if(self.type is list):
                text = self.dataframe[idx]
            else:
                text = self.dataframe.iloc[idx, 0]
            return text

    input_folder = 'final_data_grouped/final_merged_data/GROUPED/Positive'
    output_folder = 'response_1/gen_bias/gemini'
    os.makedirs(output_folder, exist_ok=True)
    existing_files = os.listdir(output_folder)
    # Regions and labels
    max_length_prompt = args.max_length_prompt
    regions = ['EA', 'MEA', 'INDIA', 'LA', 'NE']

    # Processing each region and predicting
    for region in regions:
        if f'gemini_gen_bias_{region.lower()}.csv' in existing_files:
            print(f"Skipping region {region} as it already exists.")
            continue
        region_data = {r: [] for r in regions}
        

        json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
        # Process each json file in the folder
        for json_file in json_files:
            with open(os.path.join(input_folder, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Loop through each dict entry
                if data[0]['region'] != region:
                    continue
                reg_len = len(region_mapper[region])
                sentences = []
                groups =[]
                for entry in data:
                    if len(entry['data']) <=max_length_prompt:
                        sentences.append(entry['data'])
                        groups.append(entry['group'])
                for r in regions:
                    if r == region:
                        region_data[r] = [None]*len(sentences)
                        continue
                    r_len = len(region_mapper[r])
                    dataset = CustomDataset(sentences,groups,region_mapper[region],region_mapper[r])
                    response_sentences = []
                    for batch in tqdm(dataset):
                        response = model.generate_content(batch,
                                generation_config=genai.types.GenerationConfig(
                                candidate_count=1,
                                max_output_tokens=len(batch),
                                temperature=0.1,
                            ),)
                        response_sentences.append(response.text)

                    for i in range(len(sentences)):
                        region_data[r].extend([response_sentences[i]])

                final_region_data = []
                for i in range(len(sentences)):
                    final_region_data.append([sentences[i], data[i]['group'], data[i]['label'], region_data['EA'][i], region_data['MEA'][i], region_data['INDIA'][i], region_data['LA'][i], region_data['NE'][i]])
                
        output_csv = os.path.join(output_folder, f'gemini_gen_bias_{region.lower()}.csv')
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['sentence', 'group', 'true_label', 'EA', 'MEA', 'INDIA', 'LA', 'NE'])
            # Write the data
            writer.writerows(final_region_data)

    print("Prediction and CSV generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating response using Gemini model for the gen bias task.')
    parser.add_argument('--max_length_prompt', type=int, default=100, help='Maximum length of prompt')
    args = parser.parse_args()
    main(args)