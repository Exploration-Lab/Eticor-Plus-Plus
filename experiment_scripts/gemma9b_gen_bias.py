import argparse
from tqdm import tqdm
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import logging
import json
import csv
from transformers import logging

torch.manual_seed(42)
# Set the logging level to suppress warnings
logging.set_verbosity_error()
# Set the access token for Hugging Face gated models
access_token = ""

def main(args):
    device = args.device
    batch_size = args.batch_size
    response_number = args.response_number

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()

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
        return f"""<bos><start_of_turn>user
        I will provide you with an etiquette from one region, and you need to provide, concisely in one sentence, the main corresponding etiquette from the other specified region.
        In the context of {group}, Etiquette in {region1} is: '{prompt}'.
        <end_of_turn>
        <start_of_turn>model
        Corresponding Etiquette in {region2} is: """
    

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
    output_folder = f'response_{response_number}/gen_bias/gemma9b'
    os.makedirs(output_folder, exist_ok=True)
    existing_files = os.listdir(output_folder)
    # Regions and labels
    max_length_prompt = args.max_length_prompt
    regions = ['EA', 'MEA', 'INDIA', 'LA', 'NE']

    # Processing each region and predicting
    for region in regions:
        # Check if the CSV response file already exists for a region (for continuity purpose)
        # If it exists, skip the region and continue to the next region.
        if f'gemma9b_gen_bias_{region.lower()}.csv' in existing_files:
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
                    dataloader = DataLoader(dataset,shuffle = False,batch_size = batch_size)
                    response_sentences = []
                    for sentence in tqdm(dataloader):
                        input = tokenizer.batch_encode_plus(sentence, return_tensors='pt', padding=True, truncation=True).to(device)
                        encoded_output = model.generate(**input,max_length = 2*len(input['input_ids'][0]),pad_token_id=tokenizer.eos_token_id,temperature = 0.3,top_p=0.9)
                        output = tokenizer.batch_decode(encoded_output,skip_special_tokens=True)
                        response_sentences.extend(output)
                        del input,encoded_output,output

                    for i in range(len(sentences)):
                        # Extract the response for the specific region
                        region_data[r].extend([response_sentences[i][299+len(sentences[i])+len(groups[i])+r_len+reg_len:]])
                    torch.cuda.empty_cache()

                final_region_data = []
                for i in range(len(sentences)):
                    final_region_data.append([sentences[i], data[i]['group'], data[i]['label'], region_data['EA'][i], region_data['MEA'][i], region_data['INDIA'][i], region_data['LA'][i], region_data['NE'][i]])

        output_csv = os.path.join(output_folder, f'gemma9b_gen_bias_{region.lower()}.csv')
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['sentence', 'group', 'true_label', 'EA', 'MEA', 'INDIA', 'LA', 'NE'])
            # Write the data
            writer.writerows(final_region_data)

    print("Prediction and CSV generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating responses from gemma model for the gen bias task.')
    parser.add_argument('--device', type=str,required=True, help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generating negative examples')
    parser.add_argument('--max_length_prompt', type=int, default=100, help='Maximum length of prompt')
    parser.add_argument('--response_number', type=int, required=True, help='Which number response to generate out of total responses')
    args = parser.parse_args()
    main(args)