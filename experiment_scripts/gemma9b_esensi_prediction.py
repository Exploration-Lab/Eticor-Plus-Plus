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


    def generate_prompt(prompt: str) -> str:
        """
        Generates a prompt for the model based on the input sentence.
        Args:
            prompt (str): The input sentence.
        Returns:
            str: The formatted prompt for the model.
        """
        chat = [
            {"role":"user","content":f"Answer in one word if the given cultural etiquette is (1)acceptable or (2)unacceptable.'{prompt}'."}
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    class CustomDataset(Dataset):
        def __init__(self, dataframe):
            self.type = type(dataframe)
            self.dataframe = [generate_prompt(data) for data in dataframe]

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            if(self.type is list):
                text = self.dataframe[idx]
            else:
                text = self.dataframe.iloc[idx, 0]
            return text
        
    input_folder = 'final_data_grouped/final_merged_data/GROUPED'
    output_folder = f'response_{response_number}/esensi/gemma9b'
    os.makedirs(output_folder, exist_ok=True)
    existing_files = os.listdir(output_folder)
    # Regions and labels
    regions = ['EA', 'MEA', 'INDIA', 'LA', 'NE']

    # Processing each region and predicting
    for region in regions:
        # Check if the CSV response file already exists for a region (for continuity purpose)
        # If it exists, skip the region and continue to the next region.
        region_data = []
        if f'gemma9b_{region.lower()}.csv' in existing_files:
            print(f"Skipping region {region} as it already exists.")
            continue
        # Process both positive and negative folders
        for label_folder in ['Positive', 'Negative']:
            folder_path = os.path.join(input_folder, label_folder)
            json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

            # Process each json file in the folder
            for json_file in json_files:
                with open(os.path.join(folder_path, json_file), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Loop through each dict entry
                    if data[0]['region'] != region:
                        continue
                    sentences = []
                    for entry in data:
                        sentences.append(entry['data'])
                    dataset = CustomDataset(sentences)
                    dataloader = DataLoader(dataset,shuffle = False,batch_size = batch_size)
                    response_sentences = []
                    for batch in tqdm(dataloader):
                        with torch.no_grad():
                            input = tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True, truncation=True).to(device)
                            encoded_output = model.generate(**input,max_new_tokens = len(input['input_ids'][0]),pad_token_id=tokenizer.eos_token_id,temperature = 0.3,top_p=0.9)
                            output = tokenizer.batch_decode(encoded_output,skip_special_tokens=True)
                            response_sentences.extend(output)
                            del input,encoded_output,output

                    for i in range(len(sentences)):
                        # Extract the relevant part of the response
                        region_data.append([sentences[i], data[i]['group'], data[i]['label'], response_sentences[i][102+len(sentences[i]):]])
                    torch.cuda.empty_cache()

        output_csv = os.path.join(output_folder, f'gemma9b_{region.lower()}.csv')
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['sentence', 'group', 'true_label', 'gemma9b_sentence'])
            # Write the data
            writer.writerows(region_data)

    print("Prediction and CSV generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating responses from gemma model for the esensitivity task.')
    parser.add_argument('--device', type=str,required=True, help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generating negative examples')
    parser.add_argument('--response_number', type=int, required=True, help='Which number response to generate out of total responses')
    args = parser.parse_args()
    main(args)