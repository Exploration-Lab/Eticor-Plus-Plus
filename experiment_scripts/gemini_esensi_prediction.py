import os
import google.generativeai as genai
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import logging
import json
import csv
from transformers import logging
torch.manual_seed(42)

# Set the logging level to suppress warnings
logging.set_verbosity_error()

# Set the API key for Google Generative AI
genai.configure(api_key='')

def main():

    # Select the model of your choice
    model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_prompt(prompt: str) -> str:
        """
        Generates a prompt for the model based on the input sentence.
        Args:
            prompt (str): The input sentence.
        Returns:
            str: The formatted prompt for the model.
        """
        return f"""Answer in one word if the given cultural etiquette is (1)acceptable or (2)unacceptable.'{prompt}'""".strip()

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
        
    # For Gemini, only one response is generated    
    input_folder = 'final_data_grouped/final_merged_data/GROUPED'
    output_folder = 'response_1/esensi/gemini'
    os.makedirs(output_folder, exist_ok=True)
    existing_files = os.listdir(output_folder)
    # Regions and labels
    regions = ['EA', 'MEA', 'INDIA', 'LA', 'NE']

    # Processing each region and predicting
    for region in regions:
        # Check if the CSV response file already exists for a region (for continuity purpose)
        # If it exists, skip the region and continue to the next region.
        region_data = []
        if f'gemini_{region.lower()}.csv' in existing_files:
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
                    response_sentences = []
                    for batch in tqdm(dataset):
                        f=False
                        response = None
                        while(f==False):
                            try:
                                response = model.generate_content(batch,
                                    generation_config=genai.types.GenerationConfig(
                                    candidate_count=1,
                                    max_output_tokens=50,
                                    temperature=0.1,
                                ),)
                                f=True
                            except:
                                continue
                        response_sentences.append(response.text)

                    for i in range(len(dataset)):
                        region_data.append([sentences[i], data[i]['group'], data[i]['label'], response_sentences[i]])
                    
        output_csv = os.path.join(output_folder, f'gemini_{region.lower()}.csv')
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['sentence', 'group', 'true_label', 'gemini_sentence'])
            # Write the data
            writer.writerows(region_data)

    print("Prediction and CSV generation complete.")

if __name__ == '__main__':
    main()