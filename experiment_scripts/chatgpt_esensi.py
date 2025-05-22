import os
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import logging
import json
from transformers import logging
import requests
torch.manual_seed(42)
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')
# Set the logging level to suppress warnings
logging.set_verbosity_error()


def main():


    def generate_prompt(prompt: str) -> str:
        """
        Generates a prompt for the model based on the input sentence.
        Args:
            prompt (str): The input sentence.
        Returns:
            str: The formatted prompt for the model.
        """
        return f"""Answer in one word if the given cultural etiquette is (1)acceptable or (2)unacceptable.'{prompt}'""".strip()
    
    def get_answer(sentence):
        """
        Sends a request to the OpenAI API to get a response for the given sentence.
        Args:
            sentence (str): The input sentence.
        Returns:
            str: The response from the OpenAI API.
        """
        # Set up the API key and endpoint
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
    output_folder = 'response_1/esensi/chatgpt'
    os.makedirs(output_folder, exist_ok=True)
    existing_files = os.listdir(output_folder)
    # Regions and labels
    regions = ['EA', 'MEA', 'INDIA', 'LA', 'NE']

    # Processing each region and predicting
    for region in regions:
        # Here the continuation scheme is a little different. Instead of continuging from the last file,
        # we check if the file exists and then continue from the last index i.e 600 to 700.
        if os.path.exists(f'response_1/esensi/chatgpt/chatgpt_{region.lower()}.csv'):
            df = pd.read_csv(f'response_1/esensi/chatgpt/chatgpt_{region.lower()}.csv')
        else:
            df = pd.DataFrame(columns=['sentence', 'group', 'true_label', 'chatgpt_sentence'])
            
        l = len(df)
        print(region,l)
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
                    for i,batch in enumerate(tqdm(dataset)):
                        if(i<600+l):
                            continue
                        elif(i>700):
                            break
                        response = get_answer(batch)
                        df = pd.concat([df,pd.DataFrame({'sentence':batch,'group':data[i]['group'],'true_label':data[i]['label'],'chatgpt_sentence':response},index=[0])],ignore_index=True)
                        df.to_csv(f'response_1/esensi/chatgpt/chatgpt_{region.lower()}.csv',index=False)

    print("Prediction and CSV generation complete.")

if __name__ == '__main__':
    main()