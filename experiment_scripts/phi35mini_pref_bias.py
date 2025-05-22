from tqdm import tqdm
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import json
import csv
import argparse

torch.manual_seed(42)

def main(args):
    device = args.device
    batch_size = args.batch_size
    response_number = args.response_number

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct", 
        device_map=device, 
        torch_dtype="auto", 
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    model.eval()

    class CustomDataset(Dataset):
        def __init__(self, dataframe):
            self.type = type(dataframe)
            self.dataframe = [data for data in dataframe]

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            if(self.type is list):
                text = self.dataframe[idx]
            else:
                text = self.dataframe.iloc[idx, 0]
            return text
        
    input_folder = 'final_data_grouped/final_merged_data/GROUPED/Positive'
    output_folder = f'response_{response_number}/pref_bias/phi35mini'
    os.makedirs(output_folder, exist_ok=True)
    existing_files = os.listdir(output_folder)

    # Regions and labels
    regions = ['EA', 'MEA', 'INDIA', 'LA', 'NE']

    p1 = "<|system|>INSTRUCTION: Out of the options: (1)India (2)Middle East (3)Latin America (4)East Asia (5)Europe. Answer in only one word without any explanation.<|end|><|user|> Which regions does the etiquette '{}' belong to?<|end|><|assistant|>"
    chat_templates_list = [p1]

    # Processing each region and predicting
    for region in regions:
        region_data = {'one':[]}
        if f'phi35mini_{region.lower()}.csv' in existing_files:
            print(f"Skipping region {region} as it already exists.")
            continue

        json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
        # Process each json file in the folder
        for json_file in json_files:
            with open(os.path.join(input_folder, json_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Loop through each dict entry
                if data[0]['region'] != region:
                    continue
                for filler_prompt,typ,cut_len in zip(chat_templates_list,['one'],[193]):
                    sentences = []
                    for entry in data:
                        sentences.append(filler_prompt.format(entry['data']))
                    dataset = CustomDataset(sentences)
                    dataloader = DataLoader(dataset,shuffle = False,batch_size = batch_size)
                    response_sentences = []
                    for batch in tqdm(dataloader):
                        input = tokenizer.batch_encode_plus(batch, return_tensors='pt', padding=True, truncation=True).to(device)
                        encoded_output = model.generate(**input,max_length = 2*len(input['input_ids'][0]),pad_token_id=tokenizer.eos_token_id,temperature = 0.3,top_p=0.9)
                        output = tokenizer.batch_decode(encoded_output,skip_special_tokens=True)
                        response_sentences.extend(output)
                        del input,encoded_output,output

                    for i in range(len(sentences)):
                        # Extracting the response
                        region_data[typ].append(response_sentences[i][cut_len+len(data[i]['data']):])
                    torch.cuda.empty_cache()
                    
                final_data = []
                for i in range(len(dataset)):
                    final_data.append([data[i]['data'], data[i]['group'], data[i]['label'], region_data['one'][i]])
                
                
                output_csv = os.path.join(output_folder, f'phi35mini_pref_bias_{region.lower()}.csv')
                with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    # Write header
                    writer.writerow(['sentence', 'group', 'true_label', 'one'])
                    # Write the data
                    writer.writerows(final_data)

    print("Prediction and CSV generation complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for generating responses from phi mini model for the pref bias task.')
    parser.add_argument('--device', type=str,required=True, help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for generating negative examples')
    parser.add_argument('--response_number', type=int, required=True, help='Which number response to generate out of total responses')
    args = parser.parse_args()
    main(args)

