"""
This script computes the similarity of sentences across different regions using a pre-trained sentence transformer model.
"""

import argparse
import pandas as pd
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader
import pickle
from sentence_transformers import SentenceTransformer,util

def main(args):
    data_folder = args.data_folder
    device = args.device
    model_name = args.model_name
    threshold = 0.5 if args.threshold is None else args.threshold
    region_files = os.listdir(data_folder)
    """
    Please note if the [:3] is present in the code below, it shortens the name of the region.
    """
    region_names = [file.split('.')[0].lower() for file in region_files]
    # region_files
    data = {}
    for i,file in enumerate(region_files):
        df = pd.read_excel(data_folder+'/'+file).dropna()
        try:
            data[region_names[i]] = df.iloc[df['Etiquettes'].str.len().argsort()]
            data[region_names[i]].reset_index(drop=True, inplace=True)
            print(len(data[region_names[i]]))
        except KeyError:
            print("key error happened")
            try:
                df.columns = ['sentences']
                data[region_names[i]] = df.iloc[df['sentences'].str.len().argsort()]
                data[region_names[i]].reset_index(drop=True, inplace=True)
            except Exception as e:
                print(f"file {file} failed with error {e}")
    df.reset_index(drop=True, inplace=True)



    class CustomDataset(Dataset):
        def __init__(self, dataframe):
            self.type = type(dataframe)
            self.dataframe = dataframe

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            if(self.type is list):
                text = self.dataframe[idx]
            else:
                text = self.dataframe.iloc[idx, 0]
            return text

    batch_size = 256
    dataset = {region:None for region in region_names}
    dataloader = {region:None for region in region_names}
    for k,v in data.items():
        dataset[k] = CustomDataset(v)
        dataloader[k] = DataLoader(dataset[k], batch_size=batch_size, shuffle=False)


    sim_model = SentenceTransformer(model_name,device = device)

    # Precompute and store embeddings for each sentence in every region
    embeddings_dict = {}

    for region, data in dataset.items():
        embeddings_dict[region] = sim_model.encode(data, convert_to_tensor=True)

    # Compute similar sentences across regions
    similarity_dict = {}

    # Compute similar sentences across regions
    for region, data in dataset.items():
        similarity_dict[region] = {}
        for i, sentence_embedding in enumerate(tqdm(embeddings_dict[region])):
            similarity_dict[region][i] = {}
            for region1, data1 in dataset.items():
                if region == region1:
                    continue
                similarities = util.cos_sim(sentence_embedding, embeddings_dict[region1])
                similar_sentences = [data1[j] for j, score in enumerate(similarities[0]) if score >= threshold]
                similarity_dict[region][i][region1] = similar_sentences

    # Save similarity results to disk
    with open(f'{args.output_filename}.pkl', 'wb') as f:
        pickle.dump(similarity_dict, f)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Script for precomputing sentence embeddings.')
    parser.add_argument('--data_folder', type=str,required=True , help='Path to the folder containing the Positive Labels')
    parser.add_argument('--model_name', type=str, default='all-mpnet-base-v2', help='Name of the sentence embedding model to use')
    parser.add_argument('--device', type=str,required=True, help='Device to run the model on (e.g., "cuda" or "cpu")')
    parser.add_argument('--threshold', type=float, default=0.9, help='Threshold for cosine similarity')
    parser.add_argument('--output_filename', type=str,required=True, help='name of the outputfile wheter countrywise or regionwise')
    args = parser.parse_args()
    main(args)
