import torch
from torch.utils.data.dataset import Dataset
import json
import pickle
from sklearn.model_selection import train_test_split
import os

class TransformerYelpDataset(Dataset):
    def __init__(self, tokenizer, data, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        note = []
        reviews, avg_score = self.data[index]['reviews'], self.data[index]["avg_score"]
        reviews = " ".join(reviews)
        scores = self.data[index]['scores']
        return reviews, scores, avg_score

    def collate_fn(self, data):
        #List of sentences and frames [B,]
        inputs, scores, avg_score = zip(*data)
        encodings = self.tokenizer.batch_encode_plus(
            list(inputs),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors='pt'
        ) #[B, max_len in this batch]
        
        scores = torch.FloatTensor(scores).view(-1, 10)
        score = torch.FloatTensor(avg_score).view(-1, 1)

        if 'token_type_ids' not in encodings.keys():
            encodings['token_type_ids'] = torch.zeros((1,))
        return encodings['input_ids'], encodings['attention_mask'], encodings['token_type_ids'], score, scores

    def __len__(self):
        return len(self.data)
