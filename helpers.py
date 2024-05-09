from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import re

class DatasetClass(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.df = df
        self.text = df['text']
        #self.targets = self.df[['label']].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs["token_type_ids"].flatten(),
            #'targets': torch.FloatTensor(self.targets[index])
            # 'targets' : torch.FloatTensor(self.targets[index].astype(np.float32))  # Convert to PyTorch FloatTensor
        }
    
class FineTunedBERTClass(torch.nn.Module):
    def __init__(self):
        super(FineTunedBERTClass, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased', return_dict=True)
        self.linear = torch.nn.Linear(768, 1)

    def forward(self, input_ids, attn_mask, token_type_ids):
        output = self.bert_model(
            input_ids,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids
        )
        output = self.linear(output.pooler_output)
        return output



def break_paragraph_into_chunks(paragraph, k):
    # Split the paragraph into individual sentences
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', paragraph)

    # Initialize an empty list to store pairs of sentences
    chunk_list = []

    # Iterate over the sentences in pairs
    for i in range(0, len(sentences), k):
        # Combine consecutive sentences into a chunk
        chunk = sentences[i:i + k]

        # If there are no more sentences left, break
        if not chunk:
            break

        # If there's only one sentence in the chunk, append it as is
        if len(chunk) == 1:
            chunk_list.append(chunk[0])
        else:
            # Join the sentences into a single string
            chunk = ' '.join(chunk)
            # Add the chunk to the list
            chunk_list.append(chunk)

    return chunk_list

def model_load():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    path="my_hatespeechDetectionModel_bert.pth"
    model = FineTunedBERTClass()
    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    model.to(device)
    return model

def tokenizer_load():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    return tokenizer

def evaluate_model(model, dataloader):
    model.eval()
    all_outputs = []
    all_probabilities = []
    device = torch.device('cpu')
    with torch.no_grad():
        for data in dataloader:
            ids = data['input_ids'].to(device)
            mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)

            outputs = model(ids, mask, token_type_ids)
            probabilities = torch.sigmoid(outputs)

            all_outputs.extend(torch.round(probabilities).cpu().detach().numpy())
            all_probabilities.extend(probabilities.cpu().detach().numpy())

    all_outputs = np.array(all_outputs)
    all_probabilities = np.array(all_probabilities)

    print("Predicted Output (Binary):\n", all_outputs)
    print("Predicted Probabilities:\n", all_probabilities)

    return all_outputs, all_probabilities



def inference(para, model, tokenizer):  
     
    k=1
    chunk_list = break_paragraph_into_chunks(para,k)
    print(chunk_list)
    test_df = pd.DataFrame(chunk_list, columns=['text'])

    # Add an 'ID' column starting from 1
    test_df.insert(0, 'ID', range(1, len(test_df) + 1))

    print(test_df.head())
    MAX_LEN=256
    TEST_BATCH_SIZE=32

    test_dataset = DatasetClass(test_df, tokenizer, MAX_LEN)
    
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=0
    )

    global all_outputs
    all_outputs = []

    predicted_output, predicted_probabilities = evaluate_model(model, test_data_loader)
    print(predicted_output)
    return chunk_list, predicted_probabilities
    # return df