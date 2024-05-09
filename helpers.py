from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import re
import torch.nn as nn
# import emoji as emoji
import string
# import nltk
# from nltk.tokenize import TweetTokenizer


# def tokenize(tweet):
#     # instantiate the tokenizer class
#     tokenizer = TweetTokenizer(preserve_case=False, 
#                               strip_handles=True,
#                               reduce_len=True)

#     # tokenize the tweets
#     tweet_tokens = tokenizer.tokenize(tweet)

#     tweets_clean = []
#     for word in tweet_tokens: # Go through every word in your tokens list
#         if word not in string.punctuation:  # remove punctuation
#             tweets_clean.append(word)
#     result = tweets_clean
#     return " ".join(result)

# emoticons = [':-)', ':)', '(:', '(-:', ':))', '((:', ':-D', ':D', 'X-D', 'XD', 'xD', 'xD', '<3', '</3', ':\*',
#                  ';-)',
#                  ';)', ';-D', ';D', '(;', '(-;', ':-(', ':(', '(:', '(-:', ':,(', ':\'(', ':"(', ':((', ':D', '=D',
#                  '=)',
#                  '(=', '=(', ')=', '=-O', 'O-=', ':o', 'o:', 'O:', 'O:', ':-o', 'o-:', ':P', ':p', ':S', ':s', ':@',
#                  ':>',
#                  ':<', '^_^', '^.^', '>.>', 'T_T', 'T-T', '-.-', '*.*', '~.~', ':*', ':-*', 'xP', 'XP', 'XP', 'Xp',
#                  ':-|',
#                  ':->', ':-<', '$_$', '8-)', ':-P', ':-p', '=P', '=p', ':*)', '*-*', 'B-)', 'O.o', 'X-(', ')-X']

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

from torch.nn.utils.rnn import pack_padded_sequence

# class BERT_LSTM_Model(nn.Module):

#   def __init__(self, bert, n_class):
#     dropout_rate = 0.2
#     lstm_hidden_size = None

#     super(BERT_LSTM_Model, self).__init__()
#     self.bert = bert

#     if not lstm_hidden_size:
#       self.lstm_hidden_size = self.bert.config.hidden_size
#     else:
#       self.lstm_hidden_size = lstm_hidden_size
#     self.n_class = n_class
#     self.dropout_rate = dropout_rate
#     self.lstm = nn.LSTM(self.bert.config.hidden_size, self.lstm_hidden_size, bidirectional=True)
#     self.hidden_to_softmax = nn.Linear(self.lstm_hidden_size * 2, n_class, bias=True)
#     self.dropout = nn.Dropout(p=self.dropout_rate)
#     self.softmax = nn.LogSoftmax(dim=1)

#   def forward(self, sent_id, mask):
#     encoded_layers = self.bert(sent_id, attention_mask=mask, output_hidden_states=True)[2] #,output_all_encoded_layers=False)   #output_hidden_states output_hidden_states=True
#     bert_hidden_layer = encoded_layers[12]
#     bert_hidden_layer = bert_hidden_layer.permute(1, 0, 2)   #permute rotates the tensor. if tensor.shape = 3,4,5  tensor.permute(1,0,2), then tensor,shape= 4,3,5  (batch_size, sequence_length, hidden_size)

#     sents_lengths = [36 for i in range(len(sent_id))]
#     enc_hiddens, (last_hidden, last_cell) = self.lstm(pack_padded_sequence(bert_hidden_layer, sents_lengths, enforce_sorted=False)) #enforce_sorted=False  #pack_padded_sequence(data and batch_sizes
#     output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)  # (batch_size, 2*hidden_size)
#     output_hidden = self.dropout(output_hidden)
#     pre_softmax = self.hidden_to_softmax(output_hidden)

#     return self.softmax(pre_softmax)
  

class BERT_LSTM_Model(nn.Module):
    def __init__(self, bert, n_class):
        super(BERT_LSTM_Model, self).__init__()
        self.bert = bert
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, bidirectional=True)
        self.hidden_to_softmax = nn.Linear(self.bert.config.hidden_size * 2, n_class, bias=True)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        outputs = self.bert(sent_id, attention_mask=mask)
        bert_hidden_layer = outputs.last_hidden_state
        bert_hidden_layer = bert_hidden_layer.permute(1, 0, 2)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(bert_hidden_layer)
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim=1)
        output_hidden = self.dropout(output_hidden)
        pre_softmax = self.hidden_to_softmax(output_hidden)
        return self.softmax(pre_softmax)

# def preprocess(tweet):
#     result = tweet.replace('rt @','@')
#     result = result.replace('@','<user> @')
#     # it will remove hyperlinks
#     result = re.sub(r'https?:\/\/.*[\r\n]*', '<url>', result)

#     # it will remove hashtags. We have to be careful here not to remove 
#     # the whole hashtag because text of hashtags contains huge information. 
#     # only removing the hash # sign from the word
#     result = re.sub(r'#', '<hashtag>', result)

#     # Replace multiple dots with space
#     result = re.sub('\.\.+', ' ', result) 



#     # for char in result:
#     #     if emoji.is_emoji(char):
#     #         result = result.replace(char, "<emoticon >")
#     for emo in emoticons:
#         result = result.replace(emo, "<emoticon >")

#     result = tokenize(result)
#     # it will remove single numeric terms in the tweet. 
#     result = re.sub(r'[0-9]+', '<number>', result)
#     result = re.sub(r'<number>\s?st', '<number>', result)
#     result = re.sub(r'<number>\s?nd', '<number>', result)
#     result = re.sub(r'<number>\s?rd', '<number>', result)
#     result = re.sub(r'<number>\s?th', '<number>', result)

#     return result

# def pre_process_dataset(values):
#     new_values = list()
#     for value in values:
#         new_values.append(preprocess(value.lower()))
# #     print(values[:5])
# #     print(new_values[:5])
#     return new_values

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
    path="model_HS.pth"
    
    # import BERT-base pretrained model
    bert = BertModel.from_pretrained('bert-base-uncased')
    # freeze all the parameters
    for param in bert.parameters():
        param.requires_grad = False
    model = BERT_LSTM_Model(bert, 2)

    model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    model.to(device)
    return model

def tokenizer_load():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 
    return tokenizer

def sigmoid(x):
    y = x[0]
    y1 = 1/(1+np.exp(-y))
    return [y1, 1 - y1]

def evaluate_model(model, testdf, tokenizer):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_data = testdf['text'].tolist()
    # Initialize an empty DataFrame to store predictions
    predictions_df = pd.DataFrame({'id': testdf['id']})

    # Tokenize the test data
    test_seq = torch.tensor([tokenizer.encode(i, max_length=36, pad_to_max_length=True) for i in test_data])
    test_mask = torch.tensor([[float(i > 0) for i in ii] for ii in test_seq])

    # Get predictions for test data
    with torch.no_grad():
        preds = model(test_seq.to(device), test_mask.to(device))
        preds = preds.detach().cpu().numpy()

    # Get the predicted labels for 'HS'
    predictions_df['predicted_HS'] = np.argmax(preds, axis=1)
    p = []
    for i in range(testdf.shape[0]):
        p.append(sigmoid(preds[i])[1]*100)
    print(p)    
    return predictions_df['predicted_HS'], p
    # model.eval()
    # all_outputs = []
    # all_probabilities = []
    # device = torch.device('cpu')
    # with torch.no_grad():
    #     for data in dataloader:
    #         ids = data['input_ids'].to(device)
    #         mask = data['attention_mask'].to(device)
    #         token_type_ids = data['token_type_ids'].to(device)

    #         outputs = model(ids, mask, token_type_ids)
    #         probabilities = torch.sigmoid(outputs)

    #         all_outputs.extend(torch.round(probabilities).cpu().detach().numpy())
    #         all_probabilities.extend(probabilities.cpu().detach().numpy())

    # all_outputs = np.array(all_outputs)
    # all_probabilities = np.array(all_probabilities)

    # print("Predicted Output (Binary):\n", all_outputs)
    # print("Predicted Probabilities:\n", all_probabilities)

    # return all_outputs, all_probabilities

def inference(para, model, tokenizer):  
     
    k=1
    chunk_list = break_paragraph_into_chunks(para,k)
    print(chunk_list)
    test_df = pd.DataFrame(chunk_list, columns=['text'])

    # Add an 'ID' column starting from 1
    test_df.insert(0, 'id', range(1, len(test_df) + 1))

    print(test_df.head())

    global all_outputs
    all_outputs = []

    predicted_output, predicted_probabilities = evaluate_model(model, test_df, tokenizer)
    print(predicted_output)
    return chunk_list, predicted_probabilities
    # return df