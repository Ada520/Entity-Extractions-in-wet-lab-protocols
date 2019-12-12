import torch
import torch.nn as nn
from transformers import *

class classify_entity(nn.Module):
    def __init__(self,word2id,embed_size,tag_size,pretrained_path='scibert_scivocab_uncased/weights',weight_path='/Users/apple/Desktop/Foundations of Speech and Language Processing/WLP/scibert_scivocab_uncased'):
        super(classify_entity, self).__init__()
        self.bert_model=BertModel.from_pretrained(pretrained_path)
        self.tokenizer=BertTokenizer.from_pretrained(weight_path)
        #self.bert_model.requires_grad=False

        self.word2id=word2id
        self.linear_1=nn.Linear(embed_size,400)
        self.linear_2=nn.Linear(400,100)
        self.linear_3=nn.Linear(100,tag_size)
        self.act_func=nn.Tanh()

    def get_embed_list(self,sent):
        tokenized_text=[]
        orig_to_tok_map = []

        tokenized_text.append("[CLS]")
        for orig_token in sent:
            orig_to_tok_map.append(len(tokenized_text))
            tokenized_text.extend(self.tokenizer.tokenize(orig_token))

        tokens_tensor=tokens_tensor=torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_text)])
        segments_tensors=torch.zeros(1,len(tokenized_text),dtype=torch.long)
        with torch.no_grad():
            encoded_layers, _ = self.bert_model(tokens_tensor, segments_tensors)

        #print(len(sent),tokenized_text,orig_to_tok_map,encoded_layers[:,orig_to_tok_map,:].squeeze().shape)

        return encoded_layers[:,orig_to_tok_map,:].squeeze()

    def forward(self,sent):
        return self.linear_3(self.act_func(self.linear_2(self.act_func(self.linear_1(self.get_embed_list(sent))))))