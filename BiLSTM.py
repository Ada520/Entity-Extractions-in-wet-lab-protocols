import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
import numpy as np

START_TAG = "<START>"
STOP_TAG = "<STOP>"

class BiLSTM(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim,embedding_matrix):
        super(BiLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeds.weight=nn.Parameter(embedding_matrix)

        self.lstm=nn.LSTMCell(embedding_dim,hidden_dim)
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,num_layers=1, bidirectional=True)


        #self.critic_linear = nn.Linear(self.hidden_dim*2, 1)
        self.actor_linear = nn.Linear(self.hidden_dim*2, self.tagset_size)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.hidden = self.init_hidden()

    def init_weights(self):
        for name, param in self.lstm.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal_(param)

        nn.init.xavier_uniform_(self.hidden2tag.state_dict()['weight'])
        nn.init.xavier_uniform_(self.critic_linear.state_dict()['weight'])
        nn.init.xavier_uniform_(self.actor_linear.state_dict()['weight'])
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        self.hidden2tag.bias.data.fill_(0)

    def init_hidden(self):
        return (torch.randn(1, self.hidden_dim),
                torch.randn(1, self.hidden_dim))

    def forward(self, sentence, dummy_feats,critic_net,hx=None,cx=None):

        if hx is not None and cx is not None:
            self.hidden=(hx,cx)
        self.hidden=self.init_hidden()
        #values = []
        #log_probs = []
        #rewards = []
        #entropies = []
        sample_action_list=[]
        max_action_list=[]
        max_prob_list=[]
        V_list=[]
        hidden_list=torch.zeros(sentence.shape[0],self.tagset_size)

        #self.hidden = self.init_hidden()
        for k in range(sentence.shape[0]):
            word=sentence[k]
            dummy_word=dummy_feats[k]
            embeds = self.word_embeds(word.reshape(1,-1)).reshape(1,-1)
            dummy_word=dummy_word.unsqueeze(0)
            embeds=torch.cat((embeds,dummy_word),dim=1)
            self.hidden = self.lstm(embeds,self.hidden)

            critic=critic_net(torch.cat((self.hidden[0],self.hidden[1]),dim=1)).item()
            V_list.append(critic)

            logit=self.actor_linear(torch.cat((self.hidden[0],self.hidden[1]),dim=1))
            hidden_list[k]=logit

            prob = F.softmax(logit)
            logProb=F.log_softmax(logit)
            sample_action = prob.multinomial(1)
            max_action=prob.max(1)[1]
            sample_action_list.append(sample_action.item())
            max_action_list.append(max_action.item())
            max_prob_list.append(logProb[0][max_action.item()].item())

        
        return max_action_list,max_prob_list,sample_action_list,V_list,hidden_list

