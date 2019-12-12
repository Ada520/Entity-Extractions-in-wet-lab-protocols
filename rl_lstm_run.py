from get_data_lstm import Data_Load
import utils
from BiLSTM import BiLSTM
from CriticNetwork import Critic_Network

import numpy as np

import stanfordnlp

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import warnings
import sys

warnings.simplefilter("ignore")
torch.manual_seed(1)

doc_num=sys.argv[1]   #python lstm_run.py 500 300
embed_dim=sys.argv[2]
error_path='errorcase_lstm_rl'

def evaluate(model,o_idx,output_dict=True,plot_matrix=False):
    with torch.no_grad():
        y_true=[]
        y_pred=[]
        for i in range(len(X_test)):
            numer_Y=[tag_to_ix[y] for y in y_test[i]]
            tag_seq,max_prob_list,sample_action_list,V,hidden_list=model(X_test[i][0],X_test[i][1],critic_net)
            #_,tag_seq=predict.max(1)
            y_true.extend(numer_Y)
            y_pred.extend(tag_seq)


        y_true=np.array(y_true)
        y_pred=np.array(y_pred)

        exclude_o_idx=np.where(y_true!=o_idx)
        y_pred_without_o=y_pred[exclude_o_idx]
        y_without_o=y_true[exclude_o_idx]

        y_pred_without_o_class=[id_to_tag[y] for y in y_pred_without_o]
        y_without_o_class=[id_to_tag[y] for y in y_without_o]

        #print(type(y_without_o),type(y_pred))
        #print(labels)
        perf=classification_report(y_without_o_class,y_pred_without_o_class,output_dict=output_dict,labels=labels)
        
        if plot_matrix:
            #print(__doc__)
            np.set_printoptions(precision=2)
            #print(len(X_test[0][0]),len(y_true),len(y_pred))
            f=open(error_path,'w')
            acc=0
            for i in range(len(X_test)):
                for j in range(len(X_test[i][0])):
                    if y_true[acc]!=y_pred[acc]:
                        f.write(id2word[X_test[i][0][j][4].item()]+" "+id_to_tag[y_true[acc]]+" "+id_to_tag[y_pred[acc]]+"\n")
                    acc+=1
            f.close()
            
            utils.plot_confusion_matrix(y_true,y_pred,np.array(not_removed_label))
            #print(666,y_without_o,y_pred)
            utils.plot_confusion_matrix(y_true,y_pred,np.array(not_removed_label),normalize=True)
            plt.show()

    return perf

pipeline=stanfordnlp.Pipeline()


dep_type2id={'<PAD>': 0, 'compound': 1, 'cc': 2, 'conj': 3, 'det': 4, 'amod': 5, 'obj': 6, 'punct': 7, 'advcl': 8, 'mark': 9, 'case': 10, 'obl': 11, 'nummod': 12, 'appos': 13, 'parataxis': 14, 'nmod': 15, 'advmod': 16, 'acl': 17, 'compound:prt': 18, 'nsubj': 19, 'xcomp': 20, 'nsubj:pass': 21, 'aux': 22, 'aux:pass': 23, 'obl:npmod': 24, 'list': 25, 'fixed': 26, 'flat': 27, 'nmod:poss': 28, 'goeswith': 29, 'cop': 30, 'nmod:tmod': 31, 'obl:tmod': 32, 'ccomp': 33, 'expl': 34, 'nmod:npmod': 35, 'vocative': 36, 'acl:relcl': 37, 'csubj': 38, 'cc:preconj': 39, 'discourse': 40, 'det:predet': 41, 'orphan': 42, 'iobj': 43}
pos_tag2id={'NN': 0, 'JJS': 1, 'CC': 2, 'VB': 3, 'DT': 4, 'JJ': 5, '.': 6, 'NNS': 7, 'TO': 8, 'IN': 9, '(': 10, 'CD': 11, 'SYM': 12, ')': 13, ':': 14, 'NNP': 15, 'VBG': 16, 'VBN': 17, 'VBP': 18, ',': 19, 'RB': 20, 'JJR': 21, 'RP': 22, 'VBZ': 23, 'MD': 24, 'VBD': 25, 'PRP': 26, 'PRP$': 27, '#': 28, 'FW': 29, 'EX': 30, 'LS': 31, 'WDT': 32, 'NNPS': 33, 'WRB': 34, 'HYPH': 35, 'RBS': 36, '``': 37, "''": 38, 'RBR': 39, 'POS': 40, 'PDT': 41, 'UH': 42, 'WP': 43, 'JJ|NN': 44, 'AFX': 45}

dl=Data_Load('train',pipeline,dep_type2id,pos_tag2id)
matrix,word2id,id2word,X_dummy,Y,files_len_list=dl.get_X_Y(int(doc_num))

#matrix=torch.tensor(matrix,dtype=torch.long)
#X_dummy=torch.tensor(X_dummy,dtype=torch.float)
print("load embedding...")
embeddings_matrix=utils.get_embeddings("gensim_glove_vectors"+embed_dim+"d.txt",word2id,int(embed_dim))
print("finally finish loading!!!")


#START_TAG = "<START>"
#STOP_TAG = "<STOP>"
EMBEDDING_DIM = embeddings_matrix.shape[1]*matrix.shape[1]+X_dummy.shape[1]
HIDDEN_DIM = 400

labels=list(set(Y))
not_removed_label=list(set(Y))
print(labels)
tag_to_ix=dict(zip(labels,range(len(set(Y)))))
id_to_tag=dict(zip(range(len(set(Y))),labels))
# tag_to_ix[START_TAG]=len(tag_to_ix)
# tag_to_ix[STOP_TAG]=len(tag_to_ix)

o_idx=tag_to_ix['o']
labels.remove('o')

final_X=utils.chunk_data_into_torch(matrix,files_len_list)
final_X_dummy=utils.chunk_data_into_torch(X_dummy,files_len_list)
final_Y=utils.chunk_data_into_torch(Y,files_len_list)

train_X=[]
for i in range(len(final_X)):
    train_X.append((torch.tensor(final_X[i],dtype=torch.long),torch.tensor(final_X_dummy[i],dtype=torch.float)))
X_train, X_test, y_train, y_test = train_test_split(train_X, final_Y, test_size=0.2)

best=0
model = BiLSTM(len(word2id), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM,embedding_matrix=embeddings_matrix)
critic_net=Critic_Network(HIDDEN_DIM*2)
#model.load_state_dict(torch.load('bestmodel'))
#model.init_weights()
model.train()
critic_net.train()

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-3)
crt_optimizer=optim.Adam(filter(lambda p: p.requires_grad, critic_net.parameters()),lr=5e-3)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)


try:
    for epoch in range(40):
        print("Epoch:",epoch)
        for i in range(len(X_train)):
            model.zero_grad()
            critic_net.zero_grad()

            numer_Y=torch.tensor([tag_to_ix[y] for y in y_train[i]],dtype=torch.long)
            max_list,max_prob_list,sample_action_list,V,hidden_list=model(X_train[i][0],X_train[i][1],critic_net)
            reward=[1 if max_list[j]==numer_Y[j] else 0 for j in range(len(max_list))]
            #V=[1 if sample_action_list[j]==numer_Y[j] else 0 for j in range(len(max_list))]

            n=len(max_list)

            zero = torch.tensor(0.0,requires_grad=True)
            loss=zero.clone()#torch.tensor(0.0,requires_grad=True)
            loss_prime=zero.clone()#torch.tensor(0.0,requires_grad=True)
            G=torch.zeros(len(max_list))
            delta=torch.zeros(len(max_list))
            adelta=torch.zeros(len(max_list))

            for t in range(len(max_list)):
                G[t]=sum(reward[t:t+n])+V[t+n if t+n<len(max_list) else len(max_list)-1]
                delta[t]=G[t]-V[t]
                adelta[t]=Critic_Network.adjust(max_list[t],numer_Y[t],delta[t])*delta[t]
                loss-=adelta[t]*max_prob_list[t]
                loss_prime+=delta[t]*delta[t]
            #print(loss,loss_prime)
            lamda=0.9984
            loss=lamda*loss+(1-lamda)*criterion(hidden_list,numer_Y)
            loss.backward()
            loss_prime.backward()

            optimizer.step()
            crt_optimizer.step()

        perf=evaluate(model,o_idx)
        print(perf)
        f1=perf['weighted avg']['f1-score']
        if f1>best:
            best=f1
            torch.save(model.state_dict(), 'bestmodel_lstm_rl')

except KeyboardInterrupt:
    print ("----------------- INTERRUPTED -----------------")
    model.load_state_dict(torch.load('bestmodel_lstm_rl'))
    perf=evaluate(model,o_idx,output_dict=False)
    print(perf)

model.load_state_dict(torch.load('bestmodel_lstm_rl'))
perf=evaluate(model,o_idx,output_dict=False,plot_matrix=True)
print(perf)
