import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import numpy as np
import matplotlib.pyplot as plt
from transformers import *
import stanfordnlp
import warnings
import sys

from get_data_lstm import Data_Load
import utils
from classify_entity import classify_entity

doc_num=sys.argv[1]
bert_model_name=sys.argv[2]

pretrained_path_dict={'scibert':'scibert_scivocab_uncased/weights','basebert':'bert-base-uncased'}
pretrained_path=pretrained_path_dict[bert_model_name]
weight_path_dict={'scibert':'/Users/apple/Desktop/Foundations of Speech and Language Processing/WLP/scibert_scivocab_uncased','basebert':'bert-base-uncased'}
weight_path=weight_path_dict[bert_model_name]

warnings.simplefilter("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pipeline=stanfordnlp.Pipeline()
error_path='errorcase_'+bert_model_name

def evaluate(model,o_idx,X_test,y_test,output_dict=True,plot_matrix=False):
    with torch.no_grad():
        y_true=[]
        y_pred=[]
        for i,text in enumerate(X_test):
            numer_Y=[tag_to_ix[y] for y in y_test[i]]
            raw_output=model(text)
            _,pred_Y=torch.max(raw_output,1)
            y_true.extend(numer_Y)
            y_pred.extend(pred_Y)

        
        y_true=np.array(y_true)
        y_pred=np.array(y_pred)

        exclude_o_idx=np.where(y_true!=o_idx)
        y_pred_without_o=y_pred[exclude_o_idx]
        y_without_o=y_true[exclude_o_idx]

        y_pred_without_o_class=[id_to_tag[y] for y in y_pred_without_o]
        y_without_o_class=[id_to_tag[y] for y in y_without_o]

        perf=classification_report(y_without_o_class,y_pred_without_o_class,output_dict=output_dict,labels=labels)
        
        if plot_matrix:
            np.set_printoptions(precision=2)
            f=open(error_path,'w')
            acc=0
            for idx,text in enumerate(X_test):
                for word in text:
                    if y_true[acc]!=y_pred[acc]:
                        f.write(word+" "+id_to_tag[y_true[acc]]+" "+id_to_tag[y_pred[acc]]+"\n")
                    acc+=1
            f.close()
            
            utils.plot_confusion_matrix(y_true,y_pred,np.array(not_removed_label))
            utils.plot_confusion_matrix(y_true,y_pred,np.array(not_removed_label),normalize=True)
            plt.show()

    return perf

dep_type2id={'<PAD>': 0, 'compound': 1, 'cc': 2, 'conj': 3, 'det': 4, 'amod': 5, 'obj': 6, 'punct': 7, 'advcl': 8, 'mark': 9, 'case': 10, 'obl': 11, 'nummod': 12, 'appos': 13, 'parataxis': 14, 'nmod': 15, 'advmod': 16, 'acl': 17, 'compound:prt': 18, 'nsubj': 19, 'xcomp': 20, 'nsubj:pass': 21, 'aux': 22, 'aux:pass': 23, 'obl:npmod': 24, 'list': 25, 'fixed': 26, 'flat': 27, 'nmod:poss': 28, 'goeswith': 29, 'cop': 30, 'nmod:tmod': 31, 'obl:tmod': 32, 'ccomp': 33, 'expl': 34, 'nmod:npmod': 35, 'vocative': 36, 'acl:relcl': 37, 'csubj': 38, 'cc:preconj': 39, 'discourse': 40, 'det:predet': 41, 'orphan': 42, 'iobj': 43}
pos_tag2id={'NN': 0, 'JJS': 1, 'CC': 2, 'VB': 3, 'DT': 4, 'JJ': 5, '.': 6, 'NNS': 7, 'TO': 8, 'IN': 9, '(': 10, 'CD': 11, 'SYM': 12, ')': 13, ':': 14, 'NNP': 15, 'VBG': 16, 'VBN': 17, 'VBP': 18, ',': 19, 'RB': 20, 'JJR': 21, 'RP': 22, 'VBZ': 23, 'MD': 24, 'VBD': 25, 'PRP': 26, 'PRP$': 27, '#': 28, 'FW': 29, 'EX': 30, 'LS': 31, 'WDT': 32, 'NNPS': 33, 'WRB': 34, 'HYPH': 35, 'RBS': 36, '``': 37, "''": 38, 'RBR': 39, 'POS': 40, 'PDT': 41, 'UH': 42, 'WP': 43, 'JJ|NN': 44, 'AFX': 45}

dl_train=Data_Load('train',pipeline,dep_type2id,pos_tag2id)
matrix,word2id,id2word,X_dummy,Y,files_len_list=dl_train.get_X_Y(int(doc_num))
word2id['[CLS]']=len(word2id)

final_X=utils.chunk_data_into_torch(matrix,files_len_list)
y_train=utils.chunk_data_into_torch(Y,files_len_list)
X_train=[]
for j in range(len(final_X)):
    X_train.append([id2word[final_X[j][i][4]] for i in range(final_X[j].shape[0])])


labels=list(set(Y))
not_removed_label=list(set(Y))
tag_to_ix=dict(zip(labels,range(len(set(Y)))))
print(tag_to_ix)
id_to_tag=dict(zip(range(len(set(Y))),labels))
o_idx=tag_to_ix['o']
labels.remove('o')


dl_test=Data_Load('test',pipeline,dep_type2id,pos_tag2id)
matrix_test,word2id_test,id2word_test,X_dummy_test,Y_test,files_len_list_test=dl_test.get_X_Y(int(int(doc_num)/3))
final_X_test=utils.chunk_data_into_torch(matrix_test,files_len_list_test)
y_test=utils.chunk_data_into_torch(Y_test,files_len_list_test)
X_test=[]
for j in range(len(final_X_test)):
    X_test.append([id2word_test[final_X_test[j][i][4]] for i in range(final_X_test[j].shape[0])])


dl_dev=Data_Load('dev',pipeline,dep_type2id,pos_tag2id)
matrix_dev,word2id_dev,id2word_dev,X_dummy_dev,Y_dev,files_len_list_dev=dl_dev.get_X_Y(int(int(doc_num)/3))
final_X_dev=utils.chunk_data_into_torch(matrix_dev,files_len_list_dev)
y_dev=utils.chunk_data_into_torch(Y_dev,files_len_list_dev)
X_dev=[]
for j in range(len(final_X_dev)):
    X_dev.append([id2word_dev[final_X_dev[j][i][4]] for i in range(final_X_dev[j].shape[0])])
#X_train, X_test, y_train, y_test = train_test_split(sent_list_x, final_Y, test_size=0.2)


best=0
model=classify_entity(word2id,768,len(not_removed_label),pretrained_path,weight_path)
model.train()
optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, model.parameters()), lr=0.5)
criterion = nn.CrossEntropyLoss()
try:
    for epoch in range(100):
        for idx,text in enumerate(X_train):
            numer_Y=torch.tensor([tag_to_ix[y] for y in y_train[idx]],dtype=torch.long)
            if numer_Y.shape[0]!=1:
                model.zero_grad()
                raw_output=model(text)

                # _,pred_Y=torch.max(raw_output,1)
                # print(pred_Y,numer_Y)
                
                #print(raw_output.shape,numer_Y.shape)

                loss=criterion(raw_output,numer_Y)
                loss.backward()
                optimizer.step()

        perf=evaluate(model,o_idx,X_dev,y_dev)
        print(perf)
        f1=perf['micro avg']['f1-score']
        if f1>best:
            best=f1
            torch.save(model.state_dict(), 'bestmodel_'+bert_model_name)


except KeyboardInterrupt:
    print ("----------------- INTERRUPTED -----------------")
    model.load_state_dict(torch.load('bestmodel_'+bert_model_name))
    perf=evaluate(model,o_idx,X_dev,y_dev,output_dict=False)
    print(perf)

model.load_state_dict(torch.load('bestmodel_'+bert_model_name))
perf=evaluate(model,o_idx,X_test,y_test,output_dict=False,plot_matrix=True)
print(perf)


#text="Are you ok?"
#tokenized_text=tokenizer.tokenize(text)
#tokens_tensor=torch.tensor([tokenizer.convert_tokens_to_ids(tokenized_text)]).to(device)
#regular bert
#paratemer tuning

#single ffnn
