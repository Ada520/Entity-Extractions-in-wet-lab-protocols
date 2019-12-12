import os
from get_features import get_ngram_lemmas_syn,get_sent_pos_list,get_line_list,get_labels
import numpy as np
import get_word_vocab
import torch
import torch.nn as nn
from utils import first,get_embeddings

class Data_Load():
    def __init__(self,path,pipleline,dep_type2id,pos_tag2id,glove_path="gensim_glove_vectors.txt",emb_dim=300):
        self.original_path=path+"/"
        self.tag_path=path+"Tag/"
        self.nlp=pipleline
        self.glove_path=glove_path
        self.emb_dim=emb_dim
        self.o_list=['.',',','[',']','(',')','a']
        self.dep_type2id=dep_type2id
        self.pos_tag2id=pos_tag2id

    def get_X_Y(self,max_num=10000):
        files=os.listdir(self.tag_path)
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        X=[]
        X_catg=[]
        X_pos=[]
        X_pre=[]
        Y=[]
        #dep_type2id={}
        #pos_tag2id={}
        i=0
        files_len_list=[]
        #files=['protocol_615.txt']
        for f in files:
            if i==max_num:
                break
            i+=1
            print(f)
            no_affix_f=f[:f.index(".txt")]
            sent_pos_list=get_sent_pos_list(self.tag_path+f)
            line_list=get_line_list(self.original_path+f)
            ngram_lemmas_syn_list=get_ngram_lemmas_syn(sent_pos_list,self.nlp)

            label_list=get_labels(self.original_path+no_affix_f+".ann")
            doc_list=get_line_list(self.original_path+f)

            len_list=[0]*len(doc_list)

            for j in range(1,len(doc_list)):
                len_list[j]=len_list[j-1]+len(doc_list[j-1])
            len_list.extend([1000000])
            len_label_list=[[] for i in range(len(doc_list))]
            one_len=[]
            flag=1


            
            for entityidx,info,words in label_list:
                #print(info)
                info=info.split(";")[0]
                clas,start,end=info.split(" ")
                slot,num=first(len_list,lambda x:int(start)<x)
                len_label_list[slot-1].append((words,clas))
                """if int(start)<len_list[flag]:
                    one_len.append((words,clas))
                else:
                    flag+=1
                    len_label_list.append(one_len)
                    one_len=[(words,clas)]
                if len(len_label_list)<len(doc_list):
                    len_label_list.append([])    """

            
            for idx,sent_list in enumerate(ngram_lemmas_syn_list):
                #['<START>', '<PAD>', '<PAD>', '<PAD>'], ['SpinSmart', 'NNP', 'SpinSmart'], ['Plasmid', 'plasmid_DNA', '<PAD>', '<PAD>'], ['<PAD>', '<PAD>'], ['Plasmid', 'compound'])
                #(['SpinSmart', '<PAD>', '<PAD>', '<PAD>'], ['Plasmid', 'NNP', 'Plasmid'], ['Purification', 'refining', 'refinement', 'purgation'], ['compound', 'SpinSmart'], ['Purification', 'compound']
                k=0
                for word_feat in sent_list:
                    k+=1
                    # if word_feat[3][0] not in dep_type2id.keys():
                    #     dep_type2id[word_feat[3][0]]=len(dep_type2id)
                    # if word_feat[4][1] not in dep_type2id.keys():
                    #     dep_type2id[word_feat[4][1]]=len(dep_type2id)
                    # if word_feat[1][1] not in pos_tag2id.keys():
                    #     pos_tag2id[word_feat[1][1]]=len(pos_tag2id)

                    if word_feat[1][0] in self.o_list:
                        Y.append('o')
                    else:
                        Y.append(first(len_label_list[idx],lambda x:x[0].find(word_feat[1][0])>=0)[1][1])

                    x_feat=[]
                    x_feat.extend(word_feat[0])
                    x_feat.extend([word_feat[1][0],word_feat[1][2]])
                    x_feat.extend(word_feat[2])
                    x_feat.extend([word_feat[3][1]])
                    x_feat.extend([word_feat[4][0]])

                    x_feat.extend([word_feat[1][1]])
                    x_feat.extend([word_feat[3][0]])
                    x_feat.extend([word_feat[4][1]])
                    
                    X_pre.append(x_feat)
            
                files_len_list.append(k)


        for x_feat in X_pre:
            cat_g=[]
            deprel=[0]*len(self.dep_type2id)
            deprel[self.dep_type2id[x_feat[-2]] if x_feat[-2] in self.dep_type2id else 0]=1
            govrel=[0]*len(self.dep_type2id)
            govrel[self.dep_type2id[x_feat[-1]] if x_feat[-2] in self.dep_type2id else 0]=1
            cat_g.extend(deprel)
            cat_g.extend(govrel)

            pos_feat=[0]*len(self.pos_tag2id)
            pos_feat[self.pos_tag2id[x_feat[-3]] if x_feat[-3] in self.pos_tag2id else 0]=1

            x_feat=x_feat[:-3]

            X.append(x_feat)
            X_catg.append(cat_g)
            X_pos.append(pos_feat)

        X_catg=np.array(X_catg)

        X_pos=np.array(X_pos)

        X_dummy_features=np.concatenate((X_catg,X_pos),axis=1)

        vocab=get_word_vocab.Vocab(X)
        word2id=vocab.get_word2id()
        id2word=vocab.get_id2word()
        matrix=vocab.get_matrix()

        """print("load embedding...")
        embeddings_matrix=get_embeddings(self.glove_path,word2id,self.emb_dim)
        print("finally finish loading!!!")
        embedding=nn.Embedding(embeddings_matrix.shape[0],embeddings_matrix.shape[1])
        embedding.weight=nn.Parameter(embeddings_matrix)
        embedding.weight.requires_grad=False
        embedding_X=np.array(embedding(torch.LongTensor(matrix)))
        embedding_X=embedding_X.reshape(embedding_X.shape[0],embedding_X.shape[1]*embedding_X.shape[2])"""
        #print(embedding_X.shape)

        #final_X=np.concatenate((embedding_X,X_catg,X_pos),axis=1)
        #print(final_X.shape)

        Y=np.array(Y)

        #print(dep_type2id)
        #print(pos_tag2id)
        #print(Y.shape)
        return matrix,word2id,id2word,X_dummy_features,Y,files_len_list


"""
field = Field(sequential=False, use_vocab=True)
print("build vocab")
field.build_vocab(X,vectors="glove.6B.100d")
print("done built")
train, val, test = data.TabularDataset.splits(
        path='./data/', train='train.tsv',
        validation='val.tsv', test='test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size = BATCH_SIZE)

train_iter, val_iter = Iterator.splits(
        (X[:32],X[32:]), sort_key=lambda x: len(x.Text),
        batch_sizes=(16,16))

batch = next(iter(train_iter))
print("batch text: ", batch.Text) # 对应 Fileld 的 name
print("batch label: ", batch.Label)"""





"""numer_X_dev=[]
pos_X_dev=[]
rel_X_dev=[]
i=0
for f in files:
    if i==3:
        break
    i+=1
    pos_tag,unigram,bigram,lemmas_list,syn_list,gov,rel,dep=get_all_features('dev/','devTag/',f,nlp)
    
    text_features=[]
    text_features.extend(pos_tag)
    text_features.extend(unigram)
    text_features.extend(bigram)
    text_features.extend(lemmas_list)
    text_features.extend(syn_list)
    text_features.extend(gov)
    text_features.extend(rel)
    text_features.extend(dep)
    numer_X_dev.append(text_features)
    
    pos_X_dev.append(pos_tag)

    rel_X_dev.append(rel)



print(len(numer_X_dev))
vocab=get_word_vocab.Vocab(numer_X_dev)
matrix=vocab.get_matrix()
print(matrix.shape)
word2id=vocab.get_word2id()
embeddings_matrix=get_embeddings("gensim_glove_vectors50d.txt",word2id)
embedding=nn.Embedding(embeddings_matrix.shape[0],embeddings_matrix.shape[1])
embedding.weight=nn.Parameter(embeddings_matrix)
embedding.weight.requires_grad=False
embedding_X=np.array(embedding(torch.LongTensor(matrix)))
print(embedding_X.shape)

pos_tag_cat=get_catagorical_features(pos_X_dev)
print(pos_tag_cat.shape)
rel_cat=get_catagorical_features(rel_X_dev)
print(rel_cat.shape)"""