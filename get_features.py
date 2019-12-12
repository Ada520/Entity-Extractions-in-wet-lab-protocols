import os
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from random import shuffle
import stanfordnlp
import numpy as np
import re

def get_line_list(path):
    f=open(path)
    line_list=[]
    for line in f:
        if len(line)>1:
            #if line[-1]=='\n':
            #    line=line[:-1]
            line_list.append(line)
    return line_list

def get_labels(path):
    labels=[]
    f=open(path)
    for line in f:
        if line[0]=='T':
            labels.append(line.split("\t"))
            
    return labels

def one_word_syn(word,num_of_syn=3):
    synonyms=[]
    for syn in wn.synsets(word): 
        for l in syn.lemmas():
            if not((l.name()==word) or (l.name() in synonyms)):
                synonyms.append(l.name())
                if len(synonyms)==num_of_syn:
                    return synonyms
    while len(synonyms)<num_of_syn:
        synonyms.append('<PAD>')
    return synonyms

def get_dep_and_gov(sent,pipeline):
    nlp=pipeline
    doc = nlp(sent)
    dplist=[]
    for i in range(len(doc.sentences)):
        for j in range(len(doc.sentences[i]._dependencies)):
            if not(doc.sentences[i]._dependencies[j][0].text is 'ROOT'):
                dplist.append((doc.sentences[i]._dependencies[j][0].text,doc.sentences[i]._dependencies[j][1],doc.sentences[i]._dependencies[j][2].text))
    return dplist

def look_dep(word,dplist):
    shuffle(dplist)
    for gov,rel,dep in dplist:
        if gov==word:
            return [rel,dep]
    return ['<PAD>','<PAD>']

def look_gov(word,dplist):
    shuffle(dplist)
    for gov,rel,dep in dplist:
        if dep==word:
            return [gov,rel]
    return ['<PAD>','<PAD>']


def get_ngram_lemmas_syn(sent_pos_list,pipeline,num_of_syn=3):
    wordnet_lemmatizer = WordNetLemmatizer()
    sent_pos_ngram_list=[]
    for sent_pos in sent_pos_list:
        sent=" ".join([x[0] for x in sent_pos])
        dplist=get_dep_and_gov(sent,pipeline)

        sent_pos_ngram=[0]*len(sent_pos)
        for i,word_pos_pair in enumerate(sent_pos):
            
            dep=look_dep(word_pos_pair[0],dplist)
            gov=look_gov(word_pos_pair[0],dplist)
            word_pos_pair=[word_pos_pair[0],word_pos_pair[1],wordnet_lemmatizer.lemmatize(word_pos_pair[0])]

            if i==0:
                prev=["<START>"]
                prev.extend(["<PAD>"]*num_of_syn)
            else:
                prev=[sent_pos[i-1][0]]
                prev.extend(one_word_syn(sent_pos[i-1][0].lower(),num_of_syn))

            if i==len(sent_pos)-1:
                after=["<END>"]
                after.extend(["<PAD>"]*num_of_syn)
            else:
                after=[sent_pos[i+1][0]]
                after.extend(one_word_syn(sent_pos[i+1][0].lower(),num_of_syn))
                
            
            sent_pos_ngram[i]=(prev,word_pos_pair,after,dep,gov)
        sent_pos_ngram_list.append(sent_pos_ngram)

    return sent_pos_ngram_list

def get_sent_pos_list(path):
    f=open(path)
    sent_pos_list=[]
    one_sen=[]
    for line in f:
        if len(line)>1:
            pentatuple=line.split('\t')
            one_sen.append([pentatuple[0],pentatuple[2]])
        else:
            sent_pos_list.append(one_sen)
            one_sen=[]
    return sent_pos_list


#sent_pos_list=get_sent_pos_list('devTag/protocol_0.txt')
#ngram_lemmas_syn_list=get_ngram_lemmas_syn(sent_pos_list)
#print(ngram_lemmas_syn_list)



"""

def get_pos(line_list):
    pos_tag_array=[]
    for line in line_list:
        if len(line)>1:
            pos_tag_array.append(line.split('\t')[2])
    return pos_tag_array


def get_ngram(line_list,n):
    doc_ngram=[]
    for i in range(len(line_list)-n+1):
        ngram_list=[]
        for j in range(i,i+n):
            ngram_list.append(line_list[j].split('\t')[0])
        ngram=' '.join(ngram_list)
        doc_ngram.append(ngram)
    return doc_ngram
            


def get_lemmas_and_syn(word_list):
    wordnet_lemmatizer = WordNetLemmatizer()

    lemmas_list=[]
    for word in word_list:
        lemmas_list.append(wordnet_lemmatizer.lemmatize(word))
    
    syn_list=[]
    for word in word_list:
        syn_list.extend(one_word_syn(word.lower()))

    #print(syn_list)
    return lemmas_list,syn_list




def get_all_features(folder0,folder1,file,pipeline):
    line_list=get_line_list(folder1+file)
    pos_tag=get_pos(line_list)
    unigram=get_ngram(line_list,1)
    bigram=get_ngram(line_list,2)
    lemmas_list,syn_list=get_lemmas_and_syn(unigram)
    doc_list=get_line_list(folder0+file)
    gov,rel,dep=get_dep_and_gov(doc_list,pipeline)
    return pos_tag,unigram,bigram,lemmas_list,syn_list,gov,rel,dep



def get_catagorical_features(catg_list):
    all_in_one=[]
    for l in catg_list:
        all_in_one.extend(l)
    print(len(all_in_one))
    values = np.array(all_in_one)
    print(values.shape)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded.shape)
    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    print(integer_encoded.shape)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    cat_feat=np.array(np.split(onehot_encoded,n))
    return cat_feat"""