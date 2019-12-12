import torch
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt

def first(the_iterable, condition = lambda x: True):
    for idx,i in enumerate(the_iterable):
        #print(i)
        if condition(i):
            return idx,i
    return -1,"None"


def get_embeddings(path,word2id,dim=300):
    glove_model = KeyedVectors.load_word2vec_format(path, binary=False)
    embeddings=np.zeros((len(word2id),dim))
    for word in word2id.keys():
        if word in glove_model:
            embeddings[word2id[word]]=np.array(glove_model[word])
        else:
            embeddings[word2id[word]]=np.random.randn(dim)
    return torch.from_numpy(embeddings).float()

def chunk_data_into_torch(X,len_list):
    data_list=[]
    start=0
    for i in range(len(len_list)):
        data_list.append(X[start:start+len_list[i]])
        start+=len_list[i]
    return data_list

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    #print(classes)
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax