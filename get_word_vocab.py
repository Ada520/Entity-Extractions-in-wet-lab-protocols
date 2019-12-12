import numpy as np
class Vocab:
    def __init__(self,X):
        self.word2id={}
        self.id2word={}
        self.idCounter = 0
        self.matrix=None
        self.word_vocab(X)
        self.count_matrix(X)

    def word_vocab(self,X):
        for i in range(len(X)):
            for word in X[i]:
                if word not in self.word2id:
                    self.word2id[word] = self.idCounter
                    self.id2word[self.idCounter]=word
                    self.idCounter += 1
            
    def count_matrix(self,X):
        self.matrix=np.zeros((len(X), len(X[0])))
        for i in range(len(X)):
            for j in range(len(X[i])):
                self.matrix[i][j]=self.word2id[X[i][j]]
    
    def get_matrix(self):
        return self.matrix
    
    def get_word2id(self):
        return self.word2id
    
    def get_id2word(self):
        return self.id2word
