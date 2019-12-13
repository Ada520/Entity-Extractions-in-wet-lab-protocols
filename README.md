# Entity-Extractions-in-wet-lab-protocols
Entity Extractions in wet lab protocols, MaxEnt/BiLSTM-CRF/SciBert/Actor-Critic

The wet lab protocols corpus can be retrieved from https://github.com/chaitanya2334/WLP-Dataset
You can get GENIA POS tagger from http://www.nactem.ac.uk/GENIA/tagger/ and generate 3 subfolders containing POS tagger result files of each protocol files of train/dev/test.
And Glove Embeddings file should be in the same directory with code files, it can be found here: https://nlp.stanford.edu/projects/glove/
Stanford DP can be found here:https://stanfordnlp.github.io/stanfordnlp/pipeline.html. It is actually stanford nlp pipeline and DP is one function of it. Put stanford-english-corenlp-2018-10-05-models.jar in the same directory.
SciBert can be found in allenai:https://github.com/allenai/scibert, which has the same configuration with base bert.

To run my code, do "python run_scibert.py x y". x is the total number of training files you want to load, if you want to load all, just enter an arbitrary big numebr like 1000. y is the dimension of vector used, by default it is 300.










































写reademe也太难了吧，教别人run自己的code这种事情感觉很糟糕，就好像别人马上就知道你写代码水平这么差了。。
