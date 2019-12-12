from get_data import Data_Load
import numpy as np
import stanfordnlp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.externals import joblib

pipeline=stanfordnlp.Pipeline()

dl=Data_Load('train',pipeline)
X,Y=dl.get_X_Y(500)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Training...patience...")
clf = LogisticRegression(solver='lbfgs',max_iter=1000, multi_class='multinomial', n_jobs=4).fit(X_train, Y_train)
print("Done, let's see!!")
pre_y=clf.predict(X_test)
pre_y=pre_y[np.where(Y_test!='o')]

Y_test_without_o=Y_test[np.where(Y_test!='o')]

joblib.dump(clf, 'my_model.pkl')

#print("MACRO: ", precision_recall_fscore_support(Y_test, pre_y, average='macro'))
#print("MICRO: ", precision_recall_fscore_support(Y_test, pre_y, average='micro'))
print("Confusion Metrics \n", classification_report(Y_test,pre_y,labels=list(set(Y_test_without_o))))


"""pre_y=clf.predict(X_test)
pre_y=pre_y[np.where(Y_test!='o')]
Y_test_without_o=Y_test[np.where(Y_test!='o')]
#print("MACRO: ", precision_recall_fscore_support(Y_test, pre_y, average='macro'))
#print("MICRO: ", precision_recall_fscore_support(Y_test, pre_y, average='micro'))
print("Confusion Metrics \n", classification_report(Y_test_without_o,pre_y,labels=list(set(Y_test_without_o))))
clf2 = joblib.load('drive/Colab Notebooks/WLP/my_model.pkl')
"""