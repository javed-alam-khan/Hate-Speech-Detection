# Importing libraries
import nltk
import pandas
import numpy
import string
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import csv
from sklearn import metrics

# Reading the train data set
data_set = pandas.read_csv("hate_speech_train.csv") 
text = data_set.iloc[:,0]
labels = data_set.iloc[:,1]

# Reading the test data set
test_ds = pandas.read_csv("hs_test.csv")
# print(test_ds.columns)
# test_text = test_ds.iloc[:,0]
test_text = test_ds.loc[:,'text']
# print(test_text.head())


# string.punctuation
def remove_punct(txt):
    no_punct_txt = []
    for lv in txt:
        if(lv not in string.punctuation):
            no_punct_txt.append(lv)
    return "".join(no_punct_txt)

data_set['no_punct_txt'] = data_set['text'].apply(lambda x : remove_punct(x))
# print(data_set.shape)
# print(data_set.head())
print(test_ds.head())
test_ds['no_punct_txt'] = test_ds['text'].apply(lambda x : remove_punct(x))


def tokenize(txt):
    tokens = re.split('\W+',txt)
    return tokens

data_set['tokenized_txt'] = data_set['no_punct_txt'].apply(lambda x: tokenize(x.lower()))
print(data_set.head())
test_ds['tokenized_txt'] = test_ds['no_punct_txt'].apply(lambda x: tokenize(x.lower()))


stop_words = nltk.corpus.stopwords.words('english')
# print(stop_words[:179])


def remove_stpwrds(txt):
    no_stpwrds = []
    for lv in txt:
        if lv not in stop_words:
            no_stpwrds.append(lv)
    return no_stpwrds

data_set['no_stop_words'] = data_set['tokenized_txt'].apply(lambda x: remove_stpwrds(x))
print(data_set.head())
test_ds['no_stop_words'] = test_ds['tokenized_txt'].apply(lambda x: remove_stpwrds(x))



ps = PorterStemmer()
def stemming(txt):
    stem_txt = []
    for lv in txt:
        stem_txt.append(ps.stem(lv))
    return stem_txt

data_set['stem_txt'] = data_set['no_stop_words'].apply(lambda x: stemming(x))
print(data_set.head())
test_ds['stem_txt'] = test_ds['no_stop_words'].apply(lambda x: stemming(x))


wn = nltk.WordNetLemmatizer()
def lemmatization(txt):
    lemmatized_txt = []
    for lv in txt:
         lemmatized_txt.append(wn.lemmatize(lv))
    return lemmatized_txt

data_set['lemmatized_txt'] = data_set['no_stop_words'].apply(lambda x: lemmatization(x))
print(data_set.head())
test_ds['lemmatized_txt'] = test_ds['no_stop_words'].apply(lambda x: lemmatization(x))

# tmp1 = data_set.iloc[:,0]
tmp2 = data_set.iloc[:,5]
tmp3 = test_ds.iloc[:,9]
print(tmp3.head())
value1=[' '.join([word for word in row]) for row in tmp2]
value2=[' '.join([word for word in row]) for row in tmp3]
vectorizer = TfidfVectorizer().fit(value1)
vectorized_ds = vectorizer.transform(value1)
# print(type(vectorized_ds))
# print(vectorized_ds)
vectorized_ts = vectorizer.transform(value2)
print(vectorized_ds.shape)
print(vectorized_ts.shape)

# Splitting
train_text, validate_text, train_labels, validate_labels = train_test_split(vectorized_ds, labels, test_size=0.3, random_state=42)



# Svm classifier
svclassifier = SVC(kernel = 'linear' , C = 1.0)

svclassifier.fit(train_text, train_labels)

pred_labels = svclassifier.predict(validate_text)
# f1_score(validate_labels, pred_labels, average='macro')
f1_score(validate_labels, pred_labels, average='micro')
# f1_score(validate_labels, pred_labels, average='weighted')
# f1_score(validate_labels, pred_labels, average=None)

accuracy = metrics.accuracy_score(validate_labels, pred_labels)
print("accuracy",accuracy)



# Logistic regression 
classifier = LogisticRegression(random_state = 0) 


classifier.fit(train_text, train_labels) 
pred_labels = classifier.predict(validate_text)
print(type(pred_labels))
print("f1 score:", f1_score(validate_labels, pred_labels))
print ("accuracy : ", accuracy_score(validate_labels, pred_labels))


# Logistic regression on test data
classifier = LogisticRegression(random_state = 0) 
classifier.fit(vectorized_ds, labels) 
pred_labels = classifier.predict(vectorized_ts) 



# numpy.savetxt("submission.csv",pred_labels,header='labels',fmt='%d',comments='')
output = numpy.reshape(pred_labels,(pred_labels.shape[0],1))
index = numpy.array([ i for i in range(len(output))])
index = numpy.reshape(index,(index.shape[0],1))
output = numpy.append(index, output, axis = 1)
print(output.shape)

filename = 'hs_submission.csv'
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile) 
    fields = ['','labels']
    # writing the fields  
    csvwriter.writerow(fields)  
    # writing the data rows  
    csvwriter.writerows(output)

