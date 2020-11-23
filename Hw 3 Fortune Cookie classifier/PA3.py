import pandas as pd 
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

stop = stopwords


x = pd.read_csv('traindata.txt', header = None)
y = pd.read_csv('trainlabels.txt',header=None)
z = pd.read_csv('stoplist.txt',header = None)
predData =  pd.read_csv('testdata.txt', header=None)
predLabels = pd.read_csv('testlabels.txt', header=None)

rangeX = x.size

rangePred = predData.size

x.columns = ['Data']
y.columns = ['Label']
z.columns = ['Stop']
predData.columns = ['Data']
predLabels.columns = ['Label']

dX = pd.concat([x,predData])

stopwords = z["Stop"].tolist()

dX['Filt_data'] = dX['Data'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))

del dX['Data']

dX['Tokenized_Data'] = dX.apply(lambda row: nltk.word_tokenize(row['Filt_Data']), axis = 1)
y = y['Label'].tolist()

predLabels = predLabels['Label'].tolist()

v = TfidfVectorizer()

Tfidf = v.fit_transform(dX['Filt_data'])

df1 = pd.DataFrame(Tfidf.toarray(), columns = v.get_feature_names())
print(df1)

x  = df1[0:rangeX]
predData = df1[rangeX:rangeX+rangePred]

ppn = Perceptron(max_iter=20, eta0=1, random_state=0, verbose =1)

ppn.fit(x,y)

y_pred = ppn.predict(predData)

print('Accuracy %.2f'% accuracy_score(predLabels, y_pred))

