# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 19:48:32 2018

@author: User
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import svm 
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

data = pd.read_csv('turkish cyberbullying.csv')

# veri seti tanımak için yapılan aşamalar
print("Dataset veri(satır) ve öznitelik(sütun):")
print(data.shape)

print("Dataset hakkında genel bigiler: ")
print( data.info())

print("Dataset içeriği(ilk 5):\n")
print(data.head())

print("farklı değerlerin gruplanması:")
print(data.cyberbullying.value_counts())

data1 = data[data['cyberbullying']==1]
print("zorbalık içeren:"+ str(data1.shape))

data2 = data[data['cyberbullying']==0]
print("zorbalık içermeyen:"+ str(data2.shape))

# eşit sayıda zorbalık içeren ve içermeyen verilerin kullanılmasını sağlıyoruz
data = data2.append(data1[:1498])
print("Son veriseti :"+ str(data.shape))

# Türkçede çok kullanılan sözcüklerin bağlaçların vs. çıkartılması
def remove_stopwords(df_fon):
    stopwords = open('turkce-stop-words', 'r').read().split()
    df_fon['stopwords_removed'] = list(map(lambda doc:
        [word for word in doc if word not in stopwords], df_fon['message']))

remove_stopwords(data)

# Egitim ve test verilerinin oluşturulması 
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['cyberbullying'], test_size = 0.2, random_state = 0)

# vectorize 
vect = TfidfVectorizer(min_df = 5).fit(X_train)
X_train_vectorized = vect.transform(X_train)

# GridSearchCV, TfidfVectorizer, SVC ile en iyi sonuç verecek parametrelerin belirlenmesi 
parameters = {'kernel':['linear', 'rbf'], 'C':[0.1, 1, 10]}
svc = svm.SVC()
gd_sr  = GridSearchCV(svc, parameters, cv=5)
print(gd_sr.fit(X_train_vectorized, y_train))
print("\nEn iyi sonuç verecek parametreler: ", gd_sr.best_params_)

# elde ettiğimiz parametreler ile modelin eğitilmesi
model = SVC(C = 1, kernel = 'linear')
model.fit(X_train_vectorized, y_train)
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions)) 

# sonuçlar
print(model.predict(vect.transform(["VeriUs'ta staj yapmak çok güzel"])))

print(model.predict(vect.transform(["sen karaktersizsin"])))