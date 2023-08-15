from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import pandas as pd

stopwords = stopwords.words('spanish')




def try_model(M, data, text = False, cm = False):
    
    X_train = data[0]
    X_test = data[1]
    y_train = data[2]
    y_test = data[3]
    
    if text == True:
        vectorizer = TfidfVectorizer(max_features = 1000, stop_words = stopwords + ['si', 'según', 'tras'])
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)
    
    M.fit(X_train, y_train)
    y_pred_train = M.predict(X_train); y_pred = M.predict(X_test)
    
    acc_train = metrics.accuracy_score(y_train, y_pred_train)
    acc = metrics.accuracy_score(y_test, y_pred)

    if cm == True:
        cm_test = confusion_matrix(y_test, y_pred, labels = M.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix = cm_test, display_labels = M.classes_)  

        disp.plot()
        plt.show()
    
    print(f'accuracy en train: {acc_train}, accuracy en test: {acc}')
    
    
    
    
def classifier(lista):
    
    noticia = lista[0]
    
    vectorizer = joblib.load('tfidf_vectorizer1.pkl')
    logit = joblib.load('regresion_logistica1.pkl')
    
    string_vect = vectorizer.transform(lista)
    pred = logit.predict(string_vect)[0]
    
    y = ''
    X_test_final = pd.read_csv('X_test_final.csv')
    y_test_final = pd.read_csv('y_test_final.csv')
    
    if (noticia[0] == "'" or noticia[-1] == "'") or (noticia[0] == '"' or noticia[-1] == '"'):
        noticia = noticia[1:-1]
    
    if noticia in X_test_final.values:
        i = list(X_test_final.values).index(noticia)
        y = y_test_final.iloc[i].Periódico
        
    return y, pred



