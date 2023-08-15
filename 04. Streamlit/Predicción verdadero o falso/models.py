from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

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
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu" #para gpu
    
    noticia = lista[0]
    
    tokenizer = joblib.load('tokenizer1.pkl')
    model = joblib.load('model_albert_tiny_spanish.pkl')
    
    
    val_encoding = tokenizer(noticia, truncation=True, padding=True, return_tensors="pt").to(device)
    outputs = model(**val_encoding)
    logits = outputs.logits.cpu().detach().numpy()
    pred = (np.argmax(logits))
        
    return pred



