from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

stopwords = stopwords.words('spanish')


def try_model(M, data, cm = False):
    
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]
    
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