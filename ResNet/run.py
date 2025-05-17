import numpy as np
from random import shuffle as array
from sklearn.model_selection import train_test_split
from HFGSO_DRN import DRN,HFGSO

def classify(xx,yy,tr,A,Se,Sp):
    x_train, x_test, y_train, y_test = train_test_split(xx, yy, train_size=tr-0.1)
    target = np.concatenate((y_train,y_test))
    model = DRN.classify(np.array(x_train), np.array(y_train))
    w = model.get_weights()
    # model.set_weights(w+HFGSO.algm(w))
    updated_weights = HFGSO.algm(w)
    model.set_weights(updated_weights)
    # Prediction.
    predict = []
    x_test = np.array(x_test).copy()
    x_test = np.resize(x_test, (len(x_test), 32, 32, 3))
    x_test = x_test.astype('float32') / 255
    pred = model.predict(x_test)
    for ii in range(len(pred)):
        if ii == 0:
            predict.append(np.max(pred[ii]))
        else:
            tem = []
            for j in range(len(pred[ii])):
                tem.append(np.abs(pred[ii][j] - pred[ii - 1][j]))
            predict.append(np.max(tem))
    array(predict)
    prediction = np.concatenate((y_train,predict))
    uni = np.unique(target)
    unique_clas = np.unique(y_test)
    prediction = prediction.astype(int)
    tp, tn, fn, fp = 0, 0, 0, 0
    for i1 in range(len(uni)):
        c = uni[i1]
        for i in range(len(target)):
            if (target[i] == c and prediction[i] == c):
                tp = tp + 1
            if (target[i] != c and prediction[i] != c):
                tn = tn + 1
            if (target[i] == c and prediction[i] != c):
                fn = fn + 1
            if (target[i] != c and prediction[i] == c):
                fp = fp + 1
    tn = tn / len(unique_clas)
    fn = fn / unique_clas[len(unique_clas) - 1]
    fp = fp / unique_clas[len(unique_clas) - 1]
    A.append((tp + tn) / (tp + tn + fp + fn))
    Se.append(tp / (tp + fn))
    Sp.append(tn / (tn + fp))