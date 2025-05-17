import numpy as np
import math
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
model = Sequential()
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,LeakyReLU
# from tensorflow.keras.layers.advanced_activations import LeakyReLU
from sklearn.model_selection import train_test_split
from random import shuffle as array

def main_CNN(X_train,X_test,Y_train,Y_test,tpr):
    train_X = X_train.reshape(-1, 1,X_train.shape[1], 1)
    test_X = X_test.reshape(-1, 1,X_train.shape[1], 1)
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    train_X = train_X / train_X.max()
    test_X = test_X / train_X.max()
    train_X, valid_X, train_label, valid_label = train_test_split(train_X, Y_train, test_size=0.2,random_state=13)
    batch_size = 5
    epochs = 10
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', input_shape=( 1, X_train.shape[1],1), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(32, activation = 'relu', input_dim = len(X_train[0])))
    model.add(Dense(units = 1))

    adam = Adam()
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    model.fit(train_X, train_label, batch_size=batch_size, epochs=epochs, verbose=0)#,validation_data=(valid_X, valid_label))
    test_eval_arr = model.predict(test_X)
    pred = test_eval_arr.flatten()
    return pred


def process(data):
    return data.sort(reverse=True)
def bound(f_data):

    fe = []
    sq = int(math.sqrt(len(f_data[0])))
    n = int(sq * sq)
    for i in range(len(f_data)):
        tem = []
        for j in range(n):  # attributes in each row
            tem.append(f_data[i][j])  # add value to tem array
        fe.append(tem)  # add 1 row of array value to fe
    return fe

def callmain(x1,y1,tpr,A,Se,Sp):
    X_train, X_test, y_train, y_test = train_test_split(x1, y1, train_size=tpr-0.2)
    target = np.concatenate((y_train,y_test))
    pred = main_CNN(np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test),tpr)
    tp, tn, fn, fp = 0, 0, 0, 0
    uni = np.unique(target)
    predict = np.concatenate((y_train,pred))
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(target)):
            if (target[i] == c and predict[i] == c):
                tp += 1
            if (target[i] != c and predict[i] != c):
                tn += 1
            if (target[i] == c and predict[i] != c):
                fn += 1
            if (target[i] != c and predict[i] == c):
                fp += 1
    fn = fn / uni[len(uni) - 1]
    fp = fp / len(uni)
    A.append((tp + tn) / (tp + tn + fp + fn))
    Se.append(tp / (tp + fn))
    Sp.append(tn / (tn + fp))