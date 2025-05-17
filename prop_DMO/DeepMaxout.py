import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow as tf

class Maxout(Layer):
    def __init__(self, units, **kwargs):
        super(Maxout, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                    initializer='glorot_uniform',
                                    trainable=True)
        super(Maxout, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        flattened = tf.reshape(inputs, [-1, input_shape[-1]])
        output = tf.matmul(flattened, self.kernel)
        output = tf.reshape(output, [input_shape[0], input_shape[1], input_shape[2], self.units])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], self.units)
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.models as M
import keras.layers as L
import keras.backend as K
from keras.constraints import max_norm
import random
from prop_DMO import LHFGSO
import math
from sklearn.metrics import mean_absolute_error


def maxout_activation_function(inputs,units,axis=None):
    if axis is None:
        axis=-1
    input_shape=inputs.get_shape().as_list()
    n_dims=len(input_shape)
    assert n_dims==4
    num_channels=input_shape[axis]
    if num_channels%units :
        raise ValueError('number of features({}) is not a multiple of num_units({})'.format(num_channels, units))
    input_shape[axis]=units
    input_shape+=[num_channels//units]
    output=K.reshape(inputs,(-1,input_shape[1],input_shape[2],input_shape[3],input_shape[4]))
    output_max=K.max(output,axis=-1,keepdims=False)
    return output_max


max_norm = max_norm(max_value=8, axis=[0, 1, 2])
img_shape=(64,64,3)
a,b,d,f,g=350,20,10,5,50
inp=L.Input(img_shape)
conv1=L.Conv2D(filters=64,kernel_size=(3,3),activation=None, kernel_constraint=max_norm)(inp)
maxout1 = Maxout(32)(conv1)
batch1=L.BatchNormalization(momentum=0.8)(maxout1)
pool1 = L.MaxPooling2D(pool_size=(2,2))(batch1)
drop1= L.Dropout(0.6)(pool1)
conv2=L.Conv2D(filters=128,kernel_size=(3,3),activation=None, kernel_constraint=max_norm)(drop1)
maxout2 = Maxout(64)(conv2)
batch2=L.BatchNormalization(momentum=0.8)(maxout2)
pool2 = L.MaxPooling2D(pool_size=(2,2))(batch2)
drop2= L.Dropout(0.5)(pool2)
conv3=L.Conv2D(filters=256,kernel_size=(3,3),activation=None, kernel_constraint=max_norm)(drop2)
maxout3 = Maxout(64)(conv3)
batch3=L.BatchNormalization(momentum=0.8)(maxout3)
pool3 = L.MaxPooling2D(pool_size=(2,2))(batch3)
drop3= L.Dropout(0.4)(pool3)
flatten=L.Flatten()(drop3)
dense=L.Dense(2,activation='softmax')(flatten)


model2=M.Model(inputs=inp,outputs=dense)



model2.compile(loss='categorical_crossentropy',
                     optimizer='adam',
                     metrics=['accuracy'])
#tf.keras.utils.plot_model(model2, to_file='MaxoutArchitecture.png', show_shapes=True, show_layer_names=True)
def Dmax(feature,label,tr,A,Se,Sp):
    hr = 0.5
    NF = 5
    # label=np.random.randint(10,size=(len(feature)))
    X_train, X_test, y_train, y_test = train_test_split(feature,label, train_size = tr, random_state = 42)
    target = np.concatenate((y_train, y_test))
    # X_train = tf.convert_to_tensor(X_train)
    xt = len(X_train)
    xt1 = len(X_test)
    X_train = np.resize(X_train, (xt, 64, 64, 3))
    X_test = np.resize(X_test, (xt1, 64, 64, 3))
    y_trainx = np.resize(y_train, (xt, 2))
    model2.fit(X_train, y_trainx, epochs=2, batch_size=10, verbose=0)
    Initial_weight=model2.get_weights()

    # model.set_weights(w + HFGSO.algm(w))
    #model2.set_weights(Initial_weight+LHFGSO.algm(Initial_weight))

    updated_weights = LHFGSO.algm(Initial_weight)
    model2.set_weights(updated_weights)

    predict = []
    x_test = np.array(X_test).copy()
    x_test = np.resize(x_test, (len(x_test), 32, 32, 3))
    x_test = x_test.astype('float32') / 255
    pred = model2.predict(X_test)
    for ii in range(len(pred)):
        if ii == 0:
            predict.append(np.max(pred[ii]))
        else:
            tem = []
            for j in range(len(pred[ii])):
                tem.append(np.abs(pred[ii][j] - pred[ii - 1][j]))
            predict.append(np.max(tem))
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
    # target = y_test
    # target = np.array(target).flatten()
    # y_test = np.resize(y_test, (xt1, 2))

    # predict = []
    # for i in range(len(y_trainx)): predict.append(y_trainx[i])
    #
    # for i in range(len(pred)):
    #     if i == 0:
    #         predict.append(np.argmax(pred[i]))
    #     else:
    #         tem = []
    #         for j in range(len(pred[i])):
    #             tem.append(np.abs(pred[i][j] - pred[i - 1][j]))
    #         predict.append(np.argmax(tem))
