
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

# def get_model(width=128, height=128, depth=64):
#     """Build a 3D convolutional neural network model."""
#     ## Encoding layer
#
#     inputs = keras.Input((width, height, depth, 1))
#
#     # img_input = Input(shape= (192, 256, 3))
#     # x = Conv2D(64, (3, 3), padding='same', name='conv1',strides= (1,1))(inputs)
#     # x = BatchNormalization(name='bn1')(x)
#     # x = Activation('relu')(x)
#     # x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
#     # x = BatchNormalization(name='bn2')(x)
#     # x = Activation('relu')(x)
#     # x = MaxPooling2D()(x)
#
#     x = layers.Conv3D(64, 3, padding='same', name='conv1',strides= (1,1,1))(inputs)
#     # x = layers.MaxPool3D()(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#
#     x = layers.Conv3D(64, 3, padding='same', name='conv2')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.MaxPool3D()(x)
#
#     x = layers.Conv3D(128, 3, padding='same', name='conv3')(x)
#     # x = layers.MaxPool3D()(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#
#     x = layers.Conv3D(256, 3, padding='same', name='conv4')(x)
#     # x = layers.MaxPool3D()(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     # x = layers.MaxPool3D()(x)
#     # x = layers.GlobalAveragePooling3D()(x)
#
#     x = layers.Dense(units=512, activation="relu")(x)
#     x = layers.Dense(units=512, activation="relu")(x)
#
#     ## Decoding layers
#
#     x = layers.UpSampling3D()(x)
#     x = layers.Conv3DTranspose(256, 3, padding='same', name='deconv1')(x)
#     # x = layers.MaxPool3D()(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#
#     # x = layers.UpSampling3D()(x)
#     x = layers.Conv3DTranspose(128, 3, padding='same', name='deconv2')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     # x = layers.MaxPool3D()(x)
#
#     # x = layers.UpSampling3D()(x)
#     x = layers.Conv3DTranspose(64, 3, padding='same', name='deconv3')(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#
#     # x = layers.UpSampling3D()(x)
#     x = layers.Conv3DTranspose(1, 3, padding='same', name='deconv4',strides= (1,1,1))(inputs)
#     # x = layers.MaxPool3D()(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('sigmoid')(x)
#     # x = layers.MaxPool3D()(x)
#     pred = layers.Reshape((128,128,64))(x)
#     model = Model(inputs=inputs, outputs=pred)
#     model.compile(optimizer= SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False))
#     # x = Dense(1024, activation = 'relu', name='fc1')(x)
#     # x = Dense(1024, activation = 'relu', name='fc2')(x)
#     # x = layers.Dropout(0.3)(x)
#     #
#     # outputs = layers.Dense(units=1, activation="sigmoid")(x)
#     #
#     # # Define the model.
#     # model = keras.Model(inputs, outputs, name="3dcnn")
#     return model
#
# modle = get_model(width=128, height=128, depth=64)


def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    # x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dense(units=512, activation="relu")(x)

    # x = layers.Dropout(0.3)(x)

    ## decoding

    x= layers.UpSampling3D()(x)
    x= layers.Conv3DTranspose(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x= layers.UpSampling3D()(x)
    x= layers.Conv3DTranspose(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.UpSampling3D()(x)
    x = layers.Conv3DTranspose(filters=64, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=2)(x)

    x = layers.BatchNormalization()(x)

    x = layers.Conv3DTranspose(filters=64, kernel_size=3, activation="relu")(x)
    # x = layers.MaxPool3D(pool_size=1)(x)
    x = layers.UpSampling3D()(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False),loss='mean_squared_error')
    model.fit()
    return model


# Build model.
model = get_model(width=128, height=128, depth=64)
mode
model.summary()