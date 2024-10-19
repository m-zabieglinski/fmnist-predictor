import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import numpy as np

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

mean_value = np.mean(x_train)
std_value = np.std(x_train)
x_train = (x_train - mean_value) / std_value
x_test = (x_test - mean_value) / std_value

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2, random_state = 1234)

#######################################################################################
model = models.Sequential()
model.add(layers.experimental.preprocessing.RandomCrop(28, 28, input_shape=(28, 28, 1)))
model.add(layers.experimental.preprocessing.RandomFlip("horizontal"))
#model.add(layers.experimental.preprocessing.RandomRotation(0.02))


regularizer1 = regularizers.L1(0.01)
regularizer2 = regularizers.L2(0.01)
regularizer3 = regularizers.L1L2(l1 = 0.01, l2 = 0.01)

model.add(layers.Conv2D(256, (3, 3),
                        activation='relu',
                        input_shape=(28, 28, 1),
                        padding = "same",
                        kernel_regularizer = regularizer3,
                        )
         )

# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3),
                        activation='relu',
                        padding = "same",
                        kernel_regularizer = regularizer3,
                        )
         )

# model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3),
                        activation='relu',
                        padding = "same",
                        kernel_regularizer = regularizer3,
                        )
         )

# model.add(layers.BatchNormalization())

# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(256, (3, 3),
                        activation='relu',
                        padding = "same",
                        kernel_regularizer = regularizer3,
                        )
         )

# model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(512, (3, 3),
                        activation='relu',
                        padding = "same",
                        kernel_regularizer = regularizer3,
                        )
         )

# model.add(layers.BatchNormalization())

# model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.60))
model.add(layers.Dense(10, activation='softmax'))

###################################################################

optimizer1 = optimizers.Adam(learning_rate = 0.001,
                            )

model.compile(optimizer = optimizer1,
             loss = "categorical_crossentropy",
             metrics = ["accuracy"],
             )

#model.summary()

early_stopping = EarlyStopping(monitor = "val_accuracy", patience = 10, restore_best_weights = True)


model.fit(x_train, y_train,
                #   batch_size = 32,
                  epochs = 30,
                  validation_data=(x_val, y_val),
                  callbacks=[early_stopping],
         )


model.fit(x_train, y_train,
                  batch_size = 256,
                  epochs = 100,
                  validation_data=(x_val, y_val),
                  callbacks=[early_stopping],
         )

test_loss, test_acc = model.evaluate(x_test, y_test)
show_acc = test_acc * 100
print(f'ACCURACY: {show_acc:.2f}%')
if test_acc > 0.925:
    print(f"saving model to model_{round(show_acc * 100)}.h5")
    model.save(f"model_{round(show_acc * 100)}.h5")