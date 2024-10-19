from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from keras.models import load_model
import numpy as np
model = load_model("model_9347.h5")

model.summary()

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

y_test = to_categorical(y_test, 10)

mean_value = np.mean(x_train)
std_value = np.std(x_train)

x_test = (x_test - mean_value) / std_value

test_loss, test_acc = model.evaluate(x_test, y_test)
show_acc = test_acc * 100
print(f'ACCURACY: {show_acc:.2f}%')

model.save_weights("weights_9347")