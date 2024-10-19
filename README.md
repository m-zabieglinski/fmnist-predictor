<h2>About</h2>
The repository contains files that generate a convolutional neural network (CNN) that scores up to 93.4% accuracy on the well-known fmnist dataset.
The model uses the tensorflow package.
It was run on my personal computer on GPU. It could probably be stressed further (larger convolutional layers) for higher accuracy if more memory was available.


<h2>Details</h2>
The code provided does not provide the 93.4% accuracy model. Different models were tried out in iterations. I have the models saved but even the weight files are too big for github.
Regardless, the structure/idea of all these models is the same:

<ol>
  <li>standarization and normalization of the input data</li>
  <li>random crop and random horizontal flip</li>
  <li>large convolutional layers with 1 max pooling layer to improve performance</li>
  <li>flatten into a classic deep neural network with a dropout</li>
</ol>
Training in micro batches.


<h2>Environment</h2>
There's a lot of clutter with .ipynb files and non pip freeze provided - acknowledged. This is an old project and I don't remember all the details :).
Setting up the environment was problematic - CUDA is no longer supported for AMD by neither party and the old versions conflict directly with available versions of iPython so there was a lot of patching.
My venv is hence a mess. I advise you run the code yourself and download packages as needed.



<h2>93.47% model structure</h2>
The tensorflow-provided summary for the 93.47% model is below. This model also contained standarization and normalization but that layer was outside the model proper and hence excluded from the default report.

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 random_crop (RandomCrop)    (None, 28, 28, 1)         0

 random_flip (RandomFlip)    (None, 28, 28, 1)         0

 conv2d (Conv2D)             (None, 28, 28, 256)       2560

 conv2d_1 (Conv2D)           (None, 28, 28, 512)       1180160

 max_pooling2d (MaxPooling2  (None, 14, 14, 512)       0
 D)

 conv2d_2 (Conv2D)           (None, 14, 14, 512)       2359808

 conv2d_3 (Conv2D)           (None, 14, 14, 512)       2359808

 flatten (Flatten)           (None, 100352)            0

 dense (Dense)               (None, 512)               51380736

 dropout (Dropout)           (None, 512)               0

 dense_1 (Dense)             (None, 10)                5130

=================================================================<br>
Total params: 57288202 (218.54 MB)<br>
Trainable params: 57288202 (218.54 MB)<br>
Non-trainable params: 0 (0.00 Byte)<br>
_________________________________________________________________
313/313 [==============================] - 65s 208ms/step - loss: 0.2368 - accuracy: 0.9347<br>
ACCURACY: 93.47%
