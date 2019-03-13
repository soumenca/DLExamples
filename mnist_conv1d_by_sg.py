from __future__ import absolute_import, division, print_function

import tensorflow as tf
import keras

import numpy as np 
import matplotlib.pyplot as plt 

print("TensorFlow Version: {}".format(tf.__version__))


def printImg(img, prediction):
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight','Nine']
    plt.figure()
    plt.imshow(img)
    plt.colorbar()
    plt.grid(False)
    plt.xlabel(class_names[prediction])
    plt.title('Test image and the crosponding predicted category in x-axis label')
    plt.show()


def printImgSet(train_images, train_labels, class_names):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap = plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.suptitle('25 samples from training images')
    plt.show()


def NNModel(train_x, train_y, test_x, test_y, number_epochs):
    model = keras.Sequential([
        keras.layers.Conv1D(64, 3, activation=tf.nn.relu, input_shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dropout(0.25),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=number_epochs, verbose=1)

    score = model.evaluate(test_x, test_y, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    predictions = model.predict(test_x)
    n = np.array([test_y.shape])
    index = np.random.randint(n, size=1)
    predictions = model.predict(test_x[index])
    
    #print("Random input {}".format(index))
    print("Predicted label {}".format(np.argmax(predictions)))
    print("Test label {}".format(test_y[index]))
    printImg(test_x[index].reshape(28, 28), np.argmax(predictions))


if __name__ == "__main__":

    number_epochs = 12
    mnist = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight','Nine']

    print("Dimension of the training data {}".format(train_x.shape))
    print("Dimension of the test data {}".format(test_x.shape))

    train_x = train_x / 255.0
    test_x = test_x / 255.0

    NNModel(train_x, train_y, test_x, test_y, number_epochs)

    printImgSet(train_x, train_y, class_names)