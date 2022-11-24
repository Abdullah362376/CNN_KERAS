from tensorflow.keras import datasets, layers, models
import numpy as np
validation_split = 0.2
verbosity = 1
(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data() # to load and split the data into train and test
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)      # to check the shapes of yout data

y_train = y_train.reshape(-1,) # reshape the lables into one colmn
y_test = y_test.reshape(-1,)

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
# this was a list varible that indicate class 0 is airplane and class 1 is automobile and so on  till 9 is truck

# zero center the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# normilize the data
X_train /= np.std(X_train, axis=0)
X_test /= np.std(X_test, axis=0)

cnn = models.Sequential([
    #first layer has 64 filters and kernal size 3*3
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    # max pool the result with a 2*2 grid
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # flatten the result to apply ANN
    layers.Flatten(),
    # use relu for the inner of the network
    layers.Dense(128, activation='relu'),
    #soft mak fot the outer of the network
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10, verbose=verbosity, validation_split=validation_split)
score = cnn.evaluate(X_test,y_test,verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# y_pred = cnn.predict(X_test)
# y_classes = [np.argmax(element) for element in y_pred]
