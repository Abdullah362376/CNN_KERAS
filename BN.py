import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import datasets, layers, models
import numpy as np

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

model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))

# Model configuration
batch_size = 50
no_epochs = 10
no_classes = 10
validation_split = 0.2
verbosity = 1


# Compile the model
model.compile(loss=tensorflow.keras.losses.sparse_categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
history = model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
