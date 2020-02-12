from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
logisticRegr = LogisticRegression()

def main():
    data = pd.read_csv("train_foo.csv")
    dataset = np.array(data)
    np.random.shuffle(dataset)
    X = dataset
    Y = dataset
    X = X[:, 1:2501]
    Y = Y[:, 0]

    X_train = X[0:12000, :]
    X_train = X_train / 255.
    X_test = X[12000:13201, :]
    X_test = X_test / 255.

    # Reshape
    Y = Y.reshape(Y.shape[0], 1)
    Y_train = Y[0:12000, :]
    Y_train = Y_train.T
    Y_test = Y[12000:13201, :]
    Y_test = Y_test.T

    print("number of training examples = " + str(X_train.shape[0]))
    print("number of test examples = " + str(X_test.shape[0]))
    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))
    print("X_test shape: " + str(X_test.shape))
    print("Y_test shape: " + str(Y_test.shape))
    image_x = 50
    image_y = 50

    train_y = np_utils.to_categorical(Y_train) # convert a class vector to binary class matrix
    test_y = np_utils.to_categorical(Y_test)
    train_y = train_y.reshape(train_y.shape[1], train_y.shape[2])# gives a new shape to array without changing its data
    test_y = test_y.reshape(test_y.shape[1], test_y.shape[2])
    X_train = X_train.reshape(X_train.shape[0], 50, 50, 1)
    X_test = X_test.reshape(X_test.shape[0], 50, 50, 1)
    print("X_train shape: " + str(X_train.shape))
    print("X_test shape: " + str(X_test.shape))

    model, callbacks_list = keras_model(image_x, image_y)
    model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=12, batch_size=64,
              callbacks=callbacks_list)  # used gto train the data
    scores = model.evaluate(X_test, test_y, verbose=0)  # computes the loss based on input you pass along with anyother matrix
    print(" Error: %.2f%%" % (100 - scores[1] * 100))
    print_summary(model)

    model.save('emojinator.h5')


main()



