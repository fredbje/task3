import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.01
training_epochs = 100
n_hidden_1 = 1024
n_hidden_2 = 512
n_hidden_3 = 256
n_hidden_4 = 128
RANDOM_SEED = 42

# Load data from hdf file
train_data = pd.read_hdf("train.h5", "train").as_matrix()
(N, M) = train_data.shape
target = train_data[:, 0].astype(int)
num_labels = len(np.unique(target))
all_y = np.eye(num_labels)[target]  # One liner trick!
# Prepend a column of 1s for bias
all_X = train_data[:, 1:]
#all_X = np.ones((N, M))
#all_X[:, 1:] = train_data[:, 1:]
test_data = pd.read_hdf("test.h5", "test").as_matrix()
(N_pred, M_pred) = test_data.shape
#pred_X = np.ones((N_pred, M_pred + 1))
#pred_X[:, 1:] = test_data
pred_X = test_data

tf.set_random_seed(RANDOM_SEED)
train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.33, random_state=RANDOM_SEED)

# Layer's sizes
n_features = train_X.shape[1]  # Number of input nodes: 100 features and 1 bias
n_classes = train_y.shape[1]   # Number of outcomes

# Symbols
X = tf.placeholder("float", shape=[None, n_features])
y = tf.placeholder("float", shape=[None, n_classes])

# Weight initializations
w_1 = tf.Variable(tf.random_normal((n_features, n_hidden_1), stddev=0.1))
w_2 = tf.Variable(tf.random_normal((n_hidden_1, n_hidden_2), stddev=0.1))
w_3 = tf.Variable(tf.random_normal((n_hidden_2, n_hidden_3), stddev=0.1))
w_4 = tf.Variable(tf.random_normal((n_hidden_3, n_hidden_4), stddev=0.1))
w_5 = tf.Variable(tf.random_normal((n_hidden_4, n_classes), stddev=0.1))

# Bias initializations
b_1 = tf.Variable(tf.random_normal([n_hidden_1]))
b_2 = tf.Variable(tf.random_normal([n_hidden_2]))
b_3 = tf.Variable(tf.random_normal([n_hidden_3]))
b_4 = tf.Variable(tf.random_normal([n_hidden_4]))
b_5 = tf.Variable(tf.random_normal([n_classes]))

# Forward propagation
layer_1 = tf.nn.relu(tf.add(tf.matmul(X, w_1), b_1))
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w_2), b_2))
layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, w_3), b_3))
layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, w_4), b_4))
layer_out = tf.add(tf.matmul(layer_4, w_5), b_5)
predict = tf.argmax(layer_out, axis=1)

# Backward propagation
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=layer_out))
updates = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Run SGD
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

train_accuracy = np.zeros((training_epochs, 1))
test_accuracy = np.zeros((training_epochs, 1))
best_test_accuracy = 0
n_bad_epochs = 0

for epoch in range(training_epochs):
    # Train with each example
    for i in range(len(train_X)):
        sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

    train_accuracy[epoch] = np.mean(np.argmax(train_y, axis=1) == sess.run(predict, feed_dict={X: train_X, y: train_y}))
    test_accuracy[epoch] = np.mean(np.argmax(test_y, axis=1) == sess.run(predict, feed_dict={X: test_X, y: test_y}))

    print("Epoch = {}, train accuracy = {}%, test accuracy = {}%".format(epoch + 1,
                                                                         100. * train_accuracy[epoch],
                                                                         100. * test_accuracy[epoch]))

    if test_accuracy[epoch] > best_test_accuracy:
        best_test_accuracy = test_accuracy[epoch]
        n_bad_epochs = 0
    elif n_bad_epochs > 5:
        break
    else:
        n_bad_epochs = n_bad_epochs + 1



pred_y = sess.run(predict, feed_dict={X: pred_X})
sess.close()

x_axis = np.linspace(1, training_epochs, training_epochs)
plt.plot(x_axis, test_accuracy, 'r', np.linspace(1, training_epochs, training_epochs), train_accuracy, 'b')
plt.show()

Id_pred_start = 45324
Id_pred_stop = 53460
Id_pred = np.linspace(Id_pred_start, Id_pred_stop, Id_pred_stop - Id_pred_start + 1)

with open('prediction.csv', 'w') as csvfile:
    fieldnames = ['Id', 'y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(Id_pred)):
        writer.writerow({'Id': int(Id_pred[i]), 'y': int(pred_y[i])})



