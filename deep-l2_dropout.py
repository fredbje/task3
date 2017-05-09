print (__doc__)

from time import time
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Parameters
RANDOM_SEED = 42
batch_size = 9000
beta = 0.001

#Layers
hidden_nodes_1 = 1024
hidden_nodes_2 = int(hidden_nodes_1 * 0.5)
hidden_nodes_3 = int(hidden_nodes_1 * np.power(0.5, 2))
hidden_nodes_4 = int(hidden_nodes_1 * np.power(0.5, 3))
hidden_nodes_5 = int(hidden_nodes_1 * np.power(0.5, 4))

# Load data from hdf file
train_data = pd.read_hdf("train.h5", "train").as_matrix()
(N, M) = train_data.shape
n_features = M - 1
target = train_data[:, 0].astype(int)
n_classes = len(np.unique(target))
all_y = np.eye(n_classes)[target]  # One liner trick!
# Prepend a column of 1s for bias
all_X = train_data[:, 1:]
test_data = pd.read_hdf("test.h5", "test").as_matrix()
pred_X = test_data

tf.set_random_seed(RANDOM_SEED)
train_X, valid_X, train_y, valid_y = train_test_split(all_X, all_y, test_size=0.35, random_state=RANDOM_SEED)
valid_X, test_X, valid_y, test_y = train_test_split(valid_X, valid_y, test_size=0.05, random_state=RANDOM_SEED)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])

graph = tf.Graph()
with graph.as_default():
    '''Input Data'''
    # For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_X = tf.placeholder(tf.float32, shape=(batch_size, n_features))
    tf_train_y = tf.placeholder(tf.float32, shape=(batch_size, n_classes))
    tf_valid_X = tf.constant(valid_X)
    tf_test_X = tf.constant(test_X)
    tf_pred_X = tf.constant(pred_X)

    '''Variables'''
    # Hidden RELU layer 1
    weights_1 = tf.Variable(tf.truncated_normal([n_features, hidden_nodes_1],
                                                stddev=np.sqrt(2.0 / (n_features))))
    biases_1 = tf.Variable(tf.zeros([hidden_nodes_1]))

    # Hidden RELU layer 2
    weights_2 = tf.Variable(
        tf.truncated_normal([hidden_nodes_1, hidden_nodes_2], stddev=np.sqrt(2.0 / hidden_nodes_1)))
    biases_2 = tf.Variable(tf.zeros([hidden_nodes_2]))

    # Hidden RELU layer 3
    weights_3 = tf.Variable(
        tf.truncated_normal([hidden_nodes_2, hidden_nodes_3], stddev=np.sqrt(2.0 / hidden_nodes_2)))
    biases_3 = tf.Variable(tf.zeros([hidden_nodes_3]))

    # Hidden RELU layer 4
    weights_4 = tf.Variable(
        tf.truncated_normal([hidden_nodes_3, hidden_nodes_4], stddev=np.sqrt(2.0 / hidden_nodes_3)))
    biases_4 = tf.Variable(tf.zeros([hidden_nodes_4]))

    # Hidden RELU layer 5
    weights_5 = tf.Variable(
        tf.truncated_normal([hidden_nodes_4, hidden_nodes_5], stddev=np.sqrt(2.0 / hidden_nodes_4)))
    biases_5 = tf.Variable(tf.zeros([hidden_nodes_5]))

    # Output layer
    weights_6 = tf.Variable(tf.truncated_normal([hidden_nodes_5, n_classes], stddev=np.sqrt(2.0 / hidden_nodes_5)))
    biases_6 = tf.Variable(tf.zeros([n_classes]))

    '''Training computation'''

    # Hidden RELU layer 1
    logits_1 = tf.matmul(tf_train_X, weights_1) + biases_1
    hidden_layer_1 = tf.nn.relu(logits_1)
    # Dropout on hidden layer: RELU layer
    keep_prob = tf.placeholder("float")
    hidden_layer_1_dropout = tf.nn.dropout(hidden_layer_1, keep_prob)

    # Hidden RELU layer 2
    logits_2 = tf.matmul(hidden_layer_1_dropout, weights_2) + biases_2
    hidden_layer_2 = tf.nn.relu(logits_2)
    # Dropout on hidden layer: RELU layer
    hidden_layer_2_dropout = tf.nn.dropout(hidden_layer_2, keep_prob)

    # Hidden RELU layer 3
    logits_3 = tf.matmul(hidden_layer_2_dropout, weights_3) + biases_3
    hidden_layer_3 = tf.nn.relu(logits_3)
    # Dropout on hidden layer: RELU layer
    hidden_layer_3_dropout = tf.nn.dropout(hidden_layer_3, keep_prob)

    # Hidden RELU layer 4
    logits_4 = tf.matmul(hidden_layer_3_dropout, weights_4) + biases_4
    hidden_layer_4 = tf.nn.relu(logits_4)
    # Dropout on hidden layer: RELU layer

    hidden_layer_4_dropout = tf.nn.dropout(hidden_layer_4, keep_prob)

    # Hidden RELU layer 5
    logits_5 = tf.matmul(hidden_layer_4_dropout, weights_5) + biases_5
    hidden_layer_5 = tf.nn.relu(logits_5)
    # Dropout on hidden layer: RELU layer
    hidden_layer_5_dropout = tf.nn.dropout(hidden_layer_5, keep_prob)

    # Output layer
    logits_6 = tf.matmul(hidden_layer_5_dropout, weights_6) + biases_6

    # Normal loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_6, labels=tf_train_y))
    # Loss function with L2 Regularization with decaying learning rate beta=0.5
    regularizers = tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + \
                   tf.nn.l2_loss(weights_3) + tf.nn.l2_loss(weights_4) + \
                   tf.nn.l2_loss(weights_5) + tf.nn.l2_loss(weights_6)
    loss = tf.reduce_mean(loss + beta * regularizers)

    '''Optimizer'''
    # Decaying learning rate
    global_step = tf.Variable(0)  # count the number of steps taken.
    start_learning_rate = 0.5
    learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100000, 0.96, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Predictions for the training
    train_prediction = tf.nn.softmax(logits_6)

    # Predictions for validation
    valid_logits_1 = tf.matmul(tf_valid_X, weights_1) + biases_1
    valid_relu_1 = tf.nn.relu(valid_logits_1)

    valid_logits_2 = tf.matmul(valid_relu_1, weights_2) + biases_2
    valid_relu_2 = tf.nn.relu(valid_logits_2)

    valid_logits_3 = tf.matmul(valid_relu_2, weights_3) + biases_3
    valid_relu_3 = tf.nn.relu(valid_logits_3)

    valid_logits_4 = tf.matmul(valid_relu_3, weights_4) + biases_4
    valid_relu_4 = tf.nn.relu(valid_logits_4)

    valid_logits_5 = tf.matmul(valid_relu_4, weights_5) + biases_5
    valid_relu_5 = tf.nn.relu(valid_logits_5)

    valid_logits_6 = tf.matmul(valid_relu_5, weights_6) + biases_6

    valid_prediction = tf.nn.softmax(valid_logits_6)

    # Predictions for test
    test_logits_1 = tf.matmul(tf_test_X, weights_1) + biases_1
    test_relu_1 = tf.nn.relu(test_logits_1)

    test_logits_2 = tf.matmul(test_relu_1, weights_2) + biases_2
    test_relu_2 = tf.nn.relu(test_logits_2)

    test_logits_3 = tf.matmul(test_relu_2, weights_3) + biases_3
    test_relu_3 = tf.nn.relu(test_logits_3)

    test_logits_4 = tf.matmul(test_relu_3, weights_4) + biases_4
    test_relu_4 = tf.nn.relu(test_logits_4)

    test_logits_5 = tf.matmul(test_relu_4, weights_5) + biases_5
    test_relu_5 = tf.nn.relu(test_logits_5)

    test_logits_6 = tf.matmul(test_relu_5, weights_6) + biases_6

    test_prediction = tf.nn.softmax(test_logits_6)

    # Predictions for test
    pred_logits_1 = tf.matmul(tf_pred_X, weights_1) + biases_1
    pred_relu_1 = tf.nn.relu(pred_logits_1)

    pred_logits_2 = tf.matmul(pred_relu_1, weights_2) + biases_2
    pred_relu_2 = tf.nn.relu(pred_logits_2)

    pred_logits_3 = tf.matmul(pred_relu_2, weights_3) + biases_3
    pred_relu_3 = tf.nn.relu(pred_logits_3)

    pred_logits_4 = tf.matmul(pred_relu_3, weights_4) + biases_4
    pred_relu_4 = tf.nn.relu(pred_logits_4)

    pred_logits_5 = tf.matmul(pred_relu_4, weights_5) + biases_5
    pred_relu_5 = tf.nn.relu(pred_logits_5)

    pred_logits_6 = tf.matmul(pred_relu_5, weights_6) + biases_6

    pred_prediction = tf.nn.softmax(pred_logits_6)

num_steps = 5000

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print("Initialized")
    for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_y.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_X[offset:(offset + batch_size), :]
        batch_labels = train_y[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_X: batch_data, tf_train_y: batch_labels, keep_prob: 0.5}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step {}: {}".format(step, l))
            print("Minibatch accuracy: {:.1f}".format(accuracy(predictions, batch_labels)))
            print("Validation accuracy: {:.1f}".format(accuracy(valid_prediction.eval(), valid_y)))
    print("Test accuracy: {:.1f}".format(accuracy(test_prediction.eval(), test_y)))

    pred_y = tf.argmax(pred_prediction, axis=1).eval()

    print(pred_y)
    print(pred_y.shape)
    print(53460-45324)

    Id_pred_start = 45324
    Id_pred_stop = 53460
    Id_pred = np.linspace(Id_pred_start, Id_pred_stop, Id_pred_stop - Id_pred_start + 1)

    with open('prediction.csv', 'w') as csvfile:
        fieldnames = ['Id', 'y']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(Id_pred)):
            writer.writerow({'Id': int(Id_pred[i]), 'y': int(pred_y[i])})

