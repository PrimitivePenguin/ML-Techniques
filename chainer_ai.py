import chainer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import chainer.links as L
import chainer.functions as F
from chainer import Sequential
import numpy as np
import matplotlib.pyplot as plt
import time

# Loading Iris dataset
x, t = load_iris(return_X_y=True)


print('x:', x.shape)
print('t:', t.shape)

#Changing data types to match Chainer
x = x.astype('float32')
t = t.astype('int32')

#Classifying data
x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.3, random_state=0)
x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.3, random_state=0)
#Deciding networks
#Defining network using Sequential
l = L.Linear(3, 2)

n_input = 4
n_hidden = 10
n_output = 3

net = Sequential(
    L.Linear(n_input, n_hidden), F.relu,
    L.Linear(n_hidden, n_hidden), F.relu,
    L.Linear(n_hidden, n_output)
)

#Deciding objective function
optimizer = chainer.optimizers.SGD(lr=0.01)
#Setting up gradient descent
optimizer.setup(net)

n_epoch = 30
n_batchsize = 16

#Training
iteration = 0

# Saving logs
results_train = {
    'loss': [],
    'accuracy': []
}
results_valid = {
    'loss': [],
    'accuracy': []
}
i_time = time.time()

for epoch in range(n_epoch):

    # Contains how the datasets are reordered 
    order = np.random.permutation(range(len(x_train)))

    # List for keeping the output of the objective function and classification
    loss_list = []
    accuracy_list = []

    for i in range(0, len(order), n_batchsize):
        # Prepare the batch
        index = order[i:i+n_batchsize]
        x_train_batch = x_train[index,:]
        t_train_batch = t_train[index]

        # Calculate the training value
        y_train_batch = net(x_train_batch)

        # Optimize the objective function and calculate the accuracy of classification
        loss_train_batch = F.softmax_cross_entropy(y_train_batch, t_train_batch)
        accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)

        loss_list.append(loss_train_batch.array)
        accuracy_list.append(accuracy_train_batch.array)

        # Resetting and calculating the gradient
        net.cleargrads()
        loss_train_batch.backward()

        # Update the parameters
        optimizer.update()

        # Increase count
        iteration += 1

    # List for comparison of the output of the objective function and classification against the training data
    loss_train = np.mean(loss_list)
    accuracy_train = np.mean(accuracy_list)

    # When 1 epoch is finished, evaluate the data
    # Use the data to predict
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y_val = net(x_val)

    # Using the objective function, calculate the accuracy
    loss_val = F.softmax_cross_entropy(y_val, t_val)
    accuracy_val = F.accuracy(y_val, t_val)

    # Displaying results 
    print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(
        epoch, iteration, loss_train, loss_val.array))

    # Saving logs
    results_train['loss'] .append(loss_train)
    results_train['accuracy'] .append(accuracy_train)
    results_valid['loss'].append(loss_val.array)
    results_valid['accuracy'].append(accuracy_val.array)
e_time = time.time()
t_time = e_time - i_time
print("Differenece between first and last epoch: epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}".format(epoch, iteration, loss_train, loss_val.array))
# Output of objective function
plt.plot(results_train['loss'], label='train') 
plt.plot(results_valid['loss'], label='valid')
plt.legend()  # Displaying legend

# (accuracy)
plt.plot(results_train['accuracy'], label='train')  
plt.plot(results_valid['accuracy'], label='valid')  
plt.legend()  
plt.show()

# Using test data
with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
    y_test = net(x_test)

# Calculating accuracy
accuracy_test = F.accuracy(y_test, t_test)
accuracy_test.array

# Saving the network
chainer.serializers.save_npz('my_iris.net', net)
print(t_time)