from __future__ import print_function 
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np
import os,sys,math
import orphics.tools.io as io

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches



class FeedForward(object):
    """
    A generic feed-forward neural network
    with arbitrary number of layers.
    """
    
    def __init__(self,num_features,num_nodes=[64,32],activations=None,num_outputs=2,seed=None):

        
        self.X = tf.placeholder(tf.float32, shape=(num_features,None),name="X")
        self.Y = tf.placeholder(tf.float32, shape=(num_outputs,None),name="Y")
        self.num_features = num_features
        self.num_outputs = num_outputs

        if activations is None:
            activations = ["relu"]*len(num_nodes)
        num_nodes.append(num_outputs)
        activations.append("none")
        
        self.nlayers = len(num_nodes)
        assert len(activations) == self.nlayers
        self.activations = activations

        self.W = []
        self.b = []
        wprev = num_features
        with tf.variable_scope("reuse"):
            for l in range(self.nlayers):
                self.W.append (tf.get_variable("W"+str(l), [num_nodes[l],wprev], initializer = tf.contrib.layers.xavier_initializer(seed = seed)))
                self.b.append (tf.get_variable("b"+str(l), [num_nodes[l],1], initializer = tf.zeros_initializer()))
                wprev = num_nodes[l]

            
    def activate(self,Z,l):
        act = self.activations[l]
        if act=="relu":
            return tf.nn.relu(Z)
        elif act=="none":
            return Z
        else:
            raise NotImplementedError
        
    def forward(self):

        prev = self.X
        for l in range(self.nlayers):
            W = self.W[l]
            b = self.b[l]
            Z = tf.add(tf.matmul(W,prev),b)
            A = self.activate(Z,l)
            prev = A

        return A
    
    def cost(self,Z,Y):
        return tf.squared_difference(Z,Y)


    def train(self,X_train,Y_train,learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32):

        assert X_train.shape[0] == self.num_features
        assert Y_train.shape[0] == self.num_outputs
        m = X_train.shape[1]
        assert Y_train.shape[1] == m
        tf.set_random_seed(1)
        seed = 3

        Z = self.forward()
        cost = tf.reduce_mean(self.cost(Z,self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        costs = []
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(num_epochs):

                try:
                    epoch_cost = 0.                       # Defines a cost related to an epoch
                    num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                    seed = seed + 1
                    minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

                    for minibatch in minibatches:

                        # Select a minibatch
                        (minibatch_X, minibatch_Y) = minibatch

                        _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})

                        epoch_cost += minibatch_cost / num_minibatches

                    # Print the cost every epoch
                    if epoch % 10 == 0:
                        print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                    if epoch % 5 == 0:
                        costs.append(epoch_cost)

                except KeyboardInterrupt:
                    break
                
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.savefig("cost.png")

            # lets save the parameters
            for l in range(self.nlayers):
                self.W[l] = tf.get_variable("W"+str(l), initializer = tf.constant(sess.run(self.W[l])))
                self.b[l] = tf.get_variable("b"+str(l), initializer = tf.constant(sess.run(self.b[l])))
        
            
    def predict(self,X):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            Z = self.forward()
            Y_pred = sess.run(Z,feed_dict={self.X:X})

        return Y_pred
        
    def test(self,X_test,Y_test):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            Z = self.forward()
            cost = self.cost(Z,self.Y)
            mse = sess.run(cost,feed_dict={self.X:X_test,self.Y:Y_test})
        return mse
        
def prepare(images):
    from sklearn.preprocessing import StandardScaler

    assert images.ndim==3
    m = images.shape[-1]
    ny,nx = images.shape[:-1]
    images = images.reshape((nx*ny,m))

    scaler = StandardScaler()
    std_scale = scaler.fit(images)
    return std_scale.transform(images)

def sim(num_images,ny=32,nx=32,num_outputs=2):
    m = num_images
    Y_train = np.random.uniform(1,10,size=(num_outputs,m)).astype(np.float32)
    training_images = np.random.random((ny,nx,m)).astype(np.float32)
    training_images[:ny//2,:nx//2,:] += Y_train[0,:]
    training_images[ny//2:,nx//2:,:] += Y_train[1,:]
    return training_images,Y_train


num_outputs = 2 # the two ellipticity components
m = 1000000
ny = nx = 16
num_nodes = [32,16]

training_images,Y_train = sim(m,ny,nx,num_outputs)

img = training_images[:,:,0]
io.quickPlot2d(img,"img.png")
Npix = img.size
my_net = FeedForward(num_features=Npix,num_outputs=num_outputs,num_nodes=num_nodes)


X_train = prepare(training_images)

my_net.train(X_train,Y_train,learning_rate = 0.0001, num_epochs = 200, minibatch_size = 32)

ntest = 1000
test_images,Y_test = sim(ntest,ny,nx,num_outputs)
X_test = prepare(test_images)


print (Y_test)
Y_pred = my_net.predict(X_test)
Y_rand = np.random.uniform(1,10,size=(num_outputs,ntest))

err = (Y_pred-Y_test)
err_rand = (Y_rand-Y_test)
plt.clf()
plt.scatter(err_rand[0,:],err_rand[1,:],alpha=0.1)
plt.scatter(err[0,:],err[1,:])
plt.axvline(x=0.,ls="--")
plt.axhline(y=0.,ls="--")
plt.xlim(-10,10)
plt.ylim(-10,10)
plt.savefig("scatter.png")
