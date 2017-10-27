import gzip, numpy, pickle
import _pickle as cPickle
import theano
import theano.tensor as T
import numpy 

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

# Function that loads the dataset into shared variables
def shared_dataset(data_xy):
    
    
    #Data is stored in theano shared variables so it can be copied directly
    #into the GPU memory, therefore avoiding overhead of copying data
    #from the cpu memory to the gpu memory
    
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))

    # Y data are labels which are integers, howeber data stored in the GPU
    # needs to be float. When working with labels they have to be integers
    # the return statement fixes that issue by using casting
    return shared_x, T.cast(shared_y, 'int32')

test_set_x, test_set_y = shared_dataset(test_set)
valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x, train_set_y = shared_dataset(train_set)

batch_size = 500    # size of the minibatch (Minibatch Stochastich Gradient Descent)

# accessing the third minibatch of the training set

data  = train_set_x[2 * batch_size: 3 * batch_size]
label = train_set_y[2 * batch_size: 3 * batch_size]

#Zero to one loss function:
#zero_one_loss = T.sum(T.neq(T.argmax(p_y_given_x), y))
