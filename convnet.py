from __future__ import division
import numpy as gakusei
import string
import re
import pandas
import lasagne
import theano
import lasagne.layers.conv as conv
import theano.tensor as T
import pickle as c
xar = c.load(open("mykawaii.pkl","rb"))
inp = xar.astype(gakusei.float64)

#yar = gakusei.array([4,5,4,6,4])
theano.config.exception_verbosity='high'

yar = gakusei.array([9,6,8,7,8,8,8,9,8,8,8,8,8,8,7,8,7,7,7,8,8,8,8,7,
                    7,7,7,7,7,6,7,7,5,7,7,7,7,7,7,5,6,6,6,6,6,6,6,6,6,
                    6,5,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,4,4])
yar = yar.astype(gakusei.uint8)
x=[]
xar = gakusei.asarray(xar)
t = T.tensor3('meh')
out = T.ivector('output')

def model(input_va):
    layer = lasagne.layers.InputLayer(shape=(None,1,3000),input_var = input_va)
    layer = lasagne.layers.Conv1DLayer(layer, num_filters = 7, filter_size = 80,
                 border_mode="valid", untie_biases=False,
                  b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify)
    
    layer = lasagne.layers.MaxPool1DLayer(incoming = layer,stride=1,pool_size = 3)
    
    layer = lasagne.layers.Conv1DLayer(layer, num_filters = 7, filter_size = 128,
                 border_mode="valid", untie_biases=False,
                  b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify)
    
    layer = lasagne.layers.MaxPool1DLayer(incoming = layer,stride=1,pool_size = 3)
    
    layer = lasagne.layers.Conv1DLayer(layer, num_filters = 7, filter_size = 256,
                 border_mode="valid", untie_biases=False,
                  b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify)
    
    layer = lasagne.layers.MaxPool1DLayer(incoming = layer,stride=1,pool_size = 3)
    layer = lasagne.layers.Conv1DLayer(layer, num_filters = 3, filter_size = 256,
                 border_mode="valid", untie_biases=False,
                  b=lasagne.init.Constant(0.),
                 nonlinearity=lasagne.nonlinearities.rectify)
    
   
    layer = lasagne.layers.MaxPool1DLayer(incoming = layer,stride=1,pool_size = 3)
    
    
    layer = lasagne.layers.DenseLayer(lasagne.layers.dropout(layer, p=.7),num_units=256,nonlinearity=lasagne.nonlinearities.rectify)
    
    
    layer = lasagne.layers.DenseLayer(lasagne.layers.dropout(layer, p=.7),num_units=10,nonlinearity=lasagne.nonlinearities.softmax)
    
    
    
    return layer
network = model(t)
prediction1 = lasagne.layers.get_output(network)
pre1 = theano.function([t],prediction1)

num_epochs=500
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, out)
loss = loss.mean()
params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,out)
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), out),
                      dtype=theano.config.floatX)
train_fn = theano.function([t, out], loss, updates=updates)
theano.optimizer='FAST_COMPILE'
def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = gakusei.arange(len(inputs))
        gakusei.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        
        yield inputs[excerpt], targets[excerpt]
for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        
        for batch in iterate_minibatches(xar, yar, 9, shuffle=True):
            inputs, targets = batch
            train_err = 0
            train_err += train_fn(inputs, targets)
            train_batches += 1
            lala = pre1(inputs)
            max = gakusei.argmax(lala)
            
            print (max)
            
            print train_err
            for x in lala:
                max = gakusei.argmax(x)
            
                print (max)

        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
#print len(xar),len(yar),xar,
