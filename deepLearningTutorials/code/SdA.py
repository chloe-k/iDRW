"""
 This tutorial introduces stacked denoising auto-encoders (SdA) using Theano.

 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""

from __future__ import print_function

import os
import sys
import timeit

import numpy
from sklearn import preprocessing
import pandas as pd
import csv

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import LogisticRegression, load_data, return_geneid
from mlp import HiddenLayer
from dA import dA
from numpy import *
import pdb


# start-snippet-1
class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=247,
        hidden_layers_sizes=[200],
        n_outs=100,
        corruption_levels=[0.1, 0.1]
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(123))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        #self.x = T.vector('x')
        
        # end-snippet-1

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        # start-snippet-2
        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            self.params.extend(sigmoid_layer.params)

            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)
        # end-snippet-2
        '''
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs
        )

        self.params.extend(self.logLayer.params)
        # construct a function that implements one step of finetunining

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)
        '''

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use

        # begining of a batch, given `index`
        batch_begin = index * batch_size
        #batch_begin = index * (batch_size-1)
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size
        pretrain_fns = []
        
        
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)

            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(corruption_level, value=0.01),
                    theano.In(learning_rate, value=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                    #self.x : train_set_x[batch_begin:]
                }
                
            )
           
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
    
    
    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score
        
        

    def return_activity(self, train_set_x):
        '''Given an input, this function returns the activity
        value of all the nodes in each hidden layer.'''
        
        activity_each_layer = []
        index = T.lscalar('index')  # index to a sample
        
        for dA in self.dA_layers:
            activity_fn = theano.function(inputs=[index],outputs = dA.output,
                                          givens={self.x: train_set_x[index:(index+1)]})
            activity_each_layer.append(activity_fn)
        return activity_each_layer

    def return_raw_activity(self, train_set_x):
        '''Given an input, this function returns the raw activity
        value of all the nodes in each layer.'''
        
        raw_activity_each_layer = []
        index = T.lscalar('index')  # index to a sample
        
        for dA in self.dA_layers:
            raw_activity_fn = theano.function(inputs=[index],outputs = dA.raw_output,
                                              givens={self.x: train_set_x[index:(index+1)]})
            raw_activity_each_layer.append(raw_activity_fn)
        return raw_activity_each_layer

    def return_network(self):
        '''This function returns weight matrix and bias vectors of each hidden layer in the 
        final network after training.'''

        weights_all_layer = []
        bias_all_layer = []
        bias_prime_all_layer = []

        for dA_layer in self.dA_layers:
            weight = dA_layer.W.get_value(borrow = True)
            bias = dA_layer.b.get_value(borrow = True)
            bias_prime = dA_layer.b_prime.get_value(borrow = True)
            weights_all_layer.append(weight)
            bias_all_layer.append(bias)
            bias_prime_all_layer.append(bias_prime)

        return weights_all_layer, bias_all_layer, bias_prime_all_layer
    
    
def test_SdA(finetune_lr=0.1, pretraining_epochs=15, pretrain_lr=0.001, training_epochs=1000,
        dataset='../data/BRCA_rnaseq_methylation/brca_rnaseq_imputation.csv', batch_size=1,
        output_file = "rnaseq_result_activity_value.tsv", net_file = "rnaseq_result_net.tsv",
        high_weight_feature_file = "rnaseq_result_compressed_data.tsv",
        normalized_pathway_distribution = "_result_normalized_data.tsv",):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """
    
    # load datasets
    # n_ins : # of sample
    datasets = load_data(dataset)
    train_set_x = datasets
    n_ins = 227
    
    # compute number of minibatches for training, validation and testing
    train_size = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = train_size / batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(123)
    print('... building the model')
    
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_ins,
        hidden_layers_sizes=[200],
        n_outs=n_ins
    )
    
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print('... getting the pretraining functions')
   
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print('... pre-training the model')
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels = [0.1]
    loss = []
    for i in range(sda.n_layers):
        # go through pretraining epochs
        for epoch in range(pretraining_epochs):
            # go through the training set  
            c = []
            for batch_index in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c, dtype='float64')))
            loss.append(numpy.mean(c,dtype='float64'))

    end_time = timeit.default_timer()
    print(('The pretraining code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

    # end-snippet-4

    ##############################################################
    # Return the final activity value and raw activity value
    # for each node of each input sample 
    ##############################################################
    
    output_fh = open(output_file,'w')
    raw_output_fh = open(output_file.replace('activity','rawActivity'),'w')
    each_layer_output = sda.return_activity(train_set_x=train_set_x)
    each_layer_raw_output = sda.return_raw_activity(train_set_x=train_set_x)
    for i in xrange(sda.n_layers):
        output_fh.write('layer %i \n' %(i+1))
        raw_output_fh.write('layer %i \n' %(i+1))
        for train_sample in xrange(train_size):
            node_activation = each_layer_output[i](train_sample)
            node_raw_activation = each_layer_raw_output[i](train_sample)
            numpy.savetxt(output_fh, node_activation, fmt= '%.8f', delimiter= '\t') 
            numpy.savetxt(raw_output_fh, node_raw_activation, fmt= '%.8f', delimiter= '\t')
    output_fh.close()
    raw_output_fh.close()
    
    ##############################################################
    # Return weight matrix and bias vectors of the final network #
    ##############################################################
    net_fh = open(net_file,'w')
    weight_output, bias_output, bias_prime_output = sda.return_network()
    for i in xrange(len(weight_output)):
        net_fh.write('layer %i \n' %(i+1))
        net_fh.write('weight matrix \n')
        numpy.savetxt(net_fh, weight_output[i], fmt= '%.8f', delimiter = '\t') 
        net_fh.write('hidden bias vector \n')
        numpy.savetxt(net_fh, bias_output[i], fmt= '%.8f', delimiter = '\t')
        net_fh.write('visible bias vector \n')
        numpy.savetxt(net_fh, bias_prime_output[i], fmt= '%.8f', delimiter = '\t') 
    net_fh.close()

    ################################################################
    # Read weight matrix and geneid list
    ################################################################
    geneid = return_geneid(dataset)
    input_size = n_ins
    weight = []
    recon_error = []
    input_count = 0
    network_fh = open(net_file, 'r')
    network_fh.next()
    network_fh.next()
    for i in network_fh:
        i = i.strip().split('\t')
        weight.append(i)    
        input_count += 1
        if input_count == input_size:
            break
    recon_error = loss 
    weight = numpy.array(weight, dtype = float)
    thre = numpy.float32(0)
    #pdb.set_trace()
    #################################################################
    # Return for normalized weight matrix
    # weight matrix - sample(row) * weight to hidden node(col)
    #################################################################
    with open(normalized_pathway_distribution, 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter = '\t')
        for node in xrange(weight.shape[0]):
            list = []
            list = preprocessing.scale(weight[node,:]) 
            writer.writerow(list)
    
    #################################################################
    # Return for sorted weight and reconstruction error
    #################################################################
    sorted_pathway = []
    geneid_list = []
    df = pd.read_csv(normalized_pathway_distribution, sep='\t', header=None)
    nor_weight = df.as_matrix()
    #pdb.set_trace()
    #or_weight = numpy.delete(nor_weight,(nor_weight.shape[1]-1), axis=1)
    nor_weight = numpy.transpose(nor_weight) 
    ''' 
    for node in xrange(nor_weight.shape[1]):
        aver_weight = numpy.mean(nor_weight[:,node])
        std_weight = numpy.std(nor_weight[:,node])
        cutoff_up = aver_weight + thre*std_weight
        cutoff_down = aver_weight - thre*std_weight

        for gene in xrange(nor_weight.shape[0]):
            if nor_weight[gene,node] >= cutoff_up or nor_weight[gene, node] <= cutoff_down:
                sorted_pathway.append(abs(nor_weight[gene,node]))
                geneid_list.append(geneid[gene])

        out_fh = open(high_weight_feature_file, 'w')
        out_fh.write('pathway\tweight\n')
    pdb.set_trace() 
    '''
    
    for pathway in xrange(nor_weight.shape[0]):
        sum = 0
        for node in xrange(nor_weight.shape[1]):
            sum = sum+abs(nor_weight[pathway][node])
        aver_pathway = abs(sum/nor_weight.shape[1])
        
        if aver_pathway >= thre:
            sorted_pathway.append(aver_pathway)
            geneid_list.append(geneid[pathway])
    
    '''
    for sample in xrange(nor_weight.shape[0]):
        sum = 0
        for pathway in xrange(nor_weight.shape[1]):
    '''
    ##unsorted weight information
    ''' 
    for gene in xrange(nor_weight.shape[0]):
        if nor_weight[gene,node] >= cutoff_up or nor_weight[gene, node] <= cutoff_down:
            out_fh.write(geneid[gene]+'\t'+str(nor_weight[gene, node])+'\n')
    '''
    ##sorted weight information
    '''
    for i in xrange(len(sorted_pathway)):
        print(geneid_list[i]+'\t'+str(sorted_pathway[i])+'\n')
    '''
    for i in xrange(len(sorted_pathway)):
        for j in xrange(len(sorted_pathway)-1):
            if(sorted_pathway[j] < sorted_pathway[j+1]):
                temp = sorted_pathway[j+1]
                sorted_pathway[j+1] = sorted_pathway[j]
                sorted_pathway[j] = temp
                id_temp = geneid_list[j+1]
                geneid_list[j+1] = geneid_list[j]
                geneid_list[j] = id_temp
    out_fh = open(high_weight_feature_file, 'w')
    out_fh.write('pathway\tweight\n')
    '''
    for i in range(50):
        out_fh.write(geneid_list[i]+'\t'+str(sorted_pathway[i])+'\n')
    '''
    for i in xrange(len(sorted_pathway)):
        out_fh.write(geneid_list[i]+'\t'+str(sorted_pathway[i])+'\n')
    #reconstruction error
    out_fh.write('reconstruction error\n')

    for i in range(len(recon_error)):
        out_fh.write(str(recon_error[i])+'\n')

    out_fh.close()
    ########################
    # FINETUNING THE MODEL #
    ########################
'''
    # get the training, validation and testing function for the model
    print('... getting the finetuning functions')
    train_fn, validate_model, test_model = sda.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
    )

    print('... finetunning the model')
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses, dtype='float64')
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    print(('The training code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

'''
if __name__ == '__main__':
    test_SdA(0.01, 1500,  0.1, 1000, sys.argv[1], 1, sys.argv[2]+"_result_activity_value.tsv",
            sys.argv[2]+"_result_net.tsv", sys.argv[2]+"_compressed_data.tsv",
            sys.argv[2]+"_normalized.tsv")
