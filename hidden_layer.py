from theano import tensor as T
import numpy as np
import theano

class HiddenLayer(object):
  def __init__(self,
    np_rng,
    input,
    n_in,
    n_out,
    weights=None,
    biases=None,
    activation=T.tanh):

    self.input = input

    if weights is None:
      high = np.sqrt(6.0 / (n_in + n_out))
      weight_values = np.asarray(np_rng.uniform(
          high=high,
          low=-high,
          size=(n_in, n_out)),
        dtype=theano.config.floatX)

      if activation == T.nnet.sigmoid:
        weight_values *= 4

      self.weights = theano.shared(
        value=weight_values,
        name='weights',
        borrow=True)

    if biases is None:
      bias_values = np.zeros((n_out,), dtype=theano.config.floatX)
      self.biases = theano.shared(
        value=bias_values,
        name='biases',
        borrow=True)

    linear_output = T.dot(input, self.weights) + self.biases
    if activation is None:
      self.output = linear_output
    else:
      self.output = activation(linear_output)

    self.params = [self.weights, self.biases]
