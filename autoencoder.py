from data import *
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import plot
import PIL
import numpy as np
import theano

class DenoisingAutoEncoder(object):
  def __init__(self,
      np_rng           = np.random.RandomState(1234),
      theano_rng       = None,
      input_           = T.dmatrix('input'),
      n_visible        = 3 * 424 * 424,
      n_hidden         = 500,
      weights          = None,
      tied_weights     = True,
      biases_visible   = None,
      biases_hidden    = None,
      learning_rate    = 0.01,
      corruption_level = 0.3):

    self.np_rng = np_rng
    if not theano_rng: theano_rng = RandomStreams(np_rng.randint(2 ** 30))
    self.theano_rng = theano_rng

    self.input = input_

    self.n_visible = n_visible
    self.n_hidden = n_hidden

    if not weights:
      initial_weights = self.generate_initial_weights((self.n_visible, self.n_hidden))
      weights = theano.shared(value = initial_weights, name = 'weights')
    self.weights = weights

    if tied_weights:
      weights_prime = self.weights.T
    else:
      inital_weights_prime = \
          self.generate_initial_weights((self.n_hidden, self.n_visible))
      weights_prime = \
          theano.shared(value = inital_weights_prime, name = 'weights_prime')
    self.weights_prime = weights_prime

    if not biases_visible:
      biases_visible = theano.shared(
          value = np.zeros(n_visible, dtype = theano.config.floatX),
          name = 'biases_visible')
    if not biases_hidden:
      biases_hidden = theano.shared(
          value = np.zeros(n_hidden, dtype = theano.config.floatX),
          name = 'biases_hidden')

    self.biases = biases_hidden
    self.biases_prime = biases_visible

    self.learning_rate = learning_rate

    self.corruption_level = corruption_level

    self.params = [self.weights, self.biases, self.biases_prime]
    if not tied_weights: self.params.append(self.weights_prime)

    y = self.hidden_values(self.corrupted_input())
    z = self.reconstructed_input(y)

    l = -T.sum(self.input * T.log(z) + (1 - self.input) * T.log(1 - z), axis = 1)
    self.cost = T.mean(l)

    grads = T.grad(self.cost, self.params)

    self.updates = [(param, param - self.learning_rate * grad)
        for param, grad in zip(self.params, grads)]

  def generate_initial_weights(self, size):
    high = 4 * np.sqrt(6.0 / (size[0] + size[1]))
    distribution = self.np_rng.uniform(
        low = -high,
        high = high,
        size = size)

    return np.asarray(distribution, dtype=theano.config.floatX)

  def hidden_values(self, input_):
    return T.nnet.sigmoid(T.dot(input_, self.weights) + self.biases)

  def reconstructed_input(self, hidden_values):
    return T.nnet.sigmoid(
        T.dot(hidden_values, self.weights_prime) + self.biases_prime)

  def corrupted_input(self):
    return self.input * self.theano_rng.binomial(
        size = self.input.shape, n = 1, p = 1 - self.corruption_level)

def dump_weights_as_image(da, title):
  weights = da.weights.get_value().T
  weights = weights.reshape(500, 424, 424, 3)
  red = weights[:, :, :, 0]
  blue = weights[:, :, :, 1]
  green = weights[:, :, :, 2]
  tiles = plot.tile_raster_images(X=(red, blue, green, None), img_shape=(424, 424), tile_shape=(10, 10), tile_spacing=(1, 1))
  image = PIL.Image.fromarray(tiles)
  image.save("images/%s.png" % title)

def train_autoencoder(da, galaxy_ids, batch_size = 20, n_epochs=10, dump_frequency=None):
  n_batches = galaxy_ids.shape[0] / batch_size

  train_x = T.dmatrix('train_x')
  train = theano.function([train_x], da.cost,
      updates = da.updates,
      givens = {da.input: train_x})

  for epoch in xrange(n_epochs):
    costs = []
    for index in xrange(n_batches):
      print "Minibatch index %d" % index
      images = load_images(galaxy_ids[index * batch_size:(index + 1) * batch_size])
      images = images.reshape(batch_size, 424 * 424 * 3)
      costs.append(train(images))

      if dump_frequency and (index * batch_size) % dump_frequency == 0:
        title = 'e%db%d' % (epoch, index)
        print "Dumping image %s" % title
        dump_weights_as_image(da, title)

    print "Training epoch %d, cost %f" % (epoch, np.mean(costs))

if __name__ == '__main__':
  da = DenoisingAutoEncoder(
      n_visible        = 3 * 424 * 424,
      n_hidden         = 500,
      tied_weights     = True,
      learning_rate    = 0.01,
      corruption_level = 0.3)

  data = load_data()
  galaxy_ids = data[:40, 0]

  train_autoencoder(da, galaxy_ids)

  import pdb; pdb.set_trace()
