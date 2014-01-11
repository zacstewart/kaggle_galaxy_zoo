from data import *
from hidden_layer import HiddenLayer
from logistic_layer import LogisticLayer
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import plot
import PIL
import numpy as np
import os
import theano

class DenoisingAutoEncoder(object):
  def __init__(self,
      np_rng           = np.random.RandomState(1234),
      theano_rng       = None,
      input            = T.dmatrix('input'),
      n_visible        = 3 * 424 * 424,
      n_hidden         = 500,
      weights          = None,
      tied_weights     = True,
      biases_visible   = None,
      biases_hidden    = None,
      learning_rate    = 0.001,
      corruption_level = 0.3):

    self.np_rng = np_rng
    if not theano_rng: theano_rng = RandomStreams(np_rng.randint(2 ** 30))
    self.theano_rng = theano_rng

    self.input = input

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

  def hidden_values(self, input):
    return T.nnet.sigmoid(T.dot(input, self.weights) + self.biases)

  def reconstructed_input(self, hidden_values):
    return T.nnet.sigmoid(
        T.dot(hidden_values, self.weights_prime) + self.biases_prime)

  def corrupted_input(self):
    return self.input * self.theano_rng.binomial(
        size = self.input.shape, n = 1, p = 1 - self.corruption_level)

class StackedDenoisingAutoencoders(object):
  def __init__(self,
      np_rng             = np.random.RandomState(1234),
      theano_rng         = None,
      n_in               = 424 * 424 * 3,
      n_out              = 37, # galaxy classes
      hidden_layer_sizes = [500, 500],
      corruption_levels  = [0.1, 0.2]):

    self.np_rng = np_rng
    if not theano_rng: theano_rng = RandomStreams(np_rng.randint(2 ** 30))
    self.n_in = n_in
    self.n_out = n_out
    self.hidden_layer_sizes = hidden_layer_sizes
    self.corruption_levels = corruption_levels

    self.sigmoid_layers = []
    self.da_layers = []
    self.params = []
    self.n_layers = len(hidden_layer_sizes)

    assert self.n_layers > 0, 'must have some hidden layers'

    self.x = T.dmatrix('x')
    self.y = T.dmatrix('y')

    self.build_layers()

  def build_layers(self):
    for i in xrange(self.n_layers):
      if i == 0:
        input_size = self.n_in
        layer_input = self.x
      else:
        input_size = self.hidden_layer_sizes[i - 1]
        layer_input = self.sigmoid_layers[-1].output

      sigmoid_layer = HiddenLayer(
          np_rng=self.np_rng,
          input=layer_input,
          n_in=input_size,
          n_out=self.hidden_layer_sizes[i],
          activation=T.nnet.sigmoid)
      self.sigmoid_layers.append(sigmoid_layer)
      self.params.extend(sigmoid_layer.params)

      da_layer = DenoisingAutoEncoder(
        np_rng=self.np_rng,
        input=layer_input,
        n_visible=input_size,
        n_hidden=self.hidden_layer_sizes[i],
        weights=sigmoid_layer.weights,
        biases_hidden=sigmoid_layer.biases,
        corruption_level=self.corruption_levels[i])
      self.da_layers.append(da_layer)

    self.logistic_layer = LogisticLayer(
        input=self.sigmoid_layers[-1].output,
        n_in=self.hidden_layer_sizes[-1],
        n_out=self.n_out)
    self.params.extend(self.logistic_layer.params)

    self.fine_tune_cost = self.logistic_layer.root_mean_squared_error(self.y)

def dump_weights_as_image(da, title):
  weights = da.weights.get_value().T
  weights = weights.reshape(500, 424, 424, 3)
  red = weights[:, :, :, 0]
  blue = weights[:, :, :, 1]
  green = weights[:, :, :, 2]
  tiles = plot.tile_raster_images(X=(red, blue, green, None), img_shape=(424, 424), tile_shape=(10, 10), tile_spacing=(1, 1))
  image = PIL.Image.fromarray(tiles)
  image.save("images/%s.png" % title)

def pretrain_autoencoder(sda, da, galaxy_ids, batch_size=1, n_epochs=10, dump_frequency=None):
  n_batches = galaxy_ids.shape[0] / batch_size

  train_x = T.dmatrix('train_x')
  train = theano.function([train_x], da.cost,
      updates = da.updates,
      givens = {sda.x: train_x})

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

def fine_tune_autoencoder(
    sda,
    train_ids,
    train_y,
    validate_ids,
    validate_y,
    batch_size=1,
    learning_rate=0.01,
    n_epochs=1000):

  n_train_batches = train_ids.shape[0] / batch_size

  grads = T.grad(sda.fine_tune_cost, sda.params)

  updates = [(param, param - grad * learning_rate)
      for param, grad in zip(sda.params, grads)]

  x = T.dmatrix('x')
  y = T.dmatrix('y')

  train_model = theano.function([x, y],
      outputs=sda.fine_tune_cost,
      updates=updates,
      givens={
        sda.x: x,
        sda.y: y})

  validate_model = theano.function([x, y],
      outputs=sda.fine_tune_cost,
      givens={
        sda.x: x,
        sda.y: y})

  patience = 5000
  patience_increase = 2
  improvement_threshold = 0.995

  best_params = None

  done_looping = False
  epoch = 0
  while (epoch < n_epochs) and (not done_looping):
    epoch += 1

    for index in xrange(n_train_batches):
      images = load_images(
          train_ids[index * batch_size:(index + 1) * batch_size])
      images = images.reshape((batch_size, 424 * 424 * 3))
      classes = train_y[index * batch_size:(index + 1) * batch_size]
      batch_avg_cost = train_model(images, classes)

      print "Batch %d avg cost %f" % (index, batch_avg_cost)

      if index % 10 == 0:
        images = load_images(validate_ids[:100]).reshape(100, 424 * 424 * 3)
        cv_score = validate_model(images, validate_y[:100])
        print "Batch %d validation cost %f" % (index, cv_score)

if __name__ == '__main__':
  data = load_data()
  train_ids, train_y = data[0][0][:40], data[0][1][:40]
  validate_ids, validate_y = data[1][0], data[1][1]

  sda = StackedDenoisingAutoencoders(
      n_in=424 * 424 * 3,
      n_out=37,
      hidden_layer_sizes=[500, 500],
      corruption_levels=[0.2, 0.3])

  for da in sda.da_layers:
    'Training layer %s' % da
    #pretrain_autoencoder(sda, da, train_ids)

  #fine_tune_autoencoder(sda, train_ids, train_y, validate_ids, validate_y)

  print 'Predicting'
  x = T.dmatrix('x')
  predict = theano.function([x],
      outputs=sda.logistic_layer.p_y_given_x,
      givens={sda.x: x})

  test_ids = map(lambda f: int(f.split('.')[0]), os.listdir('data/images_test'))
  predictions = np.zeros((len(test_ids), 37))
  prediction_batch_size = 100
  n_prediction_batches = len(test_ids) / prediction_batch_size

  for index in xrange(n_prediction_batches):
    print "Predicting batch %d/%d" % (index, n_prediction_batches)
    begin = index * prediction_batch_size
    end = (index + 1) * prediction_batch_size
    images = load_images(test_ids[begin:end], 'images_test').reshape(prediction_batch_size, 424 * 424 * 3)
    predictions[begin:end] = predict(images)
  predictions[n_prediction_batches * prediction_batch_size:] = load_images(test_ids[n_prediction_batches * prediction_batch_size:], 'images_test')

  submission = np.array(test_ids, predictions).transpose()

  fmt = ['%d'] + ['%f'] * 37
  np.savetxt('submission.csv', submission, fmt=fmt, delimiter=',')

  import pdb; pdb.set_trace()
