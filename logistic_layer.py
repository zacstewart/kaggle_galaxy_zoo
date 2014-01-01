from data import *
from theano import tensor as T
import numpy as np
import theano

class LogisticLayer(object):
  def __init__(self, input, n_in, n_out):
    self.weights = theano.shared(
        value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
        name='weights')
    self.biases = theano.shared(
        value=np.zeros((n_out,), dtype=theano.config.floatX),
        name='biases')

    self.p_y_given_x = T.nnet.softmax(T.dot(input, self.weights) + self.biases)

    self.params = [self.weights, self.biases]

  def root_mean_squared_error(self, y):
    return T.sqrt(T.mean((self.p_y_given_x - y) ** 2))

def train_logistic_regression(batch_size=20, learning_rate=0.01, n_epochs=1000):
  train, validate = load_data()
  train_ids = train[0]
  train_classes = train[1]
  validate_ids = validate[0]
  validate_classes = validate[1]

  n_train_batches = train_ids.shape[0] / batch_size

  x = T.dmatrix()
  y = T.dmatrix()

  classifier = LogisticLayer(
      input=x,
      n_in=424 * 424 * 3,
      n_out=37)

  cost = classifier.root_mean_squared_error(y)
  grads = T.grad(cost, classifier.params)

  updates = [(param, param - grad * learning_rate)
      for param, grad in zip(classifier.params, grads)]

  train_model = theano.function([x, y],
      outputs=cost,
      updates=updates)

  validate_model = theano.function([x, y], outputs=cost)

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
      classes = train_classes[index * batch_size:(index + 1) * batch_size]
      batch_avg_cost = train_model(images, classes)

      print "Batch %d avg cost %f" % (index, batch_avg_cost)

      if index % 10 == 0:
        images = load_images(validate_ids[:100]).reshape(100, 424 * 424 * 3)
        cv_score = validate_model(images, validate_classes[:100])
        print "Batch %d validation cost %f" % (index, cv_score)

if __name__ == '__main__':
  train_logistic_regression()
