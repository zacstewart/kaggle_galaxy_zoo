from scipy import ndimage
import numpy as np
import os

np.random.seed(1234)

def load_images(galaxy_ids=[], image_set='images_train'):
  def load_image(galaxy_id):
    path = os.path.join('data', image_set)
    return np.asarray(ndimage.imread("%s/%d.jpg" % (path, galaxy_id)))
  return np.array(map(load_image, galaxy_ids), dtype=float) / 255.0

def load_data():
  print "Loading solutions"
  solutions = np.loadtxt('data/solutions_training.csv', skiprows=1, delimiter=',')
  indices = np.random.permutation(solutions.shape[0])
  validate_size = len(indices) * 0.2
  train_idx, validate_idx = indices[validate_size:], indices[:validate_size]
  train, validate = solutions[train_idx, :], solutions[validate_idx, :]

  train_ids, train_y = train[:, 0], train[:, 1:]
  validate_ids, validate_y = validate[:, 0], validate[:, 1:]
  return ((train_ids, train_y), (validate_ids, validate_y))
