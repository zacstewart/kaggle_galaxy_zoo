from scipy import ndimage
import numpy as np

def load_images(galaxy_ids = []):
  def load_image(galaxy_id):
    return np.asarray(ndimage.imread("data/images_training/%d.jpg" % galaxy_id))
  return np.array(map(load_image, galaxy_ids), dtype=float) / 255.0

def load_data():
  print "Loading solutions"
  return np.loadtxt('data/solutions_training.csv', skiprows=1, delimiter=',')
