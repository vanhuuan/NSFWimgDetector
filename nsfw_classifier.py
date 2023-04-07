import sys
import joblib
import numpy as np
import skimage
from bs4 import BeautifulSoup
import requests
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from sklearn.base import BaseEstimator, TransformerMixin

inputURL = sys.argv[1]

# Get list of all img sources
allsrc = []
srcimg = []
allimg = []

headers = {'User-Agent':r'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36'}
page = requests.get(inputURL, headers=headers)
soup = BeautifulSoup(page.text, 'html.parser')
all = soup.find_all('img')
print('Finding image sources...')
for i,_ in enumerate(all):
  datasrc = _.get('data-src')
  if (datasrc is not None):
    print(i,datasrc)
    allsrc.append(datasrc)
  else:
    src = _.get('src')
    if (src is not None):
      print(i,src)
      allsrc.append(src)

# Checking URLs
print("Checking URLs...")
for i,url in enumerate(allsrc):
  print(i,url)
  if 'http' not in url:
      url = '{}{}'.format(inputURL, url)
  srcimg.append(url)

# Read all images, resize to 150x150 and append in array
print('Reading jpg and png images...')
for i,_ in enumerate(srcimg):
  try:
    print(i,_)
    img = imread(_)
    img = resize(img, (150,150))
    allimg.append(img)
  except:
    pass

print('')
print('Phân tích trang web: ' + inputURL)
print('')
if len(allimg) > 0:
  # Transformer define start
  class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
      """
      Convert an array of RGB images to grayscale
      """
      def __init__(self):
          pass
  
      def fit(self, X, y=None):
          """returns itself"""
          return self
  
      def transform(self, X, y=None):
          """perform the transformation and return an array"""
          return np.array([skimage.color.rgb2gray(img) for img in X])
  class HogTransformer(BaseEstimator, TransformerMixin):
      """
      Expects an array of 2d arrays (1 channel images)
      Calculates hog features for each img
      """
      def __init__(self, y=None, orientations=9,
                  pixels_per_cell=(8, 8),
                  cells_per_block=(3, 3), block_norm='L2-Hys'):
          self.y = y
          self.orientations = orientations
          self.pixels_per_cell = pixels_per_cell
          self.cells_per_block = cells_per_block
          self.block_norm = block_norm
  
      def fit(self, X, y=None):
          return self
      def transform(self, X, y=None):
          def local_hog(X):
              return hog(X,
                        orientations=self.orientations,
                        pixels_per_cell=self.pixels_per_cell,
                        cells_per_block=self.cells_per_block,
                        block_norm=self.block_norm)
  
          try: # parallel
              return np.array([local_hog(img) for img in X])
          except:
              return np.array([local_hog(img) for img in X])
  # Transformer define end

  # Load trained transformers
  grayi = joblib.load('grayify.pkl')
  hogi = joblib.load('hogify.pkl')
  scali = joblib.load('scalify.pkl')

  # Transform input images
  input_gray = grayi.transform(allimg)
  input_hog = hogi.transform(input_gray)
  input_prepared = scali.transform(input_hog)

  # Load model
  clf = joblib.load('logistic_regression_model.pkl')

  # Predict
  result = clf.predict(input_prepared)
  nsfw_count = (result=='nsfw').sum()
  sfw_count = (result=='sfw').sum()

  print('')
  print('Trang web có tổng cộng: ' + str(len(result)) + ' hình ảnh')
  print('Số hình ảnh độc hại: ' + str(nsfw_count))
  print('Số hình ảnh an toàn: ' + str(sfw_count))
  print('')
  if nsfw_count > 0:
    print('Các hình ảnh độc hại: ')
    for i,_ in enumerate(result):
      if _ == 'nsfw':
        print(i, srcimg[i])
else:
  print('Không thể lấy ảnh từ trang web')