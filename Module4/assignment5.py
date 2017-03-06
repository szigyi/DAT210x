import math
import pandas as pd
import numpy as np

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')

def Plot2D(T, title, x, y):
  # This method picks a bunch of random samples (images in your case)
  # to plot onto the chart:
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  ax.scatter(T[:,x],T[:,y], marker='.',alpha=0.7)
  

from os import listdir

file_path = "/Users/szabolcs/dev/git/DAT210x/Module4/Datasets/ALOI/32/"

#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
file_names = listdir(file_path)
samples = []

#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
for file_name in file_names:
    pic = misc.imread(file_path + file_name)
    ser = [item for sublist in pic for item in sublist]
    pic = pd.Series(ser)
    #pic = pic[::2, ::2]
    pic = pic.values.reshape(-1, 3)
    samples.append(pic)
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
df = pd.DataFrame.from_records(samples)
print(df.shape)
num_images, num_pixels = df.shape
num_pixels = int(math.sqrt(num_pixels))
for i in range(num_images):
  df.loc[i,:] = df.loc[i,:].values.reshape(num_pixels, num_pixels).T.reshape(-1)

print(df.shape)

#df.iloc[0] = pd.to_numeric(df.iloc[0], errors="coerce")
#print(df.dtypes)

#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 


#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 



#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
from sklearn.manifold import Isomap
imap = Isomap(n_components=2, n_neighbors=6)
imap.fit(df)
df_imap = imap.transform(df)

Plot2D(df_imap, "Isomap", 0, 1)

#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 




#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 



plt.show()

