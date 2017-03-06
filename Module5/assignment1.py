import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


# Look Pretty
matplotlib.style.use('ggplot')
#plt.style.use('ggplot')


#
# TODO: To procure the dataset, follow these steps:
# 1. Navigate to: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
# 2. In the 'Primary Type' column, click on the 'Menu' button next to the info button,
#    and select 'Filter This Column'. It might take a second for the filter option to
#    show up, since it has to load the entire list first.
# 3. Scroll down to 'GAMBLING'
# 4. Click the light blue 'Export' button next to the 'Filter' button, and select 'Download As CSV'



def doKMeans(df):
  #
  # INFO: Plot your data with a '.' marker, with 0.3 alpha at the Longitude,
  # and Latitude locations in your dataset. Longitude = x, Latitude = y
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(df.Longitude, df.Latitude, marker='.', c='g', alpha=0.3)

  df = df[["Latitude", "Longitude"]]

  from sklearn.cluster import KMeans
  model = KMeans(n_clusters=7)
  model.fit(df)

  #
  # INFO: Print and plot the centroids...
  centroids = model.cluster_centers_
  ax.scatter(centroids[:,1], centroids[:,0], marker='x', c='red', alpha=0.5, linewidths=3, s=169)
  print(centroids)

file_path = "/Users/szabolcs/dev/git/DAT210x/Module5/Datasets/"
file_name = "gambling.csv"

df = pd.read_csv(file_path + file_name)

df = df.dropna(axis=0)
print(df.head(2))

df.Date = pd.to_datetime(df.Date, errors="coerce")
print(df.dtypes)

# INFO: Print & Plot your data
doKMeans(df)

df = df[df["Date"] > '2011-01-01']
# INFO: Print & Plot your data
doKMeans(df)
plt.show()


