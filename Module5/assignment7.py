import pandas as pd

# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = True
Test_K = True
Test_weights = True

def plotDecisionBoundary(model, X, y):
  print("Plotting...")
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()


# 
# TODO: Load in the dataset, identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.
#
file_path = "/Users/szabolcs/dev/git/DAT210x/Module5/Datasets/"
file_name = "breast-cancer-wisconsin.data"

columns = ['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']

X = pd.read_csv(file_path + file_name)
X.columns = columns

print("X", X.shape)
X = X[X.nuclei != '?']
print("X", X.shape)
X.nuclei = X.nuclei.astype(int)

print("IsNull", X.isnull().sum())

print(X.head())
print("X", X.shape)
print(X.dtypes)

# 
# TODO: Copy out the status column into a slice, then drop it from the main
# dataframe. Always verify you properly executed the drop by double checking
# (printing out the resulting operating)! Many people forget to set the right
# axis here.
#
# If you goofed up on loading the dataset and notice you have a `sample` column,
# this would be a good place to drop that too if you haven't already.
#

y = X["status"].copy()
X = X.drop(["status"], axis=1)

X = X.drop(["sample"], axis=1)
print("X", X.shape)
print("y", y.shape)



#
# TODO: With the labels safely extracted from the dataset, replace any nan values
# with the mean feature / column value
#
# .. your code here ..



#
# TODO: Do train_test_split. Use the same variable names as on the EdX platform in
# the reading material, but set the random_state=7 for reproduceability, and keep
# the test_size at 0.5 (50%).
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7, test_size=0.5)


#
# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation. Recall: when you do pre-processing,
# which portion of the dataset is your model trained upon? Also which portion(s)
# of your dataset actually get transformed?
#
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import RobustScaler
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = StandardScaler()
#scaler = Normalizer('l2')
#scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#
# PCA and Isomap are your new best friends
model = None
if Test_PCA:
  print("Computing 2D Principle Components")
  #
  # TODO: Implement PCA here. Save your model into the variable 'model'.
  # You should reduce down to two dimensions.
  #
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  pca.fit(X_train)
  X_train = pca.transform(X_train)
  X_test = pca.transform(X_test)

else:
  print("Computing 2D Isomap Manifold")
  #
  # TODO: Implement Isomap here. Save your model into the variable 'model'
  # Experiment with K values from 5-10.
  # You should reduce down to two dimensions.
  #
  from sklearn.manifold import Isomap
  isomap = Isomap(n_neighbors=5, n_components=2)
  isomap.fit(X_train)
  X_train = isomap.transform(X_train)
  X_test = isomap.transform(X_test)


#
# TODO: Train your model against data_train, then transform both
# data_train and data_test using your model. You can save the results right
# back into the variables themselves.
#
from sklearn.neighbors import KNeighborsClassifier
uni_scores = []
dist_scores = []
if Test_weights:
    weights = ['uniform', 'distance']
    for weight in weights:
        print(weight)
        scores = []
        if Test_K:
            for k in range(1, 16):
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                #train_score = knn.score(X_train, y_train)
                test_score = knn.score(X_test, y_test)
                print(test_score)
                scores.append(test_score)
        else:
            knn = KNeighborsClassifier(n_neighbors=4)
            knn.fit(X_train, y_train)
            #train_score = knn.score(X_train, y_train)
            test_score = knn.score(X_test, y_test)
            scores.append(test_score)
            
        if weight == 'uniform':
            uni_scores = scores
        else:
            dist_scores = scores
# 
# TODO: Implement and train KNeighborsClassifier on your projected 2D
# training data here. You can use any K value from 1 - 15, so play around
# with it and see what results you can come up. Your goal is to find a
# good balance where you aren't too specific (low-K), nor are you too
# general (high-K). You should also experiment with how changing the weights
# parameter affects the results.
#
# .. your code here ..



#
# INFO: Be sure to always keep the domain of the problem in mind! It's
# WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing
# it to be benign, and then having the patient progress in cancer. Since the UDF
# weights don't give you any class information, the only way to introduce this
# data into SKLearn's KNN Classifier is by "baking" it into your data. For
# example, randomly reducing the ratio of benign samples compared to malignant
# samples from the training set.



#
# TODO: Calculate + Print the accuracy of the testing set
#
if Test_K or Test_weights:
    scores_vs_k = pd.DataFrame({'uni_score': uni_scores, 'dist_score': dist_scores, 'k': range(1, 16)})
    scores_vs_k.plot(x='k', y=['uni_score', 'dist_score'])
    print(scores_vs_k)
    
# Adjusted R^2 and R^2
import statsmodels.api as sm
X1 = sm.add_constant(X)
result = sm.OLS(y, X1).fit()
print("R^2:", result.rsquared)
print("Adjusted R^2:", result.rsquared_adj)


plotDecisionBoundary(knn, X_test, y_test)




