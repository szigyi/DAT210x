import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

matplotlib.style.use('ggplot') # Look Pretty


def drawLine(model, X_train, y_train, X_test, y_test, title):
  # This convenience method will take care of plotting your
  # test observations, comparing them to the regression line,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X_train, y_train, c='r', marker='o', alpha=0.7)
  ax.scatter(X_test, y_test, c='g', marker='o', alpha=0.7)
  ax.plot(X_test, model.predict(X_test), color='orange', linewidth=1, alpha=0.7)
  ax.plot(X_train, model.predict(X_train), color='blue', linewidth=1, alpha=0.7)

  print("Est 2014 " + title + " Life Expectancy: ", model.predict([[2014]])[0])
  print("Est 2030 " + title + " Life Expectancy: ", model.predict([[2030]])[0])
  print("Est 2045 " + title + " Life Expectancy: ", model.predict([[2045]])[0])

  score = model.score(X_test, y_test)
  title += " R2: " + str(score)
  ax.set_title(title)


  plt.show()


#
# TODO: Load up the data here into a variable called 'X'.
# As usual, do a .describe and a print of your dataset and
# compare it to the dataset loaded in a text file or in a
# spread sheet application
#
file_path = "/Users/szabolcs/dev/git/DAT210x/Module5/Datasets/"
file_name = "life_expectancy.csv"

X = pd.read_csv(file_path + file_name, sep='\t')
print("Head", X.head())
print("Describe", X.describe())
print("DTypes", X.dtypes)
print("X", X.shape)

#
# TODO: Create your linear regression model here and store it in a
# variable called 'model'. Don't actually train or do anything else
# with it yet:
#
def split(X, column):
    #
    # TODO: Slice out your data manually (e.g. don't use train_test_split,
    # but actually do the Indexing yourself. Set X_train to be year values
    # LESS than 1986, and y_train to be corresponding WhiteMale age values.
    #
    # INFO You might also want to read the note about slicing on the bottom
    # of this document before proceeding.
    #
    X_train = X[X["Year"] < 1986]
    y_train = X_train[[column]].copy()
    X_train = X_train[["Year"]]
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    
    X_test = X[X["Year"] >= 1986]
    y_test = X_test[[column]].copy()
    X_test = X_test[["Year"]]
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)
    return X_train, X_test, y_train, y_test

def predict_life_expectancy_of(X_train, X_test, y_train, y_test, column):
    model = LinearRegression()
    #
    # TODO: Train your model then pass it into drawLine with your training
    # set and labels. You can title it "WhiteMale". drawLine will output
    # to the console a 2014 extrapolation / approximation for what it
    # believes the WhiteMale's life expectancy in the U.S. will be...
    # given the pre-1986 data you trained it with. It'll also produce a
    # 2030 and 2045 extrapolation.
    #
    model.fit(X_train, y_train)
    drawLine(model, X_train, y_train, X_test, y_test, column)
    #
    # TODO: Print the actual 2014 WhiteMale life expectancy from your
    # loaded dataset
    #
    print("Expected value:", X[X["Year"] == 2014][column].mean())
    return model

# 
# TODO: Repeat the process, but instead of for WhiteMale, this time
# select BlackFemale. Create a slice for BlackFemales, fit your
# model, and then call drawLine. Lastly, print out the actual 2014
# BlackFemale life expectancy
#
X_train, X_test, y_train, y_test = split(X, "WhiteMale")
model = predict_life_expectancy_of(X_train, X_test, y_train, y_test, "WhiteMale")

X_train, X_test, y_train, y_test = split(X, "BlackFemale")
model = predict_life_expectancy_of(X_train, X_test, y_train, y_test, "BlackFemale")


#
# TODO: Lastly, print out a correlation matrix for your entire
# dataset, and display a visualization of the correlation
# matrix, just as we described in the visualization section of
# the course
#
import seaborn as sns
corr = X.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)


plt.show()




#
# INFO + HINT On Fitting, Scoring, and Predicting:
#
# Here's a hint to help you complete the assignment without pulling
# your hair out! When you use .fit(), .score(), and .predict() on
# your model, SciKit-Learn expects your training data to be in
# spreadsheet (2D Array-Like) form. This means you can't simply
# pass in a 1D Array (slice) and get away with it.
#
# To properly prep your data, you have to pass in a 2D Numpy Array,
# or a dataframe. But what happens if you really only want to pass
# in a single feature?
#
# If you slice your dataframe using df[['ColumnName']] syntax, the
# result that comes back is actually a *dataframe*. Go ahead and do
# a type() on it to check it out. Since it's already a dataframe,
# you're good -- no further changes needed.
#
# But if you slice your dataframe using the df.ColumnName syntax,
# OR if you call df['ColumnName'], the result that comes back is
# actually a series (1D Array)! This will cause SKLearn to bug out.
# So if you are slicing using either of those two techniques, before
# sending your training or testing data to .fit / .score, do a
# my_column = my_column.reshape(-1,1). This will convert your 1D
# array of [n_samples], to a 2D array shaped like [n_samples, 1].
# A single feature, with many samples.
#
# If you did something like my_column = [my_column], that would produce
# an array in the shape of [1, n_samples], which is incorrect because
# SKLearn expects your data to be arranged as [n_samples, n_features].
# Keep in mind, all of the above only relates to your "X" or input
# data, and does not apply to your "y" or labels.

