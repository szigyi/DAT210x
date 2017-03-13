import pandas as pd


#https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.names


# 
# TODO: Load up the mushroom dataset into dataframe 'X'
# Verify you did it properly.
# Indices shouldn't be doubled.
# Header information is on the dataset's website at the UCI ML Repo
# Check NA Encoding
#
columns = ["class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
           "gill_attachment", "gill_spacing", "gill_size", "gill_color",
           "stalk_shape", "stalk_root",
           "stalk_surface_above_ring", "stalk_surface_below_ring", "stalk_color_above_ring",
           "stalk_color_below_ring", "veil_type", "veil_color",
           "ring_number", "ring_type",
           "spore_print_color", "population", "habitat"]

file = "/Users/szabolcs/dev/git/DAT210x/Module6/Datasets/agaricus-lepiota.data"
X = pd.read_csv(file)
X.columns = columns
print(X.head())

#print(X.isnull().sum())



# 
# TODO: Go ahead and drop any row with a nan
#
print("X", X.shape)
print("Dropping NaNs")
X = X[X["stalk_root"] != '?']
print("X", X.shape)


#
# TODO: Copy the labels out of the dset into variable 'y' then Remove
# them from X. Encode the labels, using the .map() trick we showed
# you in Module 5 -- canadian:0, kama:1, and rosa:2
#
print("Creating y")
y = X["class"].copy()
y = y.map({'p': 0, 'e':1})
y = y.values.reshape(-1, 1)
X = X.drop(["class"], axis=1)
print("X", X.shape)
print("y", y.shape)

#
# TODO: Encode the entire dataset using dummies
#
X = pd.get_dummies(X)

print("X", X.head())
# 
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)


#
# TODO: Create an DT classifier. No need to set any parameters
#
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
 
#
# TODO: train the classifier on the training data / labels:
# TODO: score the classifier on the testing data / labels:
#
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("High-Dimensionality Score: ", round((score*100), 3))
print("feature_importances_", clf.feature_importances_)

#
# TODO: Use the code on the course's SciKit-Learn page to output a .DOT file
# Then render the .DOT to .PNGs. Ensure you have graphviz installed.
# If not, `brew install graphviz`. If you can't, use: http://webgraphviz.com/.
# On Windows 10, graphviz installs via a msi installer that you can download from
# the graphviz website. Also, a graph editor, gvedit.exe can be used to view the
# tree directly from the exported tree.dot file without having to issue a call.
#
import sklearn.tree as tree
tree.export_graphviz(clf.tree_, out_file='tree.dot', feature_names=X.columns)

from subprocess import call
call(['dot', '-T', 'png', 'tree.dot', '-o', 'tree.png'])


