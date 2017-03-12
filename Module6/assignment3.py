#
# This code is intentionally missing!
# Read the directions on the course lab page!
#
import pandas as pd
import numpy as np


file = "/Users/szabolcs/dev/git/DAT210x/Module6/Datasets/parkinsons.data"

X = pd.read_csv(file)
y = X[["status"]].values.reshape(-1, 1)
X = X.drop(["name", "status"], axis=1)
print(X.head())
print("X", X.shape)
print("y", y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

from sklearn.preprocessing import StandardScaler
#norm = KernelCenterer()
#best_score 0.915254237288
#C 1.7
#gamma 0.006

norm = StandardScaler()
#best_score 0.932203389831
#C 1.55
#gamma 0.097
norm.fit(X_train)
X_train = norm.transform(X_train)
X_test = norm.transform(X_test)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(score)

best_score = 0

from sklearn.manifold import Isomap
for n_neighbors in range(2, 6):
    for n_components in range(4, 7):
        pca = Isomap(n_neighbors=n_neighbors, n_components=n_components)
        pca.fit(X_train)
        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)
        
        for C in np.arange(0.05, 2, 0.05):
            for gamma in np.arange(0.001, 0.1, 0.001):
                model = SVC(C=C, gamma=gamma)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    print("best_score", best_score)
                    print("C", C)
                    print("gamma", gamma)
                    print("n_neighbors", n_neighbors)
                    print("n_components", n_components)
                
                
                
                
                
                
                
                
                
                
                
                