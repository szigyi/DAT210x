import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


# Do * NOT * alter this line, until instructed!
scaleFeatures = True

#
file_path = "/Users/szabolcs/dev/git/DAT210x/Module4/Datasets/"
file_name = "kidney_disease.csv"

exclude_columns = ['id', 'classification'] #, 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

df = pd.read_csv(file_path + file_name)
labels = ['red' if i=='ckd' else 'green' for i in df.classification]
df.drop(exclude_columns, axis=1, inplace=True)
print(df.head())

df = pd.get_dummies(df, columns=["rbc"])
df = pd.get_dummies(df, columns=["pc"])
df = pd.get_dummies(df, columns=["pcc"])
df = pd.get_dummies(df, columns=["ba"])
df = pd.get_dummies(df, columns=["htn"])
df = pd.get_dummies(df, columns=["dm"])
df = pd.get_dummies(df, columns=["cad"])
df = pd.get_dummies(df, columns=["appet"])
df = pd.get_dummies(df, columns=["pe"])
df = pd.get_dummies(df, columns=["ane"])

df.pcv = pd.to_numeric(df.pcv, errors="coerce")
df.wc = pd.to_numeric(df.wc, errors="coerce")
df.rc = pd.to_numeric(df.rc, errors="coerce")
df = df.dropna(axis=0)

print(df.dtypes)
print(df.describe())

if scaleFeatures: df = helper.scaleFeatures(df)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(df)
T = pca.transform(df)

# Plot the transformed data as a scatter plot. Recall that transforming
# the data will result in a NumPy NDArray. You can either use MatPlotLib
# to graph it directly, or you can convert it to DataFrame and have pandas
# do it for you.
#
# Since we've already demonstrated how to plot directly with MatPlotLib in
# Module4/assignment1.py, this time we'll convert to a Pandas Dataframe.
#
# Since we transformed via PCA, we no longer have column names. We know we
# are in P.C. space, so we'll just define the coordinates accordingly:
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()


