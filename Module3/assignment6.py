import pandas as pd
import matplotlib.pyplot as plt


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
file_path = "/Users/szabolcs/dev/git/DAT210x/Module3/Datasets/"
file_name = "wheat.data"

df = pd.read_csv(file_path + file_name)
print(df.shape)
print(df.head(3))


#
# TODO: Drop the 'id' feature, if you included it as a feature
# (Hint: You shouldn't have)
# 
df.drop(["id"], axis=1, inplace=True)


#
# TODO: Compute the correlation matrix of your dataframe
# 
correlation_matrix = df.corr()


#
# TODO: Graph the correlation matrix using imshow or matshow
# 
plt.imshow(correlation_matrix, cmap=plt.cm.Blues, interpolation="nearest")
plt.colorbar()
tick_marks = [i for i in range(len(df.columns))]
plt.xticks(tick_marks, df.columns, rotation="vertical")
plt.yticks(tick_marks, df.columns)
plt.show()


plt.show()


