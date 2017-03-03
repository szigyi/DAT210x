import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Look pretty...
# matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
file_path = "/Users/szabolcs/dev/git/DAT210x/Module3/Datasets/"
file_name = "wheat.data"

df = pd.read_csv(file_path + file_name)
print(df.shape)
print(df.head(3))



fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the area,
# perimeter and asymmetry features. Be sure to use the
# optional display parameter c='red', and also label your
# axes
# 
ax.set_xlabel("Area")
ax.set_ylabel("Perimeter")
ax.set_zlabel("Asymmetry")
ax.scatter(df.area, df.perimeter, df.asymmetry, c="r", marker=".")


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
#
# TODO: Create a new 3D subplot using fig. Then use the
# subplot to graph a 3D scatter plot using the width,
# groove and length features. Be sure to use the
# optional display parameter c='green', and also label your
# axes
# 
ax.set_xlabel("Width")
ax.set_ylabel("Groove")
ax.set_zlabel("Length")
ax.scatter(df.width, df.groove, df.length, c="g", marker=".")


plt.show()


