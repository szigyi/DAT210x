import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

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

#
# TODO: Create a 2d scatter plot that graphs the
# area and perimeter features
# 
plt.figure()
df.plot.scatter(x="area", y="perimeter")


#
# TODO: Create a 2d scatter plot that graphs the
# groove and asymmetry features
# 
df.plot.scatter(x="groove", y="asymmetry")


#
# TODO: Create a 2d scatter plot that graphs the
# compactness and width features
# 
df.plot.scatter(x="compactness", y="width")



# BONUS TODO:
# After completing the above, go ahead and run your program
# Check out the results, and see what happens when you add
# in the optional display parameter marker with values of
# either '^', '.', or 'o'.


plt.show()


