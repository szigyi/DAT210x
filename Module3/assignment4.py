import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import parallel_coordinates

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
# TODO: Drop the 'id' feature, if you included it as a feature
# (Hint: You shouldn't have)
# Also get rid of the 'area' and 'perimeter' features
# 
df.drop(["id", "area", "perimeter"], axis=1, inplace=True)



#
# TODO: Plot a parallel coordinates chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
# 
parallel_coordinates(df, "wheat_type", alpha=0.4)



plt.show()


