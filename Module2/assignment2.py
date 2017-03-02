import pandas as pd

# TODO: Load up the 'tutorial.csv' dataset
#
file_path = "/Users/szabolcs/dev/git/DAT210x/Module2/Datasets/"
file_name = "tutorial.csv"
df = pd.read_csv(file_path + file_name)

print(df)


# TODO: Print the results of the .describe() method
#
print(df.describe())


# TODO: Figure out which indexing method you need to
# use in order to index your dataframe with: [2:4,'col3']
# And print the results
#
sl = df.loc[2:4, "col3"]
print(sl)
