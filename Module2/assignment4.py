import pandas as pd


# TODO: Load up the table, and extract the dataset
# out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
url = "http://www.espn.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2"

headers = ["RK", "PLAYER", "TEAM", "GP", "G", "A", "PTS", "RATIO", "PIM",
"PTSG", "SOG", "PCT", "GWG", "PPG", "PPA", "SHG", "SHA"]

df = pd.read_html(url)[0]


# TODO: Rename the columns so that they are similar to the
# column definitions provided to you on the website.
# Be careful and don't accidentially use any names twice.
#
df.columns = headers
#print(df.head())


# TODO: Get rid of any row that has at least 4 NANs in it,
# e.g. that do not contain player points statistics
#
df = df.dropna(axis=0, thresh=4)
df = df.drop(df.index[0], axis=0)
#print(df.head())


# TODO: At this point, look through your dataset by printing
# it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows
# EXCEPT those rows?
#
df = df[df["PCT"] != "PCT"]
#print(df)

# TODO: Get rid of the 'RK' column
#
df = df.drop(["RK"], axis=1)
#print(df)


# TODO: Ensure there are no holes in your index by resetting
# it. By the way, don't store the original index
#
df = df.reset_index()
#print(df)


# TODO: Check the data type of all columns, and ensure those
# that should be numeric are numeric
#
print(df.columns)
df.GP = df.GP.astype(int)
df.G = df.G.astype(int)
df.A = df.A.astype(int)
df.PTS = df.PTS.astype(int)
df.RATIO = df.RATIO.astype(int)
df.PIM = df.PIM.astype(int)
df.PTSG = df.PTSG.astype(float)
df.SOG = df.SOG.astype(int)
df.PCT = df.PCT.astype(float)
df.GWG = df.GWG.astype(int)
df.PPG = df.PPG.astype(int)
df.PPA = df.PPA.astype(int)
df.SHG = df.SHG.astype(int)
df.SHA = df.SHA.astype(int)
print(df.dtypes)



# TODO: Your dataframe is now ready! Use the appropriate 
# commands to answer the questions on the course lab page.
#
print(df.shape)
print(df.PCT.unique().shape)
print(df.iloc[15])
print(df.iloc[16])








