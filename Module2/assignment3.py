import pandas as pd

# TODO: Load up the dataset
# Ensuring you set the appropriate header column names
#
file_path = "/Users/szabolcs/dev/git/DAT210x/Module2/Datasets/"
file_name = "servo.data"
df = pd.read_csv(file_path + file_name, names=['motor', 'screw', 'pgain', 'vgain', 'class'])

print(df.shape)
print(df.head(2))


# TODO: Create a slice that contains all entries
# having a vgain equal to 5. Then print the 
# length of (# of samples in) that slice:
#
print(df[df["vgain"] == 5].shape)


# TODO: Create a slice that contains all entries
# having a motor equal to E and screw equal
# to E. Then print the length of (# of
# samples in) that slice:
#
print(df[(df["motor"] == "E") & (df["screw"] == "E")].shape)



# TODO: Create a slice that contains all entries
# having a pgain equal to 4. Use one of the
# various methods of finding the mean vgain
# value for the samples in that slice. Once
# you've found it, print it:
#
pg = df[df["pgain"] == 4]
print(pg)
print(pg.vgain.mean())



# TODO: (Bonus) See what happens when you run
# the .dtypes method on your dataframe!

print(df.dtypes)

