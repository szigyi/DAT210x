import pandas as pd
import numpy as np


file_path = "/Users/szabolcs/dev/git/DAT210x/Module2/Datasets/"
file_name = "census.data"

#
# TODO:
# Load up the dataset, setting correct header labels.
#
df = pd.read_csv(file_path + file_name)

df.columns = ['order', 'education', 'age', 'capital-gain', 'race', 'capital-loss',
              'hours-per-week', 'sex', 'classification']
df = df.drop('order', axis=1)
#print(df.head())


#
# TODO:
# Use basic pandas commands to look through the dataset... get a
# feel for it before proceeding! Do the data-types of each column
# reflect the values you see when you look through the data using
# a text editor / spread sheet program? If you see 'object' where
# you expect to see 'int32' / 'float64', that is a good indicator
# that there is probably a string or missing value in a column.
# use `your_data_frame['your_column'].unique()` to see the unique
# values of each column and identify the rogue values. If these
# should be represented as nans, you can convert them using
# na_values when loading the dataframe.
#

m = df[df['capital-gain'] != '?'].reset_index()
capital_gain_mean = m.mean(axis=0)[2]
df['capital-gain'] = [capital_gain_mean if x == '?' else x for x in df['capital-gain']]
df['capital-gain'] = df['capital-gain'].astype(int)
#print(df.dtypes)


#
# TODO:
# Look through your data and identify any potential categorical
# features. Ensure you properly encode any ordinal and nominal
# types using the methods discussed in the chapter.
#
# Be careful! Some features can be represented as either categorical
# or continuous (numerical). If you ever get confused, think to yourself
# what makes more sense generally---to represent such features with a
# continuous numeric type... or a series of categories?
#
education_order = ['Preschool',
                   '1st-4th',
                   '5th-6th',
                   '7th-8th',
                   '9th',
                   '10th',
                   '11th',
                   '12th',
                   'Some-college',
                   'HS-grad',
                   'Bachelors',
                   'Masters',
                   'Doctorate']

df.education = df.education.astype('category', ordered=True, categories=education_order).cat.codes

classification_order = ['<=50K', '>50K']
df.classification = df.classification.astype('category', ordered=True, categories=classification_order).cat.codes

df = pd.get_dummies(df, columns=['race'])
df = pd.get_dummies(df, columns=['sex'])

print(df.dtypes)


#
# TODO:
# Print out your dataframe
#
#print(df)


