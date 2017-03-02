#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 11:58:55 2017

@author: szabolcs
"""
import pandas as pd

file_path = "/Users/szabolcs/dev/git/DAT210x/Module2/Datasets/"
file_name = "direct_marketing.csv"

df = pd.read_csv(file_path + file_name)
df.shape
df.head(3)

df.recency
df["recency"]
df[["recency"]]
df.loc[:, "recency"]
df.loc[:, ["recency"]]
df.iloc[:, 0]
df.iloc[:, [0]]
df.ix[:, 0]

# Columns
# First column
df.iloc[:, 0]

# First 5 columns
df.iloc[:, :5]

# Last column
df.iloc[:, -1]

# Last 5 columns
df.iloc[:, -5:]

# Middle columns
df.iloc[:, 5:9]

# Rows
# First row
df.iloc[0,:]
df.iloc[0]

# First 5 rows
df.iloc[:5, :]
df.iloc[:5]

# Last row
df.iloc[-1, :]
df.iloc[-1]

# Last 5 rows
df.iloc[-5:, :]
df.iloc[-5:]

# Middle rows
df.iloc[5:9, :]
df.iloc[5:9]

# Value filtering - Boolean Indexing
# Show history_segment where mens's value 1
# df["mens"] == 1
df[df["mens"] == 1][["history_segment", "mens"]]
df[df["womens"] == 1][["history_segment", "womens"]]

df[(df["recency"] < 7) & (df["newbie"] == 0)]


# this solution preserves the umatching data place as NaN
df.where(df["mens"] == 1)[["history_segment", "mens"]]

