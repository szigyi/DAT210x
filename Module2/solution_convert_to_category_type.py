#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 14:46:43 2017

@author: szabolcs
"""
import pandas as pd

# Categorical ORDINAL feature

df = pd.DataFrame({
        "income": [
                "$1 - $1000",
                "$1000 - $10000",
                "$10000 - $50000",
                "$1 - $1000",
                "+$100000",
                "$50000 - $100000",
                "+$100000"
                ]
        })

income_ordered = ["$1 - $1000",
                  "$1000 - $10000",
                  "$10000 - $50000",
                  "$50000 - $100000",
                  "+$100000"]

print(df)

df.income = df.income.astype("category", ordered=True, categories=income_ordered).cat.codes

print(df)


# Categorical Nominal feature - increasing values

df = pd.DataFrame({
        "vertebrates": [
                "Bird",
                "Mammal",
                "Fish",
                "Reptile",
                "Mammal"
                ]
        })

print(df)
df.vertebrates = df.vertebrates.astype("category").cat.codes
print(df)
                                      
# Categorical Nominal feature - category feature - dummies

df = pd.DataFrame({
        "vertebrates": [
                "Bird",
                "Mammal",
                "Fish",
                "Reptile",
                "Mammal"
                ]
        })

print(df)
df = pd.get_dummies(df, columns=["vertebrates"])
print(df)







