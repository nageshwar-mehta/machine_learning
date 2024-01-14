import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df=pd.read_csv("canada_per_capita_income.csv")
print(df.head())
# %matplotlib inline
plt.title("canada's per capita income")
plt.xlabel("year")
plt.ylabel("Income")

plt.scatter(df.year,df.income,color="red",marker="+")
plt.plot(df.year,df.income,color="grey")
plt.show()
reg = linear_model.LinearRegression()
reg.fit(df[['income']],df.year)
reg.predict(2000)