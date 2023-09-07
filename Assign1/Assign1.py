import pandas
#import numpy as np 
import matplotlib.pyplot as plt
#import seaborn
#import scipy as sp

df1=pandas.read_csv("gdp-per-capita-worldbank.csv")
df1_mask = (df1['Year'] > 2000) & (df1['Year'] <= 2021)
df1 = df1.loc[df1_mask]


gdp=df1["GDP per capita, PPP (constant 2017 international $)"]
df2=pandas.read_csv("life-expectancy-at-age-15.csv")
df2_mask = (df2['Year'] > 2000) & (df2['Year'] <= 2021)
df2 = df2.loc[df2_mask]
exp=df2["Life expectancy at 15"]
print(df1)
print(df2)

'''l = []
for element in df1[Year]:
    if element >= 2002:
        l.append(element)'''

plt.scatter(gdp,exp)
plt.title("Life Expectancy vs GDP per capital")
plt.show()

