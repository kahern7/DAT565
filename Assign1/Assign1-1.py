import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the two CSV files
df_life_expectancy = pd.read_csv("life-expectancy-at-age-15.csv")
df_gdp_per_capita = pd.read_csv("gdp-per-capita-worldbank.csv")

# Filter the data to match years
start_year = 2002
end_year = 2021
df_life_expectancy_filtered = df_life_expectancy[(df_life_expectancy['Year'] >= start_year) & (df_life_expectancy['Year'] <= end_year)]
df_gdp_per_capita_filtered = df_gdp_per_capita[(df_gdp_per_capita['Year'] >= start_year) & (df_gdp_per_capita['Year'] <= end_year)]

# Merge the filtered data based on 'Entity' and 'Year'
merged_data = pd.merge(df_life_expectancy_filtered, df_gdp_per_capita_filtered, on=['Entity', 'Year'])

# Extract the columns for the scatter plot
gdp_values = merged_data['GDP per capita, PPP (constant 2017 international $)']
life_expectancy_values = merged_data['Life expectancy at 15']

# Create a scatter plot
plt.scatter(gdp_values,life_expectancy_values)
plt.title("Life Expectancy vs GDP per capital")
plt.show()