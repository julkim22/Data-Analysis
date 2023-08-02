# -*- coding: utf-8 -*-
"""(Iuliia Kim) Final ISP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14ngvhY8OADTefJXa0yKhyTWqKHr-uk4Q
"""

import pandas as pd #library for data manipulation and analysis = analysis
import statsmodels.api as sm #library to perform panel data analysis  = table
import seaborn as sns  #library to visualization statistic analysis' results = visual
import matplotlib.pyplot as plt #ploting library = build a plot

# Read the dataset
database1 = pd.read_stata("/content/dataset_14.dta")
database2 = pd.read_stata("/content/dataset_15.dta")
database3 = pd.read_stata("/content/dataset_16.dta")

# Rename variables
dataset_14 = dataset_14.rename(columns={
    "KCh14arx002": "allergy",
    "KCh14adx003": "derma",
    "KCh14asx008": "asthma",
    "HIn14acs015": "academ"
})
dataset_15 = dataset_15.rename(columns={
    "KCh15arx001": "allergy",
    "KCh15adx001": "derma",
    "KCh15asx001": "asthma",
    "LCh15acs030": "academ"
})
dataset_16 = dataset_16.rename(columns={
    "KCh16arx001": "allergy",
    "KCh16adx001": "derma",
    "KCh16asx001": "asthma",
    "LCh16acs030": "academ"
})

# Add new variable "wave"
dataset_14["wave"] = 2014
dataset_15["wave"] = 2015
dataset_16["wave"] = 2016

# Select relevant variables for analysis
dataset_14 = dataset_14[["N_ID", "wave", "allergy", "derma", "asthma", "academ"]]
dataset_15 = dataset_15[["N_ID", "wave", "allergy", "derma", "asthma", "academ"]]
dataset_16 = dataset_16[["N_ID", "wave", "allergy", "derma", "asthma", "academ"]]

# Merge datasets
dataset_merge = pd.concat([dataset_14, dataset_15])
merged_database = pd.concat([dataset_merge, dataset_16])

# Set N_ID and wave as the panel index
merged_database = merged_database.set_index(["N_ID", "wave"])

print(merged_database) #data includes not numerical variables!

merged_database['academ'].unique() #take each not numerical to convert str-->int

#convert the values to numeric 
mapping = {'예': 1, '아니오': 0,  '상위 20% 이내': 5, '상위 21%~40%':4,'중간 50% 내외':3,'하위 21%-40%':2,'하위 20% 이내':1}  

merged_database['allergy'] = merged_database['allergy'].replace(mapping)
merged_database['derma'] = merged_database['derma'].replace(mapping)
merged_database['asthma'] = merged_database['asthma'].replace(mapping)
merged_database['academ'] = merged_database['academ'].replace(mapping)

#converts data to numeric values errors=coerce replaces non-numeric values with NaN
merged_database['allergy'] = pd.to_numeric(merged_database['allergy'], errors='coerce')
merged_database['derma'] = pd.to_numeric(merged_database['derma'], errors='coerce')
merged_database['asthma'] = pd.to_numeric(merged_database['asthma'], errors='coerce')
merged_database['academ'] = pd.to_numeric(merged_database['academ'], errors='coerce')

#Drop any rows with missing values:
dataset_merge2 = merged_database.dropna()

#check results 
print(dataset_merge2)

#error in data format --> change it to short ver.
dataset_merge2['wave'].dt.strftime('%Y')

# Perform fixed effects regression
model = sm.OLS(dataset_merge2["academ"], sm.add_constant(dataset_merge2[["derma", "allergy", "asthma"]]))
result = model.fit()
print(result.summary())

#calculate the correlation matrix
correlation = dataset_merge2[["academ", "asthma", "allergy", "derma"]].corr()

# heatmap visualization
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

#grouping data based on index levels
dataset_merge2.set_index(["N_ID", "wave"], inplace=True)

#represent year for each independent and depend variable
dataset_merge2['wave'] = dataset_merge2['wave'].dt.strftime('%Y')

#how change variables by each year
 #set the plot style to darkgrid
sns.set_style("darkgrid") 

# Plot for asthma
g1 = sns.lmplot(data=dataset_merge2, x='asthma', y='academ', col='wave', col_wrap=3, scatter_kws={"color": "blue"})
g1.set_axis_labels('Asthma', 'Academical achievements')

# Plot for allergy
g2 = sns.lmplot(data=dataset_merge2, x='allergy', y='academ', col='wave', col_wrap=3, scatter_kws={"color": "green"})
g2.set_axis_labels('Allergic Rhinits', 'Academical achievements')

# Plot for dermatitis
g3 = sns.lmplot(data=dataset_merge2, x='derma', y='academ', col='wave', col_wrap=3, scatter_kws={"color": "red"})
g3.set_axis_labels('Dermatitis', 'Academical achievements')

plt.suptitle('Impact of Chronical Disease on Academ by Year')
plt.tight_layout()
plt.show()

