#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df_dict = pd.read_excel('C:/Users/Hp/Downloads/cdc-diabetes-2018 (2).xlsx', sheet_name=None)
obesity_data = df_dict['Obesity']
inactivity_data = df_dict['Inactivity']
diabetes_data = df_dict['Diabetes']
# Merge the DataFrames
merged_data = pd.merge(obesity_data, inactivity_data, on='COUNTY')
merged_data = pd.merge(merged_data, diabetes_data, on='COUNTY')

import pandas as pd

# Extract relevant columns for analysis
diabetes_data = merged_data[['STATE_x', '% DIABETIC']]

# Group by state and calculate the mean percentage of diabetic problems
state_diabetes_mean = diabetes_data.groupby('STATE_x')['% DIABETIC'].mean()

# Find the state with the highest mean percentage of diabetic problems
highest_diabetes_state = state_diabetes_mean.idxmax()
highest_diabetes_percentage = state_diabetes_mean.max()

print(f"The state with the highest percentage of diabetic problems is {highest_diabetes_state} with {highest_diabetes_percentage}%.")
import pandas as pd
health_data = merged_data[['STATE_x', '% OBESE', '% INACTIVE', '% DIABETIC']]

# Create a health score based on the inverse of obesity, inactivity, and diabetes percentages
health_data['Health Score'] = 100 - health_data['% OBESE'] - health_data['% INACTIVE'] - health_data['% DIABETIC']

# Group by state and calculate the mean health score
state_health_mean = health_data.groupby('STATE_x')['Health Score'].mean()

# Find the state with the highest mean health score
healthiest_state = state_health_mean.idxmax()
highest_health_score = state_health_mean.max()

print(f"The state with the healthiest population is {healthiest_state} with a health score of {highest_health_score}.")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract relevant columns for analysis
bias_data = merged_data[['STATE_x', '% OBESE', '% INACTIVE', '% DIABETIC']]

# 1. Descriptive Statistics
print("Descriptive Statistics:")
print(bias_data.groupby('STATE_x').describe())
# 2. Distribution Comparison
print("\nDistribution Comparison:")
for column in ['% OBESE', '% INACTIVE', '% DIABETIC']:
    sns.histplot(data=bias_data, x=column, hue='STATE_x', multiple="stack")
    plt.title(f'Distribution of {column} across states')
    plt.show()
# 3. Visualization
print("\nVisualization:")
sns.pairplot(bias_data, hue='STATE_x', height=2.5)
plt.suptitle('Pairplot of key variables')
plt.show()
# 4. Correlation Analysis
print("\nCorrelation Analysis:")
correlation_matrix = bias_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
# 5. Representativeness
print("\nRepresentativeness:")
state_counts = bias_data['STATE_x'].value_counts()
state_counts.plot(kind='bar', color='skyblue')
plt.title('Counts of samples by state')
plt.show()

# 2. Distribution Comparison
print("\nDistribution Comparison:")
for column in ['% OBESE', '% INACTIVE', '% DIABETIC']:
    sns.histplot(data=bias_data, x=column, hue='STATE_x', multiple="stack")
    plt.title(f'Distribution of {column} across states')
    plt.show()

# 3. Visualization
print("\nVisualization:")
sns.pairplot(bias_data, hue='STATE_x', height=2.5)
plt.suptitle('Pairplot of key variables')
plt.show()

# 4. Correlation Analysis
print("\nCorrelation Analysis:")
correlation_matrix = bias_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 5. Representativeness
print("\nRepresentativeness:")
state_counts = bias_data['STATE_x'].value_counts()
state_counts.plot(kind='bar', color='skyblue')
plt.title('Counts of samples by state')
plt.show()

import pandas as pd

# Assuming your merged dataset is named 'merged_data'
# Replace 'your_column_names' with the actual column names in your dataset

# Find the state with the highest obesity rate
highest_obesity_state = merged_data.groupby('STATE_x')['% OBESE'].mean().idxmax()

# Find the state with the lowest obesity rate
lowest_obesity_state = merged_data.groupby('STATE_x')['% OBESE'].mean().idxmin()

# Find the state with the highest inactivity rate
highest_inactivity_state = merged_data.groupby('STATE_x')['% INACTIVE'].mean().idxmax()

# Find the state with the lowest inactivity rate
lowest_inactivity_state = merged_data.groupby('STATE_x')['% INACTIVE'].mean().idxmin()

# Find the state with the highest diabetes rate
highest_diabetes_state = merged_data.groupby('STATE_x')['% DIABETIC'].mean().idxmax()

# Find the state with the lowest diabetes rate
lowest_diabetes_state = merged_data.groupby('STATE_x')['% DIABETIC'].mean().idxmin()

print("State with the highest obesity rate:", highest_obesity_state)
print("State with the lowest obesity rate:", lowest_obesity_state)
print("State with the highest inactivity rate:", highest_inactivity_state)
print("State with the lowest inactivity rate:", lowest_inactivity_state)
print("State with the highest diabetes rate:", highest_diabetes_state)
print("State with the lowest diabetes rate:", lowest_diabetes_state)


# Check for missing data in each column for each state
missing_data = merged_data.groupby('STATE_x').apply(lambda x: x.isnull().sum())

# Display the results
print(missing_data)

# Scatter plot
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt

sns.scatterplot(x='% OBESE', y='% DIABETIC', data=merged_data,color='green')
plt.title('Linear Regression: % Diabetic vs. % Obese')

# Linear regression
X = merged_data['% OBESE'].values.reshape(-1, 1)
y = merged_data['% DIABETIC'].values
regressor = LinearRegression()
regressor.fit(X, y)

# Plotting the regression line
plt.plot(X, regressor.predict(X), color='red')
plt.show()

# Coefficients
print('Slope:', regressor.coef_[0])
print('Intercept:', regressor.intercept_)

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Scatter plot with green bubbles
sns.scatterplot(x='% OBESE', y='% DIABETIC', data=merged_data, color='green')

# Linear regression
X = merged_data['% OBESE'].values.reshape(-1, 1)
y = merged_data['% DIABETIC'].values
regressor = LinearRegression()

# Perform 10-fold cross-validation
cv_scores = cross_val_score(regressor, X, y, cv=10)
mean_cv_score = np.mean(cv_scores)

# Plotting the regression line
regressor.fit(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.title('Linear Regression: % Diabetic vs. % Obese')

# Coefficients
slope = regressor.coef_[0]
intercept = regressor.intercept_

print('Slope:', slope)
print('Intercept:', intercept)
print('Mean 10-fold CV Score:', mean_cv_score)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
X = merged_data['% OBESE'].values.reshape(-1, 1)
y = (merged_data['% DIABETIC'] > merged_data['% DIABETIC'].median()).astype(int)  # Binary label

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred = log_reg.predict(X_test)

# Model evaluation
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))


# PCA
X = merged_data[['% OBESE', '% INACTIVE']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA')
plt.show()


# Clustering
X = merged_data[['% OBESE', '% INACTIVE']]
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters)
merged_data['Cluster'] = kmeans.fit_predict(X)

# Visualize
sns.scatterplot(x='% OBESE', y='% INACTIVE', hue='Cluster', data=merged_data)
plt.title('Clustering: % Obese vs. % Inactive')
plt.show()

# Linear Regression for % Obese and % Inactive
X_obese_inactive = merged_data[['% OBESE']].values
y_obese_inactive = merged_data['% INACTIVE'].values

# Perform linear regression
regressor_obese_inactive = LinearRegression()
regressor_obese_inactive.fit(X_obese_inactive, y_obese_inactive)

# Visualize the linear regression line
plt.scatter(X_obese_inactive, y_obese_inactive)
plt.plot(X_obese_inactive, regressor_obese_inactive.predict(X_obese_inactive), color='red')
plt.xlabel('% Obese')
plt.ylabel('% Inactive')
plt.title('Linear Regression: % Inactive vs. % Obese')
plt.show()

# Print the coefficients
print('Slope:', regressor_obese_inactive.coef_[0])
print('Intercept:', regressor_obese_inactive.intercept_)

# Logistic Regression for % Obese and % Inactive
# Assuming you have merged_data, X_train, X_test, y_train, y_test available

# Logistic regression
log_reg_obese_inactive = LogisticRegression()
log_reg_obese_inactive.fit(X_train, y_train)

# Predictions
y_pred_obese_inactive = log_reg_obese_inactive.predict(X_test)

# Model evaluation
accuracy_logistic_obese_inactive = accuracy_score(y_test, y_pred_obese_inactive)
print('Accuracy (Logistic Regression for % Obese and % Inactive):', accuracy_logistic_obese_inactive)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_obese_inactive))

# PCA for % Obese and % Inactive
X_obese_inactive_pca = merged_data[['% OBESE', '% INACTIVE']]
scaler_obese_inactive = StandardScaler()
X_obese_inactive_scaled = scaler_obese_inactive.fit_transform(X_obese_inactive_pca)

pca_obese_inactive = PCA(n_components=2)
X_obese_inactive_pca_result = pca_obese_inactive.fit_transform(X_obese_inactive_scaled)

plt.scatter(X_obese_inactive_pca_result[:, 0], X_obese_inactive_pca_result[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: % Obese vs. % Inactive')
plt.show()

X = sm.add_constant(X) 
 
X_train, 	X_test, 	y_train, 	y_test 	= 	train_test_split(X, 	y, 	test_size=0.2, random_state=42) 
 
model = sm.OLS(y_train, X_train).fit() 
 
y_pred = model.predict(X_test) 
 
r_squared = r2_score(y_test, y_pred) print('R-squared:', r_squared) 
 
print(model.summary()) 
 
plt.figure(figsize=(10, 6)) 
plt.scatter(X_test['% INACTIVE'], y_test, color='blue', label='Actual % OBESE') plt.plot(X_test['% INACTIVE'], y_pred, color='red', linewidth=2, label='Regression Line') 
plt.xlabel('% INACTIVE') plt.ylabel('% OBESE') plt.legend() 
plt.title('Linear Regression: Actual vs. Predicted % OBESE with Regression Line') plt.show() 

# Linear Regression for % Inactive and % Diabetic
X_inactive_diabetic = merged_data[['% INACTIVE']].values
y_inactive_diabetic = merged_data['% DIABETIC'].values

# Perform linear regression
regressor_inactive_diabetic = LinearRegression()
regressor_inactive_diabetic.fit(X_inactive_diabetic, y_inactive_diabetic)

# Visualize the linear regression line
plt.scatter(X_inactive_diabetic, y_inactive_diabetic)
plt.plot(X_inactive_diabetic, regressor_inactive_diabetic.predict(X_inactive_diabetic), color='red')
plt.xlabel('% Inactive')
plt.ylabel('% Diabetic')
plt.title('Linear Regression: % Diabetic vs. % Inactive')
plt.show()

# Print the coefficients
print('Slope:', regressor_inactive_diabetic.coef_[0])
print('Intercept:', regressor_inactive_diabetic.intercept_)


# Logistic Regression for % Inactive and % Diabetic
# Assuming you have merged_data, X_train, X_test, y_train, y_test available

# Logistic regression
log_reg_inactive_diabetic = LogisticRegression()
log_reg_inactive_diabetic.fit(X_train, y_train)

# Predictions
y_pred_inactive_diabetic = log_reg_inactive_diabetic.predict(X_test)

# Model evaluation
accuracy_logistic_inactive_diabetic = accuracy_score(y_test, y_pred_inactive_diabetic)
print('Accuracy (Logistic Regression for % Inactive and % Diabetic):', accuracy_logistic_inactive_diabetic)
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred_inactive_diabetic))


# PCA for % Inactive and % Diabetic
X_inactive_diabetic_pca = merged_data[['% INACTIVE', '% DIABETIC']]
scaler_inactive_diabetic = StandardScaler()
X_inactive_diabetic_scaled = scaler_inactive_diabetic.fit_transform(X_inactive_diabetic_pca)

pca_inactive_diabetic = PCA(n_components=2)
X_inactive_diabetic_pca_result = pca_inactive_diabetic.fit_transform(X_inactive_diabetic_scaled)

plt.scatter(X_inactive_diabetic_pca_result[:, 0], X_inactive_diabetic_pca_result[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: % Inactive vs. % Diabetic')
plt.show()


# Clustering for % Inactive and % Diabetic
n_clusters_inactive_diabetic = 3

kmeans_inactive_diabetic = KMeans(n_clusters=n_clusters_inactive_diabetic)
merged_data['Cluster (Inactive-Diabetic)'] = kmeans_inactive_diabetic.fit_predict(X_inactive_diabetic_pca)

sns.scatterplot(x='% INACTIVE', y='% DIABETIC', hue='Cluster (Inactive-Diabetic)', data=merged_data)
plt.title('Clustering: % Inactive vs. % Diabetic')
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import numpy as np

# Scatter plot with green bubbles
sns.scatterplot(x='% OBESE', y='% DIABETIC', data=merged_data, color='green')

# Linear regression
X = merged_data['% OBESE'].values.reshape(-1, 1)
y = merged_data['% DIABETIC'].values
regressor = LinearRegression()

# Perform 10-fold cross-validation
cv_scores = cross_val_score(regressor, X, y, cv=10)
mean_cv_score = np.mean(cv_scores)

# Plotting the regression line
regressor.fit(X, y)
plt.plot(X, regressor.predict(X), color='red')
plt.title('Linear Regression: % Diabetic vs. % Obese')

# Coefficients
slope = regressor.coef_[0]
intercept = regressor.intercept_

print('Slope:', slope)
print('Intercept:', intercept)
print('Mean 10-fold CV Score:', mean_cv_score)

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats  # Import stats module from scipy
import numpy as np

# Scatter plot with green bubbles
sns.scatterplot(x='% OBESE', y='% DIABETIC', data=merged_data, color='green')

# Linear regression
X = merged_data['% OBESE'].values.reshape(-1, 1)
y = merged_data['% DIABETIC'].values
regressor = LinearRegression()
regressor.fit(X, y)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(regressor, X, y, cv=10)
mean_cv_score = np.mean(cv_scores)

# Plotting the regression line
plt.plot(X, regressor.predict(X), color='red')
plt.title('Linear Regression: % Diabetic vs. % Obese')

# Coefficients
slope = regressor.coef_[0]
intercept = regressor.intercept_

print('Slope:', slope)
print('Intercept:', intercept)
print('Mean 10-fold CV Score:', mean_cv_score)

# Perform t-test on the slope (correlation coefficient)
t_statistic, p_value = stats.ttest_1samp(regressor.coef_, 0)

print('T-statistic:', t_statistic)
print('P-value:', p_value)

plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the dataset from Excel
df_dict = pd.read_excel('C:/Users/Hp/Downloads/cdc-diabetes-2018 (2).xlsx', sheet_name=None)
obesity_data = df_dict['Obesity']
inactivity_data = df_dict['Inactivity']
diabetes_data = df_dict['Diabetes']

# Merge the DataFrames
merged_data = pd.merge(obesity_data, inactivity_data, on='COUNTY')
merged_data = pd.merge(merged_data, diabetes_data, on='COUNTY')

# Select relevant columns for linear regression
X = merged_data[['% INACTIVE', '% DIABETIC']]
y = merged_data['% OBESE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared value
r_squared = r2_score(y_test, y_pred)
print('R-squared:', r_squared)

# Create box plots
plt.figure(figsize=(10, 6))
sns.boxplot(x='STATE_x', y='% OBESE', data=merged_data)
plt.xticks(rotation=90)
plt.title('Box Plot of % OBESE by State')
plt.show()


# Create box plots for % OBESE by state
plt.figure(figsize=(14, 8))
sns.boxplot(x='STATE_x', y='% OBESE', data=merged_data, order=merged_data['STATE_x'].value_counts().index)
plt.xticks(rotation=90)
plt.title('Box Plot of % OBESE by State')
plt.show()

# Plot residuals
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual % OBESE')
plt.ylabel('Predicted % OBESE')
plt.title('Actual vs. Predicted % OBESE')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the dataset from Excel
df_dict = pd.read_excel('C:/Users/Hp/Downloads/cdc-diabetes-2018 (2).xlsx', sheet_name=None)
obesity_data = df_dict['Obesity']
inactivity_data = df_dict['Inactivity']
diabetes_data = df_dict['Diabetes']

# Merge the DataFrames
merged_data = pd.merge(obesity_data, inactivity_data, on='COUNTY')
merged_data = pd.merge(merged_data, diabetes_data, on='COUNTY')

# Select relevant columns for linear regression
X = merged_data[['% INACTIVE', '% DIABETIC']]
y = merged_data['% OBESE']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared value
r_squared = r2_score(y_test, y_pred)
print('R-squared:', r_squared)

# Visualize the actual data and the linear regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test['% INACTIVE'], y_test, color='blue', label='Actual % OBESE')
plt.plot(X_test['% INACTIVE'], y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('% INACTIVE')
plt.ylabel('% OBESE')
plt.legend()
plt.title('Linear Regression: Actual vs. Predicted % OBESE with Regression Line')
plt.show()

import pandas as pd

# Load the data into a DataFrame
# Assuming merged_data is the DataFrame containing the provided data

# Filter the relevant columns for diabetes
diabetes_data = merged_data[['COUNTY', 'STATEW', '% DIABETIC']]

# Find the county with the highest diabetic percentage
max_diabetic_percentage = diabetes_data['% DIABETIC'].max()
county_with_max_diabetic = diabetes_data.loc[diabetes_data['% DIABETIC'] == max_diabetic_percentage]

# Find the county with the lowest diabetic percentage
min_diabetic_percentage = diabetes_data['% DIABETIC'].min()
county_with_min_diabetic = diabetes_data.loc[diabetes_data['% DIABETIC'] == min_diabetic_percentage]

# Print the county with the highest and lowest diabetic percentage
print("County with the Highest Diabetic Percentage:")
print(county_with_max_diabetic)
print("\nCounty with the Lowest Diabetic Percentage:")
print(county_with_min_diabetic)


# Filter the relevant columns for obesity
obesity_data = merged_data[['COUNTY', 'STATEW', '% OBESE']]

# Find the county with the highest obesity percentage
max_obesity_percentage = obesity_data['% OBESE'].max()
county_with_max_obesity = obesity_data.loc[obesity_data['% OBESE'] == max_obesity_percentage]

# Find the county with the lowest obesity percentage
min_obesity_percentage = obesity_data['% OBESE'].min()
county_with_min_obesity = obesity_data.loc[obesity_data['% OBESE'] == min_obesity_percentage]

# Print the county with the highest and lowest obesity percentage
print("\nCounty with the Highest Obesity Percentage:")
print(county_with_max_obesity)
print("\nCounty with the Lowest Obesity Percentage:")
print(county_with_min_obesity)


# Filter the relevant columns for inactivity
inactivity_data = merged_data[['COUNTY', 'STATEW', '% INACTIVE']]

# Find the county with the highest inactivity percentage
max_inactivity_percentage = inactivity_data['% INACTIVE'].max()
county_with_max_inactivity = inactivity_data.loc[inactivity_data['% INACTIVE'] == max_inactivity_percentage]

# Find the county with the lowest inactivity percentage
min_inactivity_percentage = inactivity_data['% INACTIVE'].min()
county_with_min_inactivity = inactivity_data.loc[inactivity_data['% INACTIVE'] == min_inactivity_percentage]

# Print the county with the highest and lowest inactivity percentage
print("\nCounty with the Highest Inactivity Percentage:")
print(county_with_max_inactivity)
print("\nCounty with the Lowest Inactivity Percentage:")
print(county_with_min_inactivity)


# Calculate basic descriptive statistics
descriptive_stats = merged_data.describe()
print(descriptive_stats)

# Calculate the correlation matrix
correlation_matrix = merged_data[['% OBESE', '% INACTIVE', '% DIABETIC']].corr()
print(correlation_matrix)


# Get top counties for each variable
top_counties_obesity = merged_data.nlargest(5, '% OBESE')
top_counties_inactivity = merged_data.nlargest(5, '% INACTIVE')
top_counties_diabetes = merged_data.nlargest(5, '% DIABETIC')

print("Top Counties for Obesity:\n", top_counties_obesity)
print("\nTop Counties for Inactivity:\n", top_counties_inactivity)
print("\nTop Counties for Diabetes:\n", top_counties_diabetes)


import seaborn as sns
import matplotlib.pyplot as plt

# Distribution plot for obesity
plt.figure(figsize=(10, 6))
sns.histplot(merged_data['% OBESE'], kde=True)
plt.title('Obesity Distribution')
plt.xlabel('Obesity Percentage')
plt.ylabel('Count')
plt.show()


# Boxplot for diabetes percentage
plt.figure(figsize=(10, 6))
sns.boxplot(x='STATE_x', y='% DIABETIC', data=merged_data)
plt.xticks(rotation=90)
plt.title('Diabetes Percentage by State')
plt.xlabel('State')
plt.ylabel('Diabetes Percentage')
plt.show()

# Pairplot for selected variables
sns.pairplot(merged_data[['% OBESE', '% INACTIVE', '% DIABETIC']])
plt.suptitle('Pairplot of Obesity, Inactivity, and Diabetes Percentage', y=1)
plt.show()


# Heatmap for correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# Scatter plot for diabetes vs. obesity
plt.figure(figsize=(10, 6))
sns.scatterplot(x='% OBESE', y='% DIABETIC', data=merged_data)
plt.title('Diabetes vs. Obesity')
plt.xlabel('Obesity Percentage')
plt.ylabel('Diabetes Percentage')
plt.show()

import statsmodels.api as sm

# Perform linear regression (e.g., Diabetes vs. Obesity)
X = merged_data['% OBESE']  # Independent variable (Obesity)
y = merged_data['% DIABETIC']  # Dependent variable (Diabetes)

# Add a constant (intercept) to the independent variable
X = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X).fit()

# Print the summary statistics of the regression
print(model.summary())

# Extract R-squared and p-value
r_squared = model.rsquared
p_value = model.pvalues[1]

# Print R-squared and p-value
print('R-squared:', r_squared)
print('p-value:', p_value)


import matplotlib.pyplot as plt
import numpy as np

# Scatter plot
plt.scatter(merged_data['% OBESE'], merged_data['% DIABETIC'], label='Data points')

# Linear regression line
plt.plot(merged_data['% OBESE'], model.predict(X), color='red', label='Linear Regression')

plt.xlabel('Obesity (%)')
plt.ylabel('Diabetes (%)')
plt.title('Obesity vs. Diabetes with Linear Regression')
plt.legend()
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'merged_data' is your merged DataFrame with relevant data

# Box plot for obesity across states
plt.figure(figsize=(12, 6))
sns.boxplot(x='STATE_x', y='% OBESE', data=merged_data)
plt.title('Obesity Distribution Across States')
plt.xlabel('State')
plt.ylabel('% Obese')
plt.xticks(rotation=90)
plt.show()

# Box plot for inactivity across states
plt.figure(figsize=(12, 6))
sns.boxplot(x='STATE_x', y='% INACTIVE', data=merged_data)
plt.title('Inactivity Distribution Across States')
plt.xlabel('State')
plt.ylabel('% Inactive')
plt.xticks(rotation=90)
plt.show()

# Box plot for diabetes across states
plt.figure(figsize=(12, 6))
sns.boxplot(x='STATE_x', y='% DIABETIC', data=merged_data)
plt.title('Diabetes Distribution Across States')
plt.xlabel('State')
plt.ylabel('% Diabetic')
plt.xticks(rotation=90)
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Load the dataset from Excel
df_dict = pd.read_excel('C:/Users/Hp/Downloads/cdc-diabetes-2018 (2).xlsx', sheet_name=None)
obesity_data = df_dict['Obesity']
inactivity_data = df_dict['Inactivity']
diabetes_data = df_dict['Diabetes']

# Merge the DataFrames
merged_data = pd.merge(obesity_data, inactivity_data, on='COUNTY')
merged_data = pd.merge(merged_data, diabetes_data, on='COUNTY')

# Select relevant columns for linear regression
X = merged_data[['% INACTIVE', '% DIABETIC']]
y = merged_data['% OBESE']

# Add a constant to the independent variables
X = sm.add_constant(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the linear regression model
model = sm.OLS(y_train, X_train).fit()

# Make predictions
y_pred = model.predict(X_test)

# Calculate R-squared value
r_squared = r2_score(y_test, y_pred)
print('R-squared:', r_squared)

# Print the summary of the regression model which includes p-values
print(model.summary())

# Visualize the actual data and the linear regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test['% INACTIVE'], y_test, color='blue', label='Actual % OBESE')
plt.plot(X_test['% INACTIVE'], y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel('% INACTIVE')
plt.ylabel('% OBESE')
plt.legend()
plt.title('Linear Regression: Actual vs. Predicted % OBESE with Regression Line')
plt.show()



# Visualize the actual data and the linear regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_test['% INACTIVE'], y_test, color='blue', label='Actual % OBESE')
plt.plot(X_test['% INACTIVE'], y_pred, color='red', linewidth=1, label='Regression Line')
plt.xlabel('% INACTIVE')
plt.ylabel('% OBESE')
plt.legend()
plt.title('Linear Regression: Actual vs. Predicted % OBESE with Regression Line')
plt.show()




# In[ ]:




