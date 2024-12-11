import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# two decimal points
pd.options.display.float_format = '{:.2f}'.format

# Load the data
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                'marital-status', 'occupation', 'relationship', 'race', 'sex',
                'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
train_data = pd.read_csv('adult/adult.data', header=None, names=column_names, na_values=' ?')
test_data = pd.read_csv('adult/adult.test', header=None, names=column_names, skiprows=1, na_values=' ?')

# Numerical Features
# print(train_data.describe())

# Categorical Distribution
# print(train_data['education'].value_counts())
# print(train_data['income'].value_counts())
# print(train_data['workclass'].value_counts())
# print(train_data['marital-status'].value_counts())
# print(train_data['occupation'].value_counts())
# print(train_data['relationship'].value_counts())
# print(train_data['race'].value_counts())
# print(train_data['sex'].value_counts())
# print(train_data['native-country'].value_counts())

# Corelation
# corr = train_data.corr()
# plt.figure(figsize=(10, 6))
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix Heatmap')
# plt.show()


# education_income = pd.crosstab(train_data['education'], train_data['income'], normalize='index')
# education_income.plot(kind='bar', stacked=True, figsize=(10, 6))
# plt.title('Income Distribution by Education Level')
# plt.xlabel('Education Level')
# plt.ylabel('Proportion')
# plt.show()

