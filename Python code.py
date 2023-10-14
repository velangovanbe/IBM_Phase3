import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 2: Load the dataset
data = pd.read_csv("customer_data.csv")

# Step 3: Explore the dataset
print(data.head())

# Step 4: Data Preprocessing
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:\n", missing_values)

# Convert 'Genre' to a categorical variable
data['Genre'] = data['Genre'].astype('category')

# Rename columns for clarity
data = data.rename(columns={'Annual Income (k$)': 'Annual_Income', 'Spending Score (1-100)': 'Spending_Score'})

# Scale numeric columns (if needed)
scaler = StandardScaler()
data[['Age', 'Annual_Income', 'Spending_Score']] = scaler.fit_transform(data[['Age', 'Annual_Income', 'Spending_Score'])

# Feature selection (selecting only certain columns)
selected_columns = ['Age', 'Annual_Income', 'Spending_Score']
data = data[selected_columns]

# Step 5: The dataset is now preprocessed and ready for analysis
print("Preprocessed Data:\n", data.head())

Loading the dataset will result in the output of the first few rows of the dataset,
 which will look something like this:

   CustomerID   Genre  Age  Annual Income (k$)  Spending Score (1-100)
0           1    Male   19                  15                      39
1           2    Male   21                  15                      81
2           3  Female   20                  16                       6
3           4  Female   23                  16                      77
4           5  Female   31                  17                      40

Checking for missing values will show the number of missing values in each column, like this:

Missing Values:
CustomerID               0
Genre                    0
Age                      0
Annual Income (k$)       0
Spending Score (1-100)   0
dtype: int64

The final preprocessed data will look like this after the preprocessing steps:

        Age  Annual_Income  Spending_Score
0 -1.424569      -1.738999       -0.434801
1 -1.281035      -1.738999        1.195704
2 -1.352802      -1.700830       -1.715913
3 -1.137502      -1.700830        1.040418
4 -0.563369      -1.662660       -0.395980

This output shows the selected columns (Age, Annual_Income, and Spending_Score) after scaling. 
The values have been standardized (scaled) to have a mean of 0 and a standard deviation of 1.

The specific values will vary based on your dataset, but this is the general format of the output 
you can expect at each step.