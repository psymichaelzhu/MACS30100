#%%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Import country code mapping library
import pycountry
#%% load data
# Define the path to the dataset
dataset_path = os.path.join('..', 'Dataset', 'BF-dataset.csv')

# Read the CSV file
try:
    df = pd.read_csv(dataset_path, sep='\t', header=0)
    print("Dataset loaded successfully!")
    print("\nDataset shape:", df.shape) 
    print("\nFirst few rows of the dataset:")
    print(df.head())
    
except FileNotFoundError:
    print("Error: Could not find the file. Please check if the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")

# %% pre check
# Print total number of rows in the dataset
print("\nTotal number of rows:", len(df))

# Print total number of columns in the dataset
print("Total number of columns:", len(df.columns))

# Mark missing values and values outside 1-5 range as 999
first_50_cols = df.iloc[:, :50].columns
for col in first_50_cols:
    # Mark NaN values as 999
    df.loc[df[col].isnull(), col] = 999
    # Mark values outside 1-5 range as 999
    df.loc[~df[col].between(1, 5), col] = 999

# Count how many rows contain 999 before removal
rows_with_999 = df[df.iloc[:, :50].eq(999).any(axis=1)]
print(f"\nNumber of rows containing 999 before removal: {len(rows_with_999)}")

# Remove rows containing 999 in first 50 columns
df = df[~df.iloc[:, :50].eq(999).any(axis=1)]

print(f"\nNumber of rows remaining after removal: {len(df)}")

# Verify no more invalid values
missing_values = df.iloc[:, :50].isnull().sum()
print("\nMissing values after cleaning:")
print(missing_values[missing_values > 0])

invalid_ranges = {}
for col in first_50_cols:
    invalid_values = df[~df[col].between(1, 5)][col].unique()
    if len(invalid_values) > 0:
        invalid_ranges[col] = invalid_values

if invalid_ranges:
    print("\nColumns still with values outside 1-5 range:")
    for col, values in invalid_ranges.items():
        print(f"{col}: {values}")
else:
    print("\nAll values in first 50 columns are now within 1-5 range")
# Remove rows containing NaN values
df = df.dropna()
print(f"\nNumber of rows after removing NaN values: {len(df)}")
# Remove rows where country is 'NONE'
# Remove rows where any column contains 'NONE'
df = df[~df.isin(['NONE']).any(axis=1)]
print(f"\nNumber of rows after removing rows with NONE values: {len(df)}")


#%% BF preprocess
# Reverse score certain items based on the codebook
# Items to be reverse scored (convert 1->5, 2->4, etc):
reverse_items = [
    'EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10',  # Extraversion
    'EST2', 'EST4',  # Emotional Stability 
    'AGR1', 'AGR3', 'AGR5', 'AGR7',  # Agreeableness 
    'CSN2', 'CSN4', 'CSN6', 'CSN8',  # Conscientiousness
    'OPN2', 'OPN4', 'OPN6'  # Openness
]

# Function to reverse score (6 minus score)
def reverse_score(x):
    return 6 - x

# Apply reverse scoring
for item in reverse_items:
    df[item] = df[item].apply(reverse_score)

print("\nReverse scoring completed for the following items:")
print(reverse_items)





#%% Define and apply stratified sampling function
def sample_by_country(df, n_per_country=200, n_top_countries=None, random_state=42):
    """
    Sample n records from each country in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing a 'country' column
    n_per_country : int, default=200
        Number of samples to take from each country
    n_top_countries : int, optional
        Number of top countries by sample size to include. If None, include all countries.
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pandas.DataFrame
        Sampled DataFrame with n records from each country
    """
    sampled_dfs = []
    
    # Get country counts and filter countries with enough samples
    country_counts = df['country'].value_counts()
    qualified_countries = country_counts[country_counts >= n_per_country]
    
    # If number of qualified countries is less than n_top_countries,
    # use the number of qualified countries instead
    if n_top_countries and len(qualified_countries) > n_top_countries:
        selected_countries = qualified_countries.head(n_top_countries).index
    else:
        selected_countries = qualified_countries.index
    
    # Sample from each selected country
    for country in selected_countries:
        country_data = df[df['country'] == country]
        sampled_country = country_data.sample(n=n_per_country, random_state=random_state)
        sampled_dfs.append(sampled_country)
    
    # Combine all sampled data
    return pd.concat(sampled_dfs, ignore_index=True)

# Apply stratified sampling from top 10 countries
sampled_df = sample_by_country(df, n_per_country=3000, n_top_countries=40)

print("\nSampled dataset shape:", sampled_df.shape)
print("\nSamples per country:")
print(sampled_df['country'].value_counts())

#%% RT EDA
# Extract RT columns (ending with _E) and remove the _E suffix
rt_cols = sampled_df.columns[50:100]
rt_df = sampled_df[rt_cols].copy()
rt_df.columns = [col.replace('_E', '') for col in rt_cols]

print(rt_df)


#%%
# Create a long format dataframe for visualization
rt_long = rt_df.melt(var_name='item', value_name='RT')

# Extract prefix and item number
rt_long['prefix'] = rt_long['item'].str.extract('([A-Z]+)')
rt_long['item_num'] = rt_long['item'].str.extract('(\d+)').astype(int)

# Set EXT1 response time to 6000
rt_long.loc[(rt_long['prefix'] == 'EXT') & (rt_long['item_num'] == 1), 'RT'] = 6000

# Create the heatmap
plt.figure(figsize=(15, 8))
pivot_table = rt_long.pivot_table(
    values='RT', 
    index='prefix',
    columns='item_num',
    aggfunc='mean'
)

# Plot heatmap
sns.heatmap(pivot_table, 
            cmap='YlOrRd',
            annot=True, 
            fmt='.0f',
            cbar_kws={'label': 'Response Time (ms)'})

plt.title('Average Response Times by Item')
plt.xlabel('Item Number')
plt.ylabel('Question Category')
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary statistics for response times by category:")
print(rt_long.groupby('prefix')['RT'].describe())


# %%
# Extract score columns (first 50 columns) and create long format dataframe
score_df = sampled_df.iloc[:, :50].copy()
score_long = score_df.melt(var_name='item', value_name='Score')

# Extract prefix and item number 
score_long['prefix'] = score_long['item'].str.extract('([A-Z]+)')
score_long['item_num'] = score_long['item'].str.extract('(\d+)').astype(int)

# Create the heatmap
plt.figure(figsize=(15, 8))
pivot_table = score_long.pivot_table(
    values='Score',
    index='prefix', 
    columns='item_num',
    aggfunc='mean'
)

# Plot heatmap
sns.heatmap(pivot_table,
            cmap='YlOrRd', 
            annot=True,
            fmt='.2f',
            cbar_kws={'label': 'Average Score'})

plt.title('Average Scores by Item')
plt.xlabel('Item Number')
plt.ylabel('Question Category')
plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary statistics for scores by category:")
print(score_long.groupby('prefix')['Score'].describe())

# %%
# Extract response time columns and create long format dataframe
rt_cols = [col for col in sampled_df.columns if col.endswith('_E')]
rt_df = sampled_df[rt_cols].copy()
rt_long = rt_df.melt(var_name='item', value_name='RT')

# Extract corresponding score columns and values
score_cols = [col[:-2] for col in rt_cols]  # Remove '_E' suffix
rt_long['Score'] = sampled_df[score_cols].values.flatten()

# Filter for scores 1-5 only
rt_long = rt_long[rt_long['Score'].between(1, 5)]

# Remove extreme response times (below 5th and above 95th percentile)
lower = rt_long['RT'].quantile(0.05)
upper = rt_long['RT'].quantile(0.95)
rt_long = rt_long[(rt_long['RT'] >= lower) & (rt_long['RT'] <= upper)]

# Create violin plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=rt_long, x='Score', y='RT')

plt.title('Response Time Distribution by Score (Excluding Extremes)')
plt.xlabel('Score Value (1=Disagree to 5=Agree)')
plt.ylabel('Response Time (ms)')

# Add median line
median_rt = rt_long['RT'].median()
plt.axhline(y=median_rt, color='r', linestyle='--', alpha=0.5, label=f'Median RT: {median_rt:.0f}ms')
plt.legend()

plt.tight_layout()
plt.show()

# Print summary statistics
print("\nSummary statistics for response times by score (excluding extremes):")
print(rt_long.groupby('Score')['RT'].describe())

# %%
