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



# %% country distribution
# Create a mapping dictionary from ISO alpha-2 to full country names
country_mapping = {country.alpha_2: country.name for country in pycountry.countries}

n_per_country = 3000

# Calculate country counts and map codes to full names
country_counts = df['country'].map(country_mapping).value_counts()
country_counts_over = country_counts[country_counts > n_per_country]

print("\nDistribution of countries with over 5000 records:")
print("\nCounts:")
print(country_counts_over)

plt.figure(figsize=(12, 6))
country_counts_over.plot(kind='bar', logy=True)
plt.title('Distribution of Countries with Over 5000 Records (Log Scale)')
plt.xlabel('Country')
plt.ylabel('Count (Log Scale)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


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
sampled_df = sample_by_country(df, n_per_country=2000, n_top_countries=50)

print("\nSampled dataset shape:", sampled_df.shape)
print("\nSamples per country:")
print(sampled_df['country'].value_counts())
# %%  perform SVD analysis
def perform_svd_analysis(data, n_features=50):
    """
    Perform SVD analysis to find latent structures in the data
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data for SVD analysis
    n_features : int, default=50
        Number of features to use for analysis
    """
    # Select first n_features columns
    selected_data = data.iloc[:, :n_features]
    
    # Standardize the data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)
    
    # Perform SVD
    U, S, Vt = np.linalg.svd(scaled_data, full_matrices=False)
    
    # Calculate variance explained
    explained_variance_ratio = (S ** 2) / (S ** 2).sum()
    
    # Get feature loadings (correlation between features and singular vectors)
    feature_loadings = Vt.T * S
    
    # Get observation scores
    observation_scores = U * S
    
    return {
        'singular_values': S,
        'explained_variance_ratio': explained_variance_ratio,
        'feature_loadings': feature_loadings,
        'observation_scores': observation_scores,
        'feature_directions': Vt,
        'observation_directions': U
    }

def visualize_svd_results(svd_results, feature_names):
    """
    Visualize the results of SVD analysis
    """
    # Plot scree plot (variance explained)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(svd_results['singular_values']) + 1), 
            svd_results['explained_variance_ratio'], color='#4682B4', marker='o', linestyle='-')
    plt.title('Scree Plot: Variance Explained by Components')
    plt.xlabel('Component')
    plt.ylabel('Proportion of Variance Explained')
    plt.show()
    # Plot feature loadings for first two components
    plt.figure(figsize=(12, 8))
    loadings = svd_results['feature_loadings'][:, :2]
    
    # Define colors and markers for each prefix
    style_map = {
        'EST': {'color': '#FF6B6B', 'marker': 'o'},     # Circle
        'OPN': {'color': '#4D96FF', 'marker': 's'},     # Square
        'EXT': {'color': '#6BCB77', 'marker': '^'},     # Triangle
        'CSN': {'color': '#9B72AA', 'marker': 'D'},     # Diamond
        'AGR': {'color': '#FFB562', 'marker': 'p'}      # Pentagon
    }
    
    # Create scatter plots with different colors and shapes based on feature prefix
    for prefix, style in style_map.items():
        mask = [feature.startswith(prefix) for feature in feature_names]
        if any(mask):  # Only plot if there are features with this prefix
            plt.scatter(loadings[mask, 0], loadings[mask, 1],
                       c=style['color'], marker=style['marker'],
                       label=prefix, alpha=0.7,s=100)
    
    # Add feature labels
    for i, feature in enumerate(feature_names):
        plt.annotate(feature, (loadings[i, 0], loadings[i, 1]))
    
    plt.title('Feature Loadings: First Two Components')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.legend()
    plt.show()
    

# Perform SVD analysis
print("\nPerforming SVD Analysis...")
svd_results = perform_svd_analysis(sampled_df, n_features=50)

# Visualize results
visualize_svd_results(svd_results, sampled_df.columns[:50])
def plot_important_components_heatmap(svd_results, threshold=0.025):
    """
    Plot heatmap of feature loadings for important components based on explained variance ratio.
    Features are grouped by their prefixes (EST, OPN, EXT, CSN, AGR).
    
    Args:
        svd_results (dict): Dictionary containing SVD results
        threshold (float): Minimum explained variance ratio threshold to select components
    """
    # Get components that explain more than threshold variance individually
    important_components = np.where(svd_results['explained_variance_ratio'] >= threshold)[0]
    n_components = len(important_components)
    
    if n_components == 0:
        print(f"No components explain more than {threshold*100:.1f}% variance individually")
        return
        
    # Get feature loadings for selected components
    important_loadings = svd_results['feature_loadings'][:, important_components]
    
    # Sort features by prefix
    features = sampled_df.columns[:50]
    prefixes = ['EST', 'OPN', 'EXT', 'CSN', 'AGR']
    sorted_indices = []
    for prefix in prefixes:
        prefix_indices = [i for i, feature in enumerate(features) if feature.startswith(prefix)]
        sorted_indices.extend(prefix_indices)
    
    # Reorder loadings and feature names
    important_loadings = important_loadings[sorted_indices]
    sorted_features = features[sorted_indices]
    
    plt.figure(figsize=(10, 12))
    
    # Create heatmap
    ax = sns.heatmap(important_loadings,
                xticklabels=[f'{i+1} \n({svd_results["explained_variance_ratio"][i]:.1%})' 
                            for i in important_components],
                yticklabels=sorted_features,
                cmap='RdBu',
                center=0,
                annot=False,
                fmt='.2f',
                cbar_kws={'label': 'Loading Value'})

    # Add horizontal lines to separate prefixes
    current_pos = 0
    for prefix in prefixes:
        prefix_count = sum(1 for f in sorted_features if f.startswith(prefix))
        if prefix_count > 0:
            current_pos += prefix_count
            if current_pos < len(sorted_features):
                plt.axhline(y=current_pos, color='white', linewidth=3)

    plt.title('Feature Loadings for Important Components')
    plt.xlabel('Components (Explained Variance Ratio)')
    plt.ylabel('Features')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Plot heatmap for components explaining >10% variance individually
plot_important_components_heatmap(svd_results)


# %% cluster by feature
def cluster_features(svd_results, n_components=None):
    """
    Cluster features based on their loadings in the first n_components.
    Returns optimal clustering results based on silhouette score.
    
    Parameters:
    -----------
    svd_results : dict
        Dictionary containing SVD results
    n_components : int, optional
        Number of components to use for clustering. Defaults to all components.
        
    Returns:
    --------
    dict
        Dictionary containing feature names and their cluster labels
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    # Get feature loadings for specified components
    loadings = svd_results['feature_loadings']
    if n_components:
        loadings = loadings[:, :n_components]
        
    # Try different numbers of clusters and calculate silhouette scores
    silhouette_scores = []
    kmeans_models = []
    
    for n_clusters in range(2, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(loadings)
        score = silhouette_score(loadings, labels)
        silhouette_scores.append(score)
        kmeans_models.append(kmeans)
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, 11), silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.grid(True)
    plt.show()
    
    # Select optimal number of clusters based on highest silhouette score
    optimal_idx = np.argmax(silhouette_scores)
    optimal_kmeans = kmeans_models[optimal_idx]
    optimal_labels = optimal_kmeans.fit_predict(loadings)
    
    # Create dictionary of features and their cluster labels
    features = sampled_df.columns[:50]  # First 50 columns are the features
    clustering_results = {
        'features': features,
        'labels': optimal_labels,
        'n_clusters': optimal_idx + 2,
        'silhouette_score': silhouette_scores[optimal_idx]
    }
    
    # Compare original categories with cluster labels
    prefixes = ['EST', 'OPN', 'EXT', 'CSN', 'AGR']
    comparison_df = pd.DataFrame({
        'Feature': features,
        'Original_Category': [next(prefix for prefix in prefixes if feature.startswith(prefix)) 
                            for feature in features],
        'Cluster_Label': optimal_labels
    })
    
    # Create contingency table
    contingency_table = pd.crosstab(comparison_df['Original_Category'], 
                                   comparison_df['Cluster_Label'])
    
    # Plot heatmap of contingency table
    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Item Category Recovery Plot')
    plt.xlabel('Cluster Label')
    plt.ylabel('Original Subscale')
    plt.show()
    
    return clustering_results

# Example usage:
cluster_results = cluster_features(svd_results, n_components=5)
print(f"Optimal number of clusters: {cluster_results['n_clusters']}")
print(f"Silhouette score: {cluster_results['silhouette_score']:.3f}")

#%% by country
def plot_country_components(svd_results, df, n_components=3):
    """
    Plot average component scores by country.
    
    Parameters:
    -----------
    svd_results : dict
        Dictionary containing SVD results
    df : pandas.DataFrame
        Original dataframe with country information
    n_components : int, default=3
        Number of top components to plot
    """
    # Get observation scores for top components
    scores = svd_results['observation_scores'][:, :n_components]
    
    # Create dataframe with scores and country
    scores_df = pd.DataFrame(scores, columns=[f'{i+1}' for i in range(n_components)])
    scores_df['country'] = df['country']
    
    # Calculate mean scores by country
    country_means = scores_df.groupby('country').mean()
    
    # Map country codes to full names
    country_mapping = {country.alpha_2: country.name for country in pycountry.countries}
    country_means.index = country_means.index.map(country_mapping)
    
    # Plot heatmap
    plt.figure(figsize=(9, 8))
    sns.heatmap(country_means, 
                cmap='RdBu',
                center=0,
                annot=False,
                fmt='.2f',
                cbar_kws={'label': 'Average Component Score'})
    
    plt.title('Average Component Scores by Country')
    plt.xlabel('Components')
    plt.ylabel('Country')
    plt.tight_layout()
    plt.show()
    # Plot bar charts for each component
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 4*n_components))
    for i in range(n_components):
        component_data = country_means.iloc[:, i].sort_values(ascending=True)
        ax = axes[i] if n_components > 1 else axes
        component_data.plot(kind='barh', ax=ax)
        ax.set_title(f'Component {i+1} Scores by Country')
        ax.set_xlabel('Average Score')
        ax.set_xlim(-1, 1)  # Set x-axis limits to -1 to 1
        
    plt.tight_layout()
    plt.show()

# Plot country component analysis
plot_country_components(svd_results, sampled_df, n_components=7)

#%% cluster by country
def cluster_countries_by_components(svd_results, df, n_components=7):
    """
    Cluster countries based on their average component scores from SVD analysis.
    
    Parameters:
    -----------
    svd_results : dict
        Results from SVD analysis containing observation_scores
    df : pandas.DataFrame
        Original dataframe with country information
    n_components : int, default=7
        Number of components to use for clustering
        
    Returns:
    --------
    optimal_labels : array
        Cluster labels for each country
    silhouette_scores : list
        Silhouette scores for each number of clusters tried
    """
    # Get observation scores for top components
    scores = svd_results['observation_scores'][:, :n_components]
    
    # Create dataframe with scores and country
    scores_df = pd.DataFrame(scores, columns=[f'Component_{i+1}' for i in range(n_components)])
    scores_df['country'] = df['country']
    
    # Calculate mean scores by country
    country_means = scores_df.groupby('country').mean()

    print(country_means)
    # Perform hierarchical clustering
    from scipy.cluster import hierarchy
    from sklearn.preprocessing import StandardScaler
    
    # Standardize country means
    scaler = StandardScaler()
    country_means_scaled = scaler.fit_transform(country_means)
    
    # Calculate linkage matrix
    linkage_matrix = hierarchy.linkage(country_means_scaled, method='ward')
    
    # Map country codes to full names using pycountry
    country_names = [country_mapping.get(code, code) for code in country_means.index]
    
    # Plot dendrogram with full country names
    plt.figure(figsize=(10, 8))
    dendrogram = hierarchy.dendrogram(linkage_matrix, labels=country_names, leaf_rotation=90)
    plt.title('Hierarchical Clustering of Countries')
    plt.xlabel('Country')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()
    
    # Get cluster labels for a specific number of clusters (e.g. 4)
    n_clusters = 4
    cluster_labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Country': country_names,
        'Cluster': cluster_labels
    })
    
    print("\nClustering Results:")
    print(results_df.sort_values('Cluster'))
    
    return cluster_labels, None  # Return None for silhouette scores to match function signature

# Perform clustering analysis
cluster_labels, silhouette_scores = cluster_countries_by_components(svd_results, sampled_df,n_components=5)


# %%
