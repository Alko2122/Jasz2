import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import requests

# Function to download the file and load it with pandas
def download_file(url, local_filename):
    # Send GET request to GitHub raw URL
    r = requests.get(url)
    r.raise_for_status()  # Will raise an error if the download fails
    with open(local_filename, 'wb') as f:
        f.write(r.content)

def load_data():
    try:
        # URLs for Excel files
        file_urls = {
            "factor_loadings": "https://github.com/Alko2122/Jasz2/raw/refs/heads/main/ResultsSDG_Factor_Loadings.xlsx",
            "regression_coefficients": "https://github.com/Alko2122/Jasz2/raw/refs/heads/main/Linear_Regression_Coefficients_SDG.xlsx",
            "feature_importance": "https://github.com/Alko2122/Jasz2/raw/refs/heads/main/Feature_Importances.xlsx",
            "merged_data": "https://github.com/Alko2122/Jasz2/raw/refs/heads/main/Merged_Data_SDG_GCI.xlsx"
        }

        # Download files locally
        download_file(file_urls["factor_loadings"], "ResultsSDG_Factor_Loadings.xlsx")
        download_file(file_urls["regression_coefficients"], "Linear_Regression_Coefficients_SDG.xlsx")
        download_file(file_urls["feature_importance"], "Feature_Importances.xlsx")
        download_file(file_urls["merged_data"], "Merged_Data_SDG_GCI.xlsx")

        # Now, load the downloaded files
        factor_loadings = pd.read_excel("ResultsSDG_Factor_Loadings.xlsx")
        regression_coefficients = pd.read_excel("Linear_Regression_Coefficients_SDG.xlsx")
        feature_importance = pd.read_excel("Feature_Importances.xlsx")
        merged_data = pd.read_excel("Merged_Data_SDG_GCI.xlsx")

        return factor_loadings, regression_coefficients, feature_importance, merged_data
    except Exception as e:
        st.error(f"Error downloading or loading data: {e}")
        return None, None, None, None

# Load data
factor_loadings, regression_coefficients, feature_importance, merged_data = load_data()

# Check if data was loaded successfully
if factor_loadings is None:
    st.stop()  # Stop the app if data loading fails

# Sidebar Navigation (reordered)
st.sidebar.title("Navigation")
options = ["EDA - Correlation Matrix", "RF Feature Importance", "Regression Coefficients", "Factor Loading"]
selection = st.sidebar.radio("Choose an option", options)

# Correlation Matrix (EDA)
if selection == "EDA - Correlation Matrix":
    st.title("Correlation Matrix")
    st.write("Visualizing the correlation matrix of the merged dataset.")
    
    # Exclude non-relevant columns like 'Time' and 'year'
    columns_to_exclude = ['Time', 'year']
    filtered_data = merged_data.drop(columns=columns_to_exclude, errors='ignore')
    
    # Multi-select widget for numeric column selection
    available_columns = filtered_data.select_dtypes(include=[np.number]).columns.tolist()
    selected_columns = st.multiselect("Select Numeric Columns", available_columns, default=available_columns)
    
    # Select numeric columns
    numeric_columns = filtered_data[selected_columns]
    corr_matrix = numeric_columns.corr()

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Random Forest Feature Importance with feature selection
elif selection == "RF Feature Importance":
    st.title("Random Forest Feature Importance")
    
    # Get the unique SDGs and the unique first column's data
    available_sdg = feature_importance['SDG'].unique().tolist()  # Get unique SDGs
    available_first_column = feature_importance.iloc[:, 0].unique().tolist()  # First column data (unique)
    
    # Multi-select widget for SDG and First Column selection
    selected_sdg = st.multiselect("Select SDGs", available_sdg, default=available_sdg)
    selected_first_column = st.multiselect("Select First Column Data", available_first_column, default=available_first_column)
    
    # Filter the dataframe to show only selected SDGs and first column data
    filtered_feature_importance = feature_importance[
        feature_importance['SDG'].isin(selected_sdg) & 
        feature_importance.iloc[:, 0].isin(selected_first_column)  # Using dynamic column selection
    ]
    
    # Pivot table for visualization
    pivot_table = filtered_feature_importance.pivot(index='SDG', columns='Target', values='Importance')
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".4f",
        cmap='coolwarm',
        cbar_kws={'label': 'Importance'},
        linewidths=0.5,
        ax=ax
    )
    st.pyplot(fig)

# Regression Coefficients Display with feature selection
elif selection == "Regression Coefficients":
    st.title("Regression Coefficients")
    available_features = regression_coefficients['Feature'].tolist()
    
    # Multi-select widget for feature selection
    selected_features = st.multiselect("Select Features", available_features, default=available_features)
    
    # Filter the dataframe to only show selected features
    filtered_regression_coefficients = regression_coefficients[regression_coefficients['Feature'].isin(selected_features)]
    
    st.dataframe(filtered_regression_coefficients)
    st.bar_chart(filtered_regression_coefficients.set_index('Feature')['Coefficient'])

# Factor Loading Display with feature selection (unique first column values)
elif selection == "Factor Loading":
    st.title("Factor Loadings")
    
    # Get unique values from the first column (Variable column)
    available_features = factor_loadings.iloc[:, 0].unique().tolist()  # Using first column
    
    # Multi-select widget for unique feature selection from the first column
    selected_features = st.multiselect("Select Variables", available_features, default=available_features)
    
    # Filter the dataframe to only show selected features
    filtered_factor_loadings = factor_loadings[factor_loadings.iloc[:, 0].isin(selected_features)]
    
    st.dataframe(filtered_factor_loadings)
    
st.sidebar.info("Developed for analyzing SDG and HR metrics with Random Forest, Factor Analysis, and Regression.")
