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

# Sidebar Navigation
st.sidebar.title("Navigation")
options = ["Factor Loading", "EDA - Correlation Matrix", "Regression Coefficients", "RF Feature Importance"]
selection = st.sidebar.radio("Choose an option", options)

# Factor Loading Display
if selection == "Factor Loading":
    st.title("Factor Loadings")
    st.dataframe(factor_loadings)
    st.bar_chart(factor_loadings.set_index('Variable')['Factor Loading'])

# Correlation Matrix (EDA)
elif selection == "EDA - Correlation Matrix":
    st.title("Correlation Matrix")
    st.write("Visualizing the correlation matrix of the merged dataset.")
    
    # Exclude non-relevant columns like 'Time' and 'year'
    columns_to_exclude = ['Time', 'year']
    filtered_data = merged_data.drop(columns=columns_to_exclude, errors='ignore')
    
    # Select numeric columns
    numeric_columns = filtered_data.select_dtypes(include=[np.number])
    corr_matrix = numeric_columns.corr()

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Regression Coefficients Display
elif selection == "Regression Coefficients":
    st.title("Regression Coefficients")
    st.dataframe(regression_coefficients)
    st.bar_chart(regression_coefficients.set_index('Feature')['Coefficient'])

# Random Forest Feature Importance
elif selection == "RF Feature Importance":
    st.title("Random Forest Feature Importance")
    st.dataframe(feature_importance)
    
    pivot_table = feature_importance.pivot(index='SDG', columns='Target', values='Importance')
    
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

st.sidebar.info("Developed for analyzing SDG and HR metrics with Random Forest, Factor Analysis, and Regression.")
