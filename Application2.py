import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import math
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pandas.io.formats.excel import ExcelFormatter
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.decomposition import FactorAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
# Load datasets
sdg_data = pd.read_excel(https://github.com/Alko2122/Jasz2/blob/38edb1d7e6e0b05b559ee2129945439d3422014b/SDGs.xlsx)
gci_data = pd.read_excel(r"C:\Users\alkoj\OneDrive\Desktop\WEF.xlsx")

# %% [markdown]
# # Data Cleaning - GCI Dataset

# %% [markdown]
# ### Drop irrelevant columns

# %%
# List of columns to keep
columns_to_keep = [
    'Country',
    'year',
    '7.01 Cooperation in labor-employer relations, 1-7 (best)',
    '7.02 Flexibility of wage determination, 1-7 (best)',
    '7.03 Hiring and firing practices, 1-7 (best)',
    '7.04 Redundancy costs, weeks of salary*',
    '7.05 Effect of taxation on incentives to work, 1-7 (best)',
    '7.06 Pay and productivity, 1-7 (best)',
    '7.07 Reliance on professional management, 1-7 (best)',
    '7.08 Country capacity to retain talent, 1-7 (best)',
    '7.09 Country capacity to attract talent, 1-7 (best)',
    '7.10 Women in labor force, ratio to men*'
]

# Filter GCI dataset to retain only the specified columns
gci_data_filtered = gci_data[columns_to_keep]

# Display the first few rows to verify columns
gci_data_filtered.head()

# %% [markdown]
# ### Filtered GCI data overview

# %%
gci_data_filtered.info()

# %%
# Rename the columns with more concise names
gci_data_filtered.rename(columns={
    '7.01 Cooperation in labor-employer relations, 1-7 (best)': 'Cooperation_Labor_Employer',
    '7.02 Flexibility of wage determination, 1-7 (best)': 'Wage_Flexibility',
    '7.03 Hiring and firing practices, 1-7 (best)': 'Hiring_Firing_Practices',
    '7.04 Redundancy costs, weeks of salary*': 'Redundancy_Costs',
    '7.05 Effect of taxation on incentives to work, 1-7 (best)': 'Taxation_Incentives_Work',
    '7.06 Pay and productivity, 1-7 (best)': 'Pay_Productivity',
    '7.07 Reliance on professional management, 1-7 (best)': 'Professional_Management',
    '7.08 Country capacity to retain talent, 1-7 (best)': 'Talent_Retention',
    '7.09 Country capacity to attract talent, 1-7 (best)': 'Talent_Attraction',
    '7.10 Women in labor force, ratio to men*': 'Women_Labor_Ratio'
}, inplace=True)

# Verify renamed columns
gci_data_filtered.head()

# %%
# Convert 'Redundancy costs' column to float, coercing any non-numeric values to NaN
gci_data_filtered['Redundancy_Costs'] = pd.to_numeric(
    gci_data_filtered['Redundancy_Costs'], errors='coerce'
)

# %%
# Check if column is converted to the right data type
gci_data_filtered.info()

# %%
# Check for missing values
print("Missing values:")
print(gci_data_filtered.isnull().sum())

# Check for duplicates
print(f"\nNumber of duplicated rows: {gci_data_filtered.duplicated().sum()}")

# %%
# Interpolate missing values in historical data
gci_data_interpolated = gci_data_filtered.copy()
forecast_columns = [
    'Cooperation_Labor_Employer',
    'Wage_Flexibility',
    'Hiring_Firing_Practices',
    'Redundancy_Costs',
    'Taxation_Incentives_Work',
    'Pay_Productivity',
    'Professional_Management',
    'Talent_Retention',
    'Talent_Attraction',
    'Women_Labor_Ratio'
]

# Sort data for interpolation to occur in year order within each country
gci_data_interpolated = gci_data_interpolated.sort_values(['Country', 'year']).reset_index(drop=True)

# Interpolate for each country and each specified column
for country in gci_data_interpolated['Country'].unique():
    country_data = gci_data_interpolated[gci_data_interpolated['Country'] == country]
    for column in forecast_columns:
        gci_data_interpolated.loc[gci_data_interpolated['Country'] == country, column] = (
            country_data[column].interpolate(method='linear', limit_direction='both')
        )

# %%
# Backfill any remaining missing values
gci_data_interpolated[forecast_columns] = gci_data_interpolated[forecast_columns].fillna(method='bfill')

for country in gci_data_interpolated['Country'].unique():
    country_data = gci_data_interpolated[gci_data_interpolated['Country'] == country].sort_values('year')
    for column in forecast_columns:
        if country_data[column].isnull().any():
            # Filter non-NaN values for modeling
            non_nan_data = country_data.dropna(subset=[column])
            if len(non_nan_data) > 1:  # Ensure there's enough data for forecasting
                model = ExponentialSmoothing(
                    non_nan_data[column],
                    trend='add', seasonal=None, initialization_method="estimated"
                ).fit()
                forecast_values = model.predict(start=country_data['year'].min(), end=country_data['year'].max())
                # Fill in missing values in the column for the country's data
                gci_data_interpolated.loc[gci_data_interpolated['Country'] == country, column] = (
                    country_data[column].combine_first(forecast_values)
                )

# Create the `Country_year` column after interpolation and backfilling
gci_data_interpolated['Country_year'] = gci_data_interpolated['Country'] + '_' + gci_data_interpolated['year'].astype(str)

# %%
# Verify missing values after interpolation and backfill
missing_values_after_interpolation = gci_data_interpolated.isnull().sum()
print("Missing values after interpolation and backfill:")
print(missing_values_after_interpolation[missing_values_after_interpolation > 0])

# Check for duplicated values after interpolation
duplicates_after_interpolation = gci_data_interpolated.duplicated().sum()
print(f"\nNumber of duplicated rows after interpolation: {duplicates_after_interpolation}")

# %%
# Forecast future values for each country
forecasted_data = pd.DataFrame()
countries = gci_data_interpolated['Country'].unique()

for country in countries:
    country_data = gci_data_interpolated[gci_data_interpolated['Country'] == country].sort_values('year')
    forecast_results = {col: [] for col in forecast_columns}

    # Forecast future values for each specified column
    for column in forecast_columns:
        if not country_data[column].isnull().all():
            model = ExponentialSmoothing(
                country_data[column],
                trend='add', seasonal=None, initialization_method="estimated"
            ).fit()
            forecast = model.forecast(2022 - country_data['year'].max())
        else:
            forecast = [np.nan] * (2022 - country_data['year'].max())

        # Append forecast or NaN values to the forecast results for each column
        forecast_results[column] = list(forecast)

    # Generate the forecasted rows
    forecast_years = range(country_data['year'].max() + 1, 2023)
    for year_idx, year in enumerate(forecast_years):
        forecasted_data = pd.concat([forecasted_data, pd.DataFrame({
            'Country': [country],
            'year': [year],
            **{col: [forecast_results[col][year_idx]] for col in forecast_columns},
            'Country_year': [f"{country}_{year}"]
        })], ignore_index=True)

# %%
# Verify missing values in forecasted data
missing_values_in_forecasted_data = forecasted_data.isnull().sum()
print("\nMissing values in forecasted data:")
print(missing_values_in_forecasted_data[missing_values_in_forecasted_data > 0])

# Check for duplicated values in forecasted data
duplicates_in_forecasted_data = forecasted_data.duplicated().sum()
print(f"\nNumber of duplicated rows in forecasted data: {duplicates_in_forecasted_data}")

# %%
# Display the interpolated data
print("Interpolated Data:")
gci_data_interpolated.head()

# %%
# Display the forecasted data
print("\nForecasted Data:")
forecasted_data.head()

# %%
# Combine interpolated and forecasted data
combined_gcidata = pd.concat([gci_data_interpolated, forecasted_data], ignore_index=True)

# Final check for missing values and duplicates in the combined data
missing_values_combined = combined_gcidata.isnull().sum()
print("\nMissing values in combined data:")
print(missing_values_combined[missing_values_combined > 0])

duplicates_combined = combined_gcidata.duplicated().sum()
print(f"\nNumber of duplicated rows in combined data: {duplicates_combined}")

# %%
# Standardize and scale the GCI data, excluding 'Country', 'year', and 'Country_year'
gci_data_to_scale = combined_gcidata.drop(columns=['Country', 'year', 'Country_year'])

# Apply scaling to the GCI data
scaler = StandardScaler()
gci_data_scaled = pd.DataFrame(scaler.fit_transform(gci_data_to_scale), columns=gci_data_to_scale.columns)

# Add back the 'Country', 'year', and 'Country_year' columns
gci_data_scaled[['Country', 'year', 'Country_year']] = combined_gcidata[['Country', 'year', 'Country_year']]


# %%
# Display a sample of the final scaled GCI data
print("\nSample of the final scaled GCI data:")
gci_data_scaled.head()

# %% [markdown]
# # Data Cleaning - SDG Dataset

# %%
# Data overview
sdg_data.info()
sdg_data.head()

# %%
# Drop irrelevant columns
sdg_data = sdg_data.drop(columns=['Country Code', 'Time Code'])

# Check if columns are removed successfully
sdg_data.info()
sdg_data.head()

# %%
# Replace any non-numeric values (e.g., "..") with NaN
sdg_data.replace("..", np.nan, inplace=True)

# Check the data type for the 'Country Name' column
print("Data type for the 'Country Name' column:")
print(sdg_data['Country Name'].dtype)

# Convert all columns except 'Country Name' and 'Time' to numeric
cols_to_convert = sdg_data.columns.difference(['Country Name', 'Time'])
sdg_data[cols_to_convert] = sdg_data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Coerce 'Time' column to numeric (turn invalid values to NaN)
sdg_data['Time'] = pd.to_numeric(sdg_data['Time'], errors='coerce')

# Check for missing values in the 'Time' column
print("\nMissing values in 'Time' column:")
print(sdg_data['Time'].isna().sum())

# Optionally, remove rows where 'Time' is NaN
sdg_data.dropna(subset=['Time'], inplace=True)

# Convert 'Time' column to integer after handling NaNs
sdg_data['Time'] = sdg_data['Time'].astype(int)

# Check the data again after removing rows with missing 'Time' and converting to int
print("\nData after removing rows with missing 'Time' values and converting 'Time' to int:")
sdg_data.info()
sdg_data.head()

# %%
# Check for missing values
print("Missing values:")
missing_values = sdg_data.isnull().sum()
missing_values[missing_values > 0]

# %%
# Check for duplicates
print(f"\nNumber of duplicated rows: {sdg_data.duplicated().sum()}")

# %%
# Handle missing values by dropping columns with more than 50% missing values
threshold = 0.5  # 50%
missing_percentage = sdg_data.isnull().mean()
columns_to_drop = missing_percentage[missing_percentage > threshold].index
sdg_data_cleaned = sdg_data.drop(columns=columns_to_drop)

# Display the number of columns dropped and their names
print(f"Number of columns dropped: {len(columns_to_drop)}")
print(f"Columns dropped: {(columns_to_drop)}")

# Check the data again after dropping columns with more than 50% missing values
print("\nData after dropping columns with more than 50% missing values:")
sdg_data_cleaned.info()
sdg_data_cleaned.head()

# %%
# Interpolate missing values in historical data
forecast_columns = sdg_data_cleaned.select_dtypes(include=['number']).columns

# Sort data for interpolation to occur in year order within each country
sdg_data_cleaned = sdg_data_cleaned.sort_values(['Country Name', 'Time']).reset_index(drop=True)

# Interpolate missing values for each country and column in the cleaned data
for country in sdg_data_cleaned['Country Name'].unique():
    country_data = sdg_data_cleaned[sdg_data_cleaned['Country Name'] == country]
    for column in forecast_columns:
        # Perform linear interpolation for each numeric column
        sdg_data_cleaned.loc[sdg_data_cleaned['Country Name'] == country, column] = (
            country_data[column].interpolate(method='linear', limit_direction='both')
        )

# %%
# Backfill remaining missing values
sdg_data_cleaned[forecast_columns] = sdg_data_cleaned[forecast_columns].fillna(method='bfill')

# Forecast remaining missing values for each country and column
for country in sdg_data_cleaned['Country Name'].unique():
    country_data = sdg_data_cleaned[sdg_data_cleaned['Country Name'] == country].sort_values('Time')
    for column in forecast_columns:
        if country_data[column].isnull().any():
            # Filter non-NaN values for modeling
            non_nan_data = country_data.dropna(subset=[column])
            if len(non_nan_data) > 1:  # Ensure enough data for forecasting
                model = ExponentialSmoothing(
                    non_nan_data[column],
                    trend='add', seasonal=None, initialization_method="estimated"
                ).fit()
                forecast_values = model.predict(start=country_data['Time'].min(), end=country_data['Time'].max())
                # Combine interpolated and forecasted values
                sdg_data_cleaned.loc[sdg_data_cleaned['Country Name'] == country, column] = (
                    country_data[column].combine_first(forecast_values)
                )

# Create the `Country_year` column after interpolation and backfilling
sdg_data_cleaned['Country_year'] = sdg_data_cleaned['Country Name'] + '_' + sdg_data_cleaned['Time'].astype(str)

# %%
# Check for missing values again after backfilling and forecasting
missing_values = sdg_data_cleaned.isnull().sum()

# Filter columns with missing values
missing_columns = missing_values[missing_values > 0]

# Output results
if not missing_columns.empty:
    print("Columns with missing values after backfilling and forecasting:")
    print(missing_columns)
else:
    print("No missing values in any column after backfilling and forecasting.")

# Check for duplicated values after interpolation
duplicates_values = sdg_data_cleaned.duplicated().sum()
print(f"\nNumber of duplicated rows after interpolation: {duplicates_values}")

# %%
# Drop the specific column with missing values
sdg_data_cleaned = sdg_data_cleaned.drop(columns=['Prevalence of undernourishment (% of population) [SN.ITK.DEFC.ZS]'])

# Check if columns are removed successfully
sdg_data_cleaned.info()
sdg_data_cleaned.head()

# %%
# Ensure forecast_columns only includes existing columns
forecast_columns = [col for col in forecast_columns if col in sdg_data_cleaned.columns]

# Initialize an empty DataFrame for forecasted data
forecasted_data = pd.DataFrame()
countries = sdg_data_cleaned['Country Name'].unique()

for country in countries:
    country_data = sdg_data_cleaned[sdg_data_cleaned['Country Name'] == country].sort_values('Time')
    forecast_results = {col: [] for col in forecast_columns}

    # Forecast future values for each specified column
    for column in forecast_columns:
        if not country_data[column].isnull().all():
            model = ExponentialSmoothing(
                country_data[column],
                trend='add', seasonal=None, initialization_method="estimated"
            ).fit()
            forecast = model.forecast(2022 - country_data['Time'].max())
        else:
            forecast = [np.nan] * (2022 - country_data['Time'].max())

        # Store the forecasted values for the current column
        forecast_results[column] = list(forecast)

    # Generate the forecasted rows
    forecast_years = range(country_data['Time'].max() + 1, 2023)
    for year_idx, year in enumerate(forecast_years):
        forecasted_data = pd.concat([forecasted_data, pd.DataFrame({
            'Country Name': [country],
            'Time': [year],
            **{col: [forecast_results[col][year_idx]] for col in forecast_columns},
            'Country_year': [f"{country}_{year}"]
        })], ignore_index=True)

# %%
# Verify missing values in forecasted data
missing_values_forecasted_data = forecasted_data.isnull().sum()
print("\nMissing values in forecasted data:")
print(missing_values_forecasted_data[missing_values_forecasted_data > 0])

# Check for duplicated values in forecasted data
duplicates_forecasted_data = forecasted_data.duplicated().sum()
print(f"\nNumber of duplicated rows in forecasted data: {duplicates_forecasted_data}")

# %%
# Display the interpolated data
print("Interpolated Data:")
print(sdg_data_cleaned.head())

# %%
# Display the forecasted data
forecasted_data['Time'] = forecasted_data['Time'].astype(int)
print("\nForecasted Data:")
print(forecasted_data.head())

# %%
# Combine interpolated and forecasted data
combined_sdgdata = pd.concat([sdg_data_cleaned, forecasted_data], ignore_index=True)

# Final check for missing values and duplicates in the combined data
missing_values_combined_sdg = combined_sdgdata.isnull().sum()
print("\nMissing values in combined SDG data:")
print(missing_values_combined_sdg[missing_values_combined_sdg > 0])

duplicates_combined_sdg = combined_sdgdata.duplicated().sum()
print(f"\nNumber of duplicated rows in combined SDG data: {duplicates_combined_sdg}")

# %%
# Standardize and scale the SDG data, excluding 'Country', 'year', and 'Country_year'
sdg_data_to_scale = combined_sdgdata.drop(columns=['Country Name', 'Time', 'Country_year'])

# Apply scaling to the SDG data
scaler = StandardScaler()
sdg_data_scaled = pd.DataFrame(scaler.fit_transform(sdg_data_to_scale), columns=sdg_data_to_scale.columns)

# Add back the 'Country', 'year', and 'Country_year' columns
sdg_data_scaled[['Country Name', 'Time', 'Country_year']] = combined_sdgdata[['Country Name', 'Time', 'Country_year']]


# %%
# Display a sample of the final scaled SDG data
print("\nSample of the final scaled SDG data:")
sdg_data_scaled.head()

# %%
# Step 1: Exclude the last three columns (not relevant for SDG mapping)
relevant_data = sdg_data_scaled.iloc[:, :-3]  # Keep all other columns

# Step 2: Extract column names (variables) from the relevant data
original_column_names = relevant_data.columns.tolist()

# Step 3: Clean the text in column names
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Create a mapping of original to cleaned column names
cleaned_column_names = [clean_text(col) for col in original_column_names]
column_mapping = dict(zip(cleaned_column_names, original_column_names))

# Step 4: SDG keyword dictionary
sdg_keywords = {
    1: ["poverty", "inequality", "income"],
    2: ["hunger", "food", "nutrition", "agriculture"],
    3: ["health", "well-being", "disease", "medicine"],
    4: ["education", "literacy", "learning", "schools"],
    5: ["gender", "women", "equality", "empowerment"],
    6: ["water", "sanitation", "hygiene"],
    7: ["energy", "renewable", "electricity"],
    8: ["employment", "economy", "growth", "jobs", "labor"],
    9: ["industry", "infrastructure", "innovation"],
    10: ["inequality", "social", "disparity"],
    11: ["cities", "urban", "sustainable", "habitat"],
    12: ["consumption", "production", "sustainable", "resources"],
    13: ["climate", "emissions", "environment", "carbon"],
    #14: ["ocean", "marine", "sea", "coast"],
    #15: ["forest", "biodiversity", "ecosystems", "land"],
    16: ["justice", "peace", "institutions", "law"],
    17: ["partnerships", "goals", "cooperation", "funding"]
}

# Step 5: Semantic similarity mapping
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare SDG keyword embeddings
sdg_embeddings = {}
for sdg, keywords in sdg_keywords.items():
    sdg_embeddings[sdg] = model.encode(keywords, convert_to_tensor=True)

# Map each variable to its closest SDG
def map_to_sdg(variable_name):
    variable_embedding = model.encode(variable_name, convert_to_tensor=True)
    max_similarity = 0
    mapped_sdg = None
    for sdg, embeddings in sdg_embeddings.items():
        similarity = util.cos_sim(variable_embedding, embeddings).mean().item()
        if similarity > max_similarity:
            max_similarity = similarity
            mapped_sdg = sdg
    return mapped_sdg

# Apply mapping to cleaned column names
variable_sdg_mapping = {col: map_to_sdg(col) for col in cleaned_column_names}

# Step 6: Group the variables by SDG and include the excluded columns
# Create a dictionary to hold the columns for each SDG
sdg_columns = {i: [] for i in range(1, 18)}  # SDG columns will be stored here

# Map variables to their respective SDGs
for cleaned_col, sdg in variable_sdg_mapping.items():
    original_col = column_mapping[cleaned_col]
    sdg_columns[sdg].append(original_col)

# Step 7: Perform Factor Analysis and aggregate data for each SDG
sdg_data = pd.DataFrame()

for sdg, variables in sdg_columns.items():
    if variables:  # Ensure there are variables to aggregate
        # Extract relevant variables for the SDG
        sdg_variables_data = relevant_data[variables]
        
        # Apply Factor Analysis (n_components=1 to get the main factor)
        fa = FactorAnalysis(n_components=1, random_state=42)
        sdg_factor = fa.fit_transform(sdg_variables_data)
        
        # Add the factor as a new column representing the SDG
        sdg_data[f'SDG_{sdg}'] = sdg_factor.flatten()

# Step 8: Add back the last three excluded columns
excluded_columns = sdg_data_scaled.iloc[:, -3:]  # Extract the last three columns
final_sdg_data = pd.concat([excluded_columns.reset_index(drop=True), sdg_data.reset_index(drop=True)], axis=1)

for sdg, variables in sdg_columns.items():
    print(f"SDG {sdg}: {variables}")

# Step 9: Save the final data to an Excel file
# final_sdg_data.to_excel("/Users/mediastrategy/Desktop/Final_sdg_aggregated_data_FactorAnalysis.xlsx", index=False)

# Step 9: Print a preview of the final DataFrame
print("Final Aggregated SDG Data (using Factor Analysis):")
final_sdg_data.head()

# %%
# Drop the 'SDG_16' column
final_sdg_data.drop('SDG_16', axis=1, inplace=True)

# Print a preview of the updated DataFrame
print("Final SDG Data:")
final_sdg_data.head()

# %%
# To check how many variables and which variables were mapped into each SDG 

# Prepare the data for the Excel file
sdg_variable_counts = []
sdg_variable_names = []

# Loop through each SDG and gather the data, skipping SDG 16
for sdg, variables in sdg_columns.items():
    if sdg == 16:
        continue  # Skip SDG 16 as it is dropped
    variable_count = len(variables)
    variable_names = ", ".join(variables)  # Join variable names with commas
    sdg_variable_counts.append(variable_count)
    sdg_variable_names.append(variable_names)

# Create a DataFrame for the SDG variable mapping (excluding SDG 16)
sdg_mapping_df = pd.DataFrame({
    'SDG': [f"SDG {i}" for i in range(1, 18) if i != 16],  # Exclude SDG 16
    'Number of Variables': sdg_variable_counts,
    'Variables': sdg_variable_names
})

# Display the DataFrame for preview
print("SDG Variable Mapping Data :")
sdg_mapping_df.head()

# %%
# Step 1: Perform Factor Analysis and aggregate data for each SDG with factor loadings
factor_loading_results = {}

# Perform Factor Analysis for each SDG and capture factor loadings
for sdg, variables in sdg_columns.items():
    if variables:  # Ensure there are variables to aggregate
        # Extract relevant variables for the SDG
        sdg_variables_data = relevant_data[variables]
        
        # Apply Factor Analysis (n_components=1 to get the main factor)
        fa = FactorAnalysis(n_components=1, random_state=42)
        fa.fit(sdg_variables_data)
        
        # Get factor loadings (coefficients of the variables in the factor)
        factor_loadings = fa.components_.flatten()  # 1D array of loadings
        
        # Store the factor loadings in a dictionary
        factor_loading_results[sdg] = {
            'variables': variables,
            'factor_loadings': factor_loadings
        }

# Step 2: Display factor loadings for each SDG
for sdg, result in factor_loading_results.items():
    print(f"\nFactor Loadings for SDG {sdg}:")
    
    for var, loading in zip(result['variables'], result['factor_loadings']):
        print(f"Variable: {var}, Factor Loading: {loading:.4f}")

# Step 3: Optionally, save the factor loadings to an Excel file
factor_loading_df = []

for sdg, result in factor_loading_results.items():
    for var, loading in zip(result['variables'], result['factor_loadings']):
        factor_loading_df.append({
            'SDG': f"SDG {sdg}",
            'Variable': var,
            'Factor Loading': loading
        })

# Convert the results to a DataFrame
factor_loading_df = pd.DataFrame(factor_loading_df)

# Save to Excel
factor_loading_df.to_excel(r"C:\Users\alkoj\OneDrive\Desktop\Jas2\ResultsSDG_Factor_Loadings.xlsx", index=False)

# Print a preview of the factor loadings DataFrame
print("\nFactor Loadings Data (saved to 'SDG_Factor_Loadings.xlsx'):")
factor_loading_df.head()


# %%
# Merge the scaled SDG and GCI datasets
merged_data = pd.merge(final_sdg_data, gci_data_scaled, on='Country_year', how='inner')

# Check the merged data
print("Merged Data Overview:")
merged_data.info()
merged_data.head()

# %%
# Define dependent and independent variables
dependent_vars = [
    'Cooperation_Labor_Employer', 'Wage_Flexibility', 'Hiring_Firing_Practices',
    'Redundancy_Costs', 'Taxation_Incentives_Work', 'Pay_Productivity',
    'Professional_Management', 'Talent_Retention', 'Talent_Attraction',
    'Women_Labor_Ratio'
]
independent_vars = [
    'SDG_1', 'SDG_2', 'SDG_3', 'SDG_4', 'SDG_5', 'SDG_6', 'SDG_7', 
    'SDG_8', 'SDG_9', 'SDG_11', 'SDG_12', 'SDG_13', 'SDG_17'
]

# %%
# Subset the data for regression, including 'Country' and 'year'
regression_data = merged_data[['Country', 'year', 'Country_year' ] + dependent_vars + independent_vars].dropna()

# Ensure 'Country' values are properly formatted
regression_data['Country'] = regression_data['Country'].astype(str).str.strip()

# Debug: Check unique values in 'Country'
print("Unique countries:", regression_data['Country'].unique())

# %%
# One-hot encode the 'Country' variable
encoder = OneHotEncoder(drop='first', sparse_output=False)
country_dummies = encoder.fit_transform(regression_data[['Country']])

# Create a DataFrame for the encoded countries
country_dummies_df = pd.DataFrame(
    country_dummies,
    columns=encoder.get_feature_names_out(['Country']),  # Generate column names
    index=regression_data.index  # Align with the regression_data DataFrame
)

# Add 'Country' and 'year' columns to country_dummies_df for context
country_dummies_df['Country'] = regression_data['Country']
country_dummies_df['year'] = regression_data['year']

# Display the head of the DataFrame to ensure correctness
print("Head of the one-hot encoded country dummies DataFrame:")
country_dummies_df.head()


# %%
country_dummies_df.describe()

# %%
# Calculate correlations
numeric_country_dummies_df = country_dummies_df.drop(columns=['Country', 'year'])
correlation_matrix = numeric_country_dummies_df.corr()

# Dynamically set figure size based on the number of countries
num_features = len(correlation_matrix.columns)
plt.figure(figsize=(num_features * 0.5, num_features * 0.5))  # Scale with the number of features

sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap='coolwarm',
    xticklabels=correlation_matrix.columns,
    yticklabels=correlation_matrix.columns
)

# Rotate x-axis labels to avoid overlap
plt.xticks(rotation=45, ha='right')  # Adjust angle and alignment
plt.yticks(rotation=0)              # Keep y-axis labels horizontal
plt.title('Correlation Heatmap of Country Dummies')
plt.tight_layout()
plt.show()

# %%
# Ensure 'Country_year' is created in dummies data frame
country_dummies_df['Country_year'] = country_dummies_df['Country'].astype(str) + '_' + country_dummies_df['year'].astype(str)

# Merge the data
regression_data = pd.merge(regression_data, country_dummies_df, on='Country_year', how='left')

# Check the final merged data
print("Head of the final regression data:")
regression_data.head()

# %%
# Check the data types of each column in the regression_data DataFrame
for col in regression_data.columns:
    print(f"Column '{col}': {regression_data[col].dtype}")

# %%
# Create a copy of the DataFrame to avoid modifying the original
heatmap_data = regression_data.copy()

# Combine 'country' and 'year' into a single column
heatmap_data['Country_year'] = heatmap_data['Country_x'].astype(str) + '_' + heatmap_data['year_x'].astype(str)

# Set figure size dynamically based on the number of features
num_features = heatmap_data.shape[1]
plt.figure(figsize=(min(20, num_features / 2), 30))  # Adjust width based on the number of features

# Create a heatmap of missing values
sns.heatmap(
    heatmap_data.isnull(),
    cbar=False,
    yticklabels=heatmap_data['Country_year'], 
    xticklabels=True,  # Ensure features are displayed
    cmap='viridis'
)

# Add plot labels and title
plt.title('Missing Values Heatmap', fontsize=16)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Country_Year', fontsize=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=90, fontsize=10)

# Show the plot
plt.tight_layout()  # Ensure everything fits within the figure
plt.show()

# %%
# Calculate the correlation matrix for numeric columns
correlation_matrix = regression_data.select_dtypes(include=np.number).corr(method='pearson', min_periods=1)

# Display the correlation matrix
print("Correlation Matrix:")
correlation_matrix


# %%
# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))  # Adjust figure size as needed
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap='coolwarm', 
    cbar=True, 
    square=True,
    linewidths=0.5
)
plt.title("Correlation Matrix Heatmap")
plt.show()

# %%
# 1. Overview of data
print("Merged Dataset Overview")
merged_data.info()

print("\nSummary Statistics for Merged Dataset:")
merged_data.describe()


# %%
# 2. Check for missing values
print("\nMissing Values in Merged Dataset:")
merged_data.isnull().sum()

# %%
# 3. Correlation Matrix for Numerical Features 
print("\nCorrelation Matrix for Numerical Features:")

# Exclude non-relevant columns like 'Time' and 'year' from the correlation matrix
columns_to_exclude = ['Time', 'year']
filtered_data = merged_data.drop(columns=columns_to_exclude)

# Select only numeric columns for the correlation matrix
numeric_columns = filtered_data.select_dtypes(include=[np.number])
correlation_matrix = numeric_columns.corr()

# Mask the upper triangle to avoid repeating the correlations
# mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create heatmap with correlation values annotated
plt.figure(figsize=(14, 10))  # Adjust the figure size to make more space
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', cbar=True, fmt=".2f", linewidths=0.5, annot_kws={'size': 9})

# Rotate the x and y axis labels for better readability
plt.xticks(rotation=90, ha='right', fontsize=10)
plt.yticks(rotation=0, ha='right', fontsize=10)

# Set the title and adjust layout
plt.title("Correlation Matrix", fontsize=14)
plt.tight_layout()

plt.show()

# %%
# 4. Distribution of Key Features 
numerical_columns = merged_data.select_dtypes(include='number').columns

# Exclude 'Time' and 'year' columns
columns_to_exclude = ['Time', 'year']
numerical_columns = [col for col in numerical_columns if col not in columns_to_exclude]

# Plot distributions for the remaining numerical columns
for column in numerical_columns:
    sns.histplot(merged_data[column], kde=True, bins=30, color='blue')
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

# %%
# Get numerical columns, excluding 'Time' and 'year'
numerical_columns = merged_data.select_dtypes(include='number').columns
excluded_columns = ['Time', 'year']  # Columns to exclude
numerical_columns = [col for col in numerical_columns if col not in excluded_columns]

# Set up the grid for subplots (calculate rows and columns to fit all histograms)
num_columns = len(numerical_columns)
columns_per_row = 4  # Adjust this number for layout
num_rows = math.ceil(num_columns / columns_per_row)

# Create the subplots
fig, axes = plt.subplots(num_rows, columns_per_row, figsize=(20, num_rows * 5))  # Adjust figure size as needed
axes = axes.flatten()  # Flatten axes array for easy iteration

# Loop through each column and plot its histogram
for i, column in enumerate(numerical_columns):
    sns.histplot(merged_data[column], kde=True, bins=30, color='blue', ax=axes[i])
    axes[i].set_title(f"Distribution of {column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Frequency")

# Turn off unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

# Add overall title
plt.suptitle("Distributions of Numerical Indicators (Excluding 'Time' and 'year')", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# %%
# List of SDG columns (from SDG_1 to SDG_17)
sdg_columns = ['SDG_1', 'SDG_2', 'SDG_3', 'SDG_4', 'SDG_5', 'SDG_6', 'SDG_7', 'SDG_8', 'SDG_9', 'SDG_11', 'SDG_12', 'SDG_13', 'SDG_17']

# Group the data by 'Time' (or any time period like year) and calculate the mean for each SDG
time_grouped_sdg_data = merged_data.groupby('Time')[sdg_columns].mean()

# Create line plots for each SDG
plt.figure(figsize=(14, 10))

for sdg in sdg_columns:
    plt.figure(figsize=(10, 6))
    plt.plot(time_grouped_sdg_data.index, time_grouped_sdg_data[sdg], label=sdg, color='blue')
    plt.title(f"{sdg} over Time", fontsize=14)
    plt.xlabel("Time (Year)", fontsize=12)
    plt.ylabel(f"{sdg} Value", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
# List of HR performance metrics
hr_performance_columns = ['Cooperation_Labor_Employer', 'Wage_Flexibility', 'Hiring_Firing_Practices', 'Redundancy_Costs', 
                           'Taxation_Incentives_Work', 'Pay_Productivity', 'Professional_Management', 'Talent_Retention', 
                           'Talent_Attraction', 'Women_Labor_Ratio']

# Group the data by 'Time' (or any time period like year) and calculate the mean for each HR metric
time_grouped_data = merged_data.groupby('Time')[hr_performance_columns].mean()

# Create line plots for each HR performance metric
plt.figure(figsize=(12, 8))

for metric in hr_performance_columns:
    plt.figure(figsize=(10, 6))
    plt.plot(time_grouped_data.index, time_grouped_data[metric], label=metric, color='blue')
    plt.title(f"{metric} over Time", fontsize=14)
    plt.xlabel("Time (Year)", fontsize=12)
    plt.ylabel(f"{metric} Value", fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
# Select the relevant subset of the data (SDG and HR performance metrics)
selected_columns = sdg_columns + hr_performance_columns
subset_data = merged_data[selected_columns]

# Define a list of IVs and DVs 
ivs = sdg_columns  
dvs = hr_performance_columns  

# Loop through each combination of IV and DV and create individual scatter plots
for iv in ivs:
    for dv in dvs:
        # Create a scatter plot between each IV and DV
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=subset_data, x=iv, y=dv, alpha=0.6)
        
        # Set the title and labels
        plt.title(f'Scatter plot: {iv} vs {dv}', fontsize=14)
        plt.xlabel(iv, fontsize=12)
        plt.ylabel(dv, fontsize=12)
        
        # Show the plot
        plt.tight_layout()
        plt.show()

# %%
# Group by 'Country' and calculate the mean for each SDG indicator
country_mean = merged_data.groupby('Country')[hr_performance_columns].mean()

# Set up the plotting style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Loop through each SDG indicator and plot top 3-5 countries
for indicator in hr_performance_columns:
    # Sort countries by the indicator and select the top 5
    top_countries = country_mean[indicator].sort_values(ascending=False).head(5)
    
    # Plot a horizontal bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=top_countries.values,
        y=top_countries.index,
        palette="viridis"
    )
    plt.title(f"Top 5 Countries for {indicator}", fontsize=14)
    plt.xlabel("Average Score", fontsize=12)
    plt.ylabel("Country", fontsize=12)
    plt.tight_layout()
    plt.show()

# %%
# Group by 'Country' and calculate the mean for each SDG indicator
country_mean = merged_data.groupby('Country')[sdg_columns].mean()

# Set up the plotting style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# Loop through each SDG indicator and plot top 3-5 countries
for indicator in sdg_columns:
    # Sort countries by the indicator and select the top 5
    top_countries = country_mean[indicator].sort_values(ascending=False).head(5)
    
    # Plot a horizontal bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=top_countries.values,
        y=top_countries.index,
        palette="viridis"
    )
    plt.title(f"Top 5 Countries for {indicator}", fontsize=14)
    plt.xlabel("Average Score", fontsize=12)
    plt.ylabel("Country", fontsize=12)
    plt.tight_layout()
    plt.show()

# %%
# Define the dependent variables
dependent_vars = [
    'Cooperation_Labor_Employer', 'Wage_Flexibility', 'Hiring_Firing_Practices',
    'Redundancy_Costs', 'Taxation_Incentives_Work', 'Pay_Productivity',
    'Professional_Management', 'Talent_Retention', 'Talent_Attraction',
    'Women_Labor_Ratio'
]

# Define non-numeric columns to remove
non_numeric_cols = ['country_x', 'year_x', 'country_y', 'year_y', 'country_year']

# Create the all_columns list by extracting all column names from regression_data
all_columns = [col for col in regression_data.columns if col not in dependent_vars + non_numeric_cols]

# Create X using the modified list of columns
X = regression_data[all_columns].reset_index(drop=True)

# Dependent Variables (HR Performance Metrics)
y = regression_data[dependent_vars].reset_index(drop=True)

# Split Data into Train and Test (Using regression_data)
train_data = regression_data[regression_data['year_x'].between(2018, 2021)]
test_data = regression_data[regression_data['year_x'] == 2022]

# Features (Independent Variables) for Model Training
X_train = train_data[all_columns].select_dtypes(include=[np.number])  # Only numeric columns
X_test = test_data[all_columns].select_dtypes(include=[np.number])    # Only numeric columns

# Ensure Target Columns (HR Performance Metrics) Are Numeric
for target in dependent_vars:
    if train_data[target].dtype == 'object':  # If the target is categorical
        train_data[target] = train_data[target].astype('category').cat.codes
        test_data[target] = test_data[target].astype('category').cat.codes

# Define Models
models = {
    'Linear Regression': LinearRegression(),
    'Support Vector Machine': SVR(kernel='linear'),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Store Results
results_summary = []

# Train and Evaluate Models
for target in dependent_vars:
    print(f"\n--- Evaluating Models for Target: {target} ---")
    Y_train = train_data[target]
    Y_test = test_data[target]

    for name, model in models.items():
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        # Metrics Calculation
        mse = mean_squared_error(Y_test, Y_pred)
        mae = mean_absolute_error(Y_test, Y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(Y_test, Y_pred)

        # Store Results
        results_summary.append({
            'Model': name,
            'Target': target,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        })

# Convert Results to DataFrame
results_df = pd.DataFrame(results_summary)
print("\nModel Performance Summary:")
results_df

# %%
# Define the list of SDG features
sdg_features = [
    'SDG_1', 'SDG_2', 'SDG_3', 'SDG_4', 'SDG_5', 'SDG_6', 'SDG_7',
    'SDG_8', 'SDG_9', 'SDG_11', 'SDG_12', 'SDG_13', 'SDG_17'
]

# Filter X_train to include only the SDG features
X_train_sdg = X_train[sdg_features]

# Extract coefficients for Linear Regression
coefficients_summary = []  # List to store coefficients

for target in dependent_vars:
    print(f"\n--- Extracting Coefficients for Target: {target} ---")
    
    # Fit Linear Regression Model using only SDG features
    model = models['Linear Regression']
    model.fit(X_train_sdg, train_data[target])  # Fit the model to the filtered data
    
    # Get the coefficients
    coefficients = model.coef_
    feature_names = X_train_sdg.columns
    
    # Store coefficients in a list for later analysis
    for feature, coef in zip(feature_names, coefficients):
        coefficients_summary.append({
            'Target': target,
            'Feature': feature,
            'Coefficient': coef
        })

# Convert coefficients to DataFrame for better readability
coefficients_df = pd.DataFrame(coefficients_summary)

# Display the coefficients for linear regression
print("\nLinear Regression Coefficients Summary:")
print(coefficients_df)

# Optionally, save the coefficients to an Excel file
coefficients_df.to_excel(r"C:\Users\alkoj\OneDrive\Desktop\Jas2\Linear_Regression_Coefficients_SDG.xlsx", sheet_name="Coefficients", index=False)

# %%
# Assuming ⁠ results_df ⁠ contains the model evaluation results as shown

# Pivot the results to create a matrix where rows are models and columns are metrics
pivot_df = results_df.pivot_table(index='Model', columns='Target', values=['MAE', 'MSE', 'RMSE', 'R²'])

# Set up the matplotlib figure with a larger size to avoid overlapping
plt.figure(figsize=(16, 12))

# Loop through each metric and plot
metrics = ['MAE', 'MSE', 'RMSE', 'R²']

for i, metric in enumerate(metrics):
    plt.subplot(2, 2, i + 1)
    sns.heatmap(pivot_df[metric], annot=True, cmap='coolwarm', fmt=".4f", 
                linewidths=0.5, cbar_kws={'label': metric}, annot_kws={'size': 8})
    plt.title(f'{metric} for Each Model')
    plt.xlabel('Target', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    
    # Rotate x and y axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0, ha='right')

# Display the heatmap
plt.tight_layout()
plt.show()

# %%
# Define the list of SDG features
sdg_features = [
    'SDG_1', 'SDG_2', 'SDG_3', 'SDG_4', 'SDG_5', 'SDG_6', 'SDG_7',
    'SDG_8', 'SDG_9', 'SDG_11', 'SDG_12', 'SDG_13', 'SDG_17'
]

# Filter X_train to include only the SDG features
X_train_sdg = X_train[sdg_features]

# Ensure models is a dictionary with a 'Linear Regression' entry
models = {
    'Linear Regression': LinearRegression(),
    # Add other models if necessary
}

# Extract coefficients for Linear Regression
coefficients_summary = []  # List to store coefficients

for target in dependent_vars:
    print(f"\n--- Extracting Coefficients for Target: {target} ---")
    
    # Fit Linear Regression Model using only SDG features
    model = models['Linear Regression']  # Accessing the model correctly
    model.fit(X_train_sdg, train_data[target])  # Fit the model to the filtered data
    
    # Get the coefficients
    coefficients = model.coef_
    feature_names = X_train_sdg.columns
    
    # Store coefficients in a list for later analysis
    for feature, coef in zip(feature_names, coefficients):
        coefficients_summary.append({
            'Target': target,
            'Feature': feature,
            'Coefficient': coef
        })

# Convert coefficients to DataFrame for better readability
coefficients_df = pd.DataFrame(coefficients_summary)

# Display the coefficients for linear regression
print("\nLinear Regression Coefficients Summary:")
print(coefficients_df)

# Optionally, save the coefficients to an Excel file
coefficients_df.to_excel("Linear_Regression_Coefficients_SDG.xlsx", sheet_name="Coefficients", index=False)

# %%
# Filter X_train and X_test to include only the specified SDGs
X_train_filtered = X_train[sdg_features]
X_test_filtered = X_test[sdg_features]

# Using Random Forest as the best model
selected_model = RandomForestRegressor(random_state=42)

# Store results in a structured format for saving
feature_importance_results = []

for target in dependent_vars:
    Y_train = train_data[target]
    selected_model.fit(X_train_filtered, Y_train)

    feature_importances = selected_model.feature_importances_
    sdg_importance = dict(zip(X_train_filtered.columns, feature_importances))
    
    # Append importance data for each target variable
    print(f"\nFeature Importance for {target}:")
    for sdg, importance in sdg_importance.items():
        print(f"{sdg}: {importance:.4f}")

        # Store results for heatmap
        feature_importance_results.append({
            'Target': target,
            'SDG': sdg,
            'Importance': importance
        })

# Convert to DataFrame
feature_importance_df = pd.DataFrame(feature_importance_results)

# Save to Excel
feature_importance_df.to_excel(r"C:\Users\alkoj\OneDrive\Desktop\Jas2\Feature_Importances.xlsx", sheet_name="Feature Importance", index=False)

print("\nFeature importance saved to 'Feature_Importances.xlsx'.")


# %%
# Pivot the table to get SDGs as rows and HR metrics as columns
pivot_table = feature_importance_df.pivot(index='SDG', columns='Target', values='Importance')

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(
    pivot_table,
    annot=True,            # Display values in the heatmap
    fmt='.4f',             # Four decimal places for values
    cmap='coolwarm',       # Color scheme
    cbar_kws={'label': 'Feature Importance'},  # Add color bar label
    linewidths=0.5         # Add gridlines
)

# Add titles and labels
plt.title('Feature Importance Heatmap (SDGs Across HR Metrics)', fontsize=16)
plt.xlabel('HR Metric', fontsize=12)
plt.ylabel('SDG', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()

# Display the heatmap
plt.show()

# Save the pivot table to an Excel file for further analysis
pivot_table.to_excel("Feature_Importance_Heatmap.xlsx", sheet_name="Feature Importance", index=True)

print("\nFeature importance heatmap saved to 'Feature_Importance_Heatmap.xlsx'.")

# Load pre-saved outputs
@st.cache_data
def load_data():
    # Replace these file paths with the actual locations of your outputs
    factor_loadings = pd.read_excel(r"C:\Users\alkoj\OneDrive\Desktop\Jas2\ResultsSDG_Factor_Loadings.xlsx")
    regression_coefficients = pd.read_excel(r"C:\Users\alkoj\OneDrive\Desktop\Jas2\Linear_Regression_Coefficients_SDG.xlsx")
    feature_importance = pd.read_excel(r"C:\Users\alkoj\OneDrive\Desktop\Jas2\Feature_Importances.xlsx")
    return factor_loadings, regression_coefficients, feature_importance, merged_data

# Load data
factor_loadings, regression_coefficients, feature_importance, merged_data = load_data()

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
