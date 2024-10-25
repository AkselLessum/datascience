import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Pre processing data and mergin all into one file
df_impExp = pd.read_csv('energy_import_export.csv')
df_solarConsum = pd.read_csv('solar_self_consumption_main_building.csv')
met_data = pd.read_csv('timeseries_met_data_202409050822.csv')

# Remove all exports from non solar building
df_impExp = df_impExp[~((df_impExp['Retning'] == 'EXPORT') & (df_impExp['Måler-Id'] == 707057500085390523))]
# Make export values into negative values to signify export
condition = df_impExp['Retning'] == 'EXPORT'
df_impExp.loc[condition, 'Verdi'] = -df_impExp.loc[condition, 'Verdi']
# Clean impExp data and reformat for easier merge with other datasets
df_impExp = df_impExp.drop(['Energikilde', 'Retning', 'Målernavn'], axis=1)
df_impExp.rename(columns={'Måler-Id': 'property_id'}, inplace=True)
df_impExp['property_id'] = df_impExp['property_id'].replace({
    707057500042745649: 10724,
    707057500038344962: 10703,
    707057500085390523: 4462,
    707057500042201572: 4746
    })
df_impExp['Tidspunkt'] = pd.to_datetime(df_impExp['Tidspunkt']).dt.tz_localize(None)

# Reformat weather data datetimet o fit with impExp
met_data.rename(columns={'starting_at': 'Tidspunkt'}, inplace=True)
met_data['Tidspunkt'] = pd.to_datetime(met_data['Tidspunkt']).dt.tz_localize(None)

# Merge met_data and impExp
merged_first = pd.merge(df_impExp, met_data, on=['property_id', 'Tidspunkt'], how='inner')

# Reformat weather in solarconsum
df_solarConsum.rename(columns={'starting_at': 'Tidspunkt'}, inplace=True)
df_solarConsum['Tidspunkt'] = pd.to_datetime(df_solarConsum['Tidspunkt'], format='mixed').dt.tz_localize(None)

# Merge solarconsum
df_merged = pd.merge(merged_first, df_solarConsum, on=['Tidspunkt'], how='inner')
df_merged.set_index('Tidspunkt', inplace=True)

# Clean up merged df
df_merged = df_merged.drop(['Unnamed: 0'], axis=1) #Leftover index column
# Set solar consumption to 0 for all rows that are not the solar building
condition = df_merged['property_id'] != 10724
df_merged.loc[condition, 'solar_consumption'] = 0

# Saved merged and cleaned df
df_merged.to_csv('ml/new_datasets/allMergedCleaned.csv')


