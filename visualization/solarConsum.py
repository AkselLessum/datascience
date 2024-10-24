import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_impExp = pd.read_csv('energy_import_export.csv')
df_solarConsum = pd.read_csv('solar_self_consumption_main_building.csv')

# Remove all exports to only get consumption data
df_impExp = df_impExp[df_impExp['Retning'] != 'EXPORT']
# Remove unecessary columns
df_impExp = df_impExp.drop(['Energikilde', 'Retning'], axis=1)
# Reformat date and set as index
df_impExp['Tidspunkt'] = pd.to_datetime(df_impExp['Tidspunkt'])
df_impExp.set_index('Tidspunkt', inplace=True)

df_solarConsum = df_solarConsum.rename(columns={'starting_at': 'Tidspunkt'})
df_solarConsum['Tidspunkt'] = pd.to_datetime(df_solarConsum['Tidspunkt'], format='mixed')

# Split into different dfs for each building
df_solar = df_impExp[df_impExp['Måler-Id'] == 707057500042745649]

# Create new dataset with only the main building consumption
# Merge with solar consumption to get total consumption and solar consumption (summed for each day)
merged_df = pd.merge(df_solarConsum, df_solar, on='Tidspunkt', how='inner')
merged_df.set_index('Tidspunkt', inplace=True)
merged_df.to_csv('visualization/new_datasets/mergedSolar.csv')
merged_df = merged_df[['solar_consumption', 'Verdi']].resample('D').sum()
merged_df['total_consum'] = merged_df['Verdi'] + merged_df['solar_consumption']
merged_df = merged_df[merged_df['total_consum'] >= 80]
merged_df.to_csv('visualization/new_datasets/mergedSolarSummed.csv')

plt.figure(figsize=(10, 6))
plt.title('Solar consumption and total consumption')
plt.ylabel('KW')
plt.xlabel('Day')
plt.plot(merged_df.index, merged_df['solar_consumption'].values, color='#FFEB3B', label='Solar consumption')
plt.plot(merged_df.index, merged_df['total_consum'].values, color='#81C784', label='Total consumption')
plt.grid(True)
plt.legend()
plt.tight_layout()  # Adjust layout to ensure everything fits
plt.show()