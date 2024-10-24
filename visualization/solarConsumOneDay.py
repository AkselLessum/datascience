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
# Get one specific date
day = '2024-06-21'
dayData = merged_df.loc[day].copy()
dayData['total_consum'] = dayData['Verdi'] + dayData['solar_consumption']
dayData.to_csv('visualization/new_datasets/mergedSolarSummedOneDay.csv')

plt.figure(figsize=(10, 6))
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
plt.title('Imported energy vs. total consumption with solar (2024-06-21)')
plt.ylabel('KW')
plt.xlabel('Hour')
plt.plot(dayData.index, dayData['Verdi'].values, color='purple', label='Consumption (without solar)', linewidth=2)
plt.plot(dayData.index, dayData['total_consum'].values, color='#FF8C00', label='Consumption (with solar)', linewidth=2)
plt.fill_between(dayData.index, dayData['Verdi'], dayData['total_consum'], 
                 where=(dayData['Verdi'] < dayData['total_consum']), 
                 color='#FFD700', alpha=0.2, label='Solar production difference')
plt.grid(True)
plt.legend()
plt.tight_layout()  # Adjust layout to ensure everything fits
plt.show()