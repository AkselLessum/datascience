import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_impExp = pd.read_csv('energy_import_export.csv')

# Remove all exports to only get consumption data
df_impExp = df_impExp[df_impExp['Retning'] != 'EXPORT']
# Remove unecessary columns
df_impExp = df_impExp.drop(['Energikilde', 'Retning'], axis=1)
# Reformat date and set as index
df_impExp['Tidspunkt'] = pd.to_datetime(df_impExp['Tidspunkt'])
df_impExp.set_index('Tidspunkt', inplace=True)

# Split into different dfs for each building
df_solar = df_impExp[df_impExp['Måler-Id'] == 707057500042745649]
df_2 = df_impExp[df_impExp['Måler-Id'] == 707057500038344962]
df_3 = df_impExp[df_impExp['Måler-Id'] == 707057500085390523]
df_4 = df_impExp[df_impExp['Måler-Id'] == 707057500042201572]

# Group each df by month
m_peaks_solar = df_solar['Verdi'].resample('ME').max()
m_peaks_2 = df_2['Verdi'].resample('ME').max()
m_peaks_3 = df_3['Verdi'].resample('ME').max()
m_peaks_4 = df_4['Verdi'].resample('ME').max()

print(m_peaks_solar)
print(m_peaks_2)
print(m_peaks_3)
print(m_peaks_4)

total_consumption = {
    'Building 10724 (solar)': df_solar['Verdi'].sum(),
    'Building 4746': df_2['Verdi'].sum(),
    'Building 4462': df_3['Verdi'].sum(),
    'Building 10703': df_4['Verdi'].sum()
}

# Print the total consumption for each building
print("Total Consumption for Each Building (KW):")
for building, consumption in total_consumption.items():
    print(f"{building}: {consumption:.2f} KW")

# Plot the montly KW maxes
plt.figure(figsize=(10, 6))  # Set figure size
plt.title('Monthly peak energy consumption (KW)')
plt.ylabel('KW')
plt.xlabel('Month')
plt.plot(m_peaks_solar.index, m_peaks_solar.values, marker='o', color='#66b3ff', label='Building 10724 (solar)')
plt.plot(m_peaks_2.index, m_peaks_2.values, marker='o', color='#99ff99', label='Building 10703')
plt.plot(m_peaks_3.index, m_peaks_3.values, marker='o', color='#ff9999', label='Building 4462')
plt.plot(m_peaks_4.index, m_peaks_4.values, marker='o', color='#ffcc99', label='Building 4746')

# Add grid and legend
plt.grid(True)
plt.legend()

# Show the plot
plt.tight_layout()  # Adjust layout to ensure everything fits
plt.show()

