import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filepath_metData = 'timeseries_met_data_202409050822.csv'
filepath_impExp = 'energy_import_export.csv'
filepath_consum = 'solar_self_consumption_main_building.csv'

met_data = pd.read_csv(filepath_metData)
impExp_dat = pd.read_csv(filepath_impExp)
consum_data  = pd.read_csv(filepath_consum)

#print(impExp_dat['Måler-Id'].value_counts())

# Remove the insignificant EXPORT columns from the building without solar panels
impExp_dat = impExp_dat[~((impExp_dat['Retning'] == 'EXPORT') & (impExp_dat['Måler-Id'] == 707057500085390523))]
impExp_dat = impExp_dat.drop_duplicates()

#print(impExp_dat['Måler-Id'].value_counts())

'''df_main = impExp_dat[impExp_dat['Måler-Id'] == 707057500042745649]
df_main.to_csv('visualization/new_datasets/main.csv', index=False)

df_diff = impExp_dat[impExp_dat['Måler-Id'] == 707057500038344962]
df_diff.to_csv('visualization/new_datasets/diff.csv', index=False)'''

df_noExport = impExp_dat[impExp_dat['Retning'] != 'EXPORT']
sumImport = df_noExport.groupby('Måler-Id')['Verdi'].sum().reset_index()
sumImport.rename(columns={'Verdi': 'Total_consumption', 'Måler-Id': 'Building_Id'}, inplace=True)

# Rename meters to building Ids
sumImport['Building_Id'] = sumImport['Building_Id'].replace({
    707057500042745649: '10724',
    707057500038344962: '10703',
    707057500085390523: '4462',
    707057500042201572: '4746'
    })

sumsScaled = { 
        '4462': 292527.46/1095,
        '10724 (Solar)': 325692.39/1199,
        '10703': 345302.86/1167,
        '4746': 446010.64/1384}

print(sumsScaled)

# Create the bar chart
fig, ax = plt.subplots(figsize=(9, 8))

# Adjust the width to bring the bars closer together
bar_width = 0.6  # Set a smaller width for the bars
bars = ax.bar(sumsScaled.keys(), sumsScaled.values(), 
               color=["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"], 
               edgecolor='black',  # Set the edge color for outlines
               linewidth=1.5,      # Width of the outline
               width=bar_width)

# Add labels and title
ax.set_xlabel("Building Ids", fontsize=14)
ax.set_ylabel("Energy consumption per m²", fontsize=14)
ax.set_title("Total Energy Consumption for Buildings per Square Meter", fontsize=16)

# Add grid lines for better readability
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

# Add value labels on top of the bars
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',  # Format value to 2 decimal places
                xy=(bar.get_x() + bar.get_width() / 2, height),  # Position above the bar
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)


# Show the plot
plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
plt.show()

'''# Convert to datetime format
df['time_column'] = pd.to_datetime(df['time_column'])

# Format to desired output
df['formatted_time'] = df['time_column'].dt.strftime('%Y-%m-%d %H:%M')'''