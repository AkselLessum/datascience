import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the spot prices data
spot_prices_file = '../spotpriser.csv'
spot_df = pd.read_csv(spot_prices_file)

# Extract timestamp from 'Dato/klokkeslett' and set it as the index
spot_df['timestamp'] = pd.to_datetime(
    spot_df['Dato/klokkeslett'].str.replace("Kl. ", "").str.slice(stop=-3), 
    format='%Y-%m-%d %H'
)
spot_prices_no2 = spot_df[['timestamp', 'NO2']].set_index('timestamp')

# Load the predictions data
predictions_file = '../ml/new_datasets/combined_with_predictions.csv'
pred_df = pd.read_csv(predictions_file)

# Filter predictions for the specific houses
houses = [4462, 10703, 4746]
pred_df = pred_df[pred_df['property_id'].isin(houses)]

# Convert prediction timestamps to datetime and set as index
pred_df['Tidspunkt'] = pd.to_datetime(pred_df['Tidspunkt'])

# print(pred_df)
# print(spot_prices_no2)
# Merge predictions with spot prices on exact timestamps
merged_df = pd.merge(pred_df, spot_prices_no2, left_on='Tidspunkt', right_index=True, how='left')
# print(merged_df)

# Calculate cost savings for each house
merged_df['solar_cost'] = merged_df['solar_consumption'] * merged_df['NO2']
merged_df['total_cost'] = merged_df['Verdi'] * merged_df['NO2']
merged_df['savings'] = merged_df['total_cost'] - merged_df['solar_cost']

# Plotting with specific timestamps on the x-axis
plt.figure(figsize=(10, 6))

# Define color mapping for each building
buildingColors = {
    4462: "#ff9999",
    10703: "#99ff99",
    4746: "#ffcc99"
}

for house in houses:
    # Filter for each house and calculate cumulative savings
    house_df = merged_df[merged_df['property_id'] == house].copy()
    house_df['cumulative_savings'] = house_df['savings'].cumsum()  # Cumulative sum of savings
    
    # Plot cumulative savings
    plt.plot(house_df['Tidspunkt'], house_df['cumulative_savings'], 
             label=f'House {house}',
             color=buildingColors.get(house, '#333333'))

plt.title('Cumulative savings by using solar consumption')
plt.ylabel('Cumulative savings (NOK)')
plt.xlabel('Timestamp')
# Set the time range for the plot
start_date = pd.to_datetime("2023-07-01 09:00")
end_date = pd.to_datetime("2024-09-04 23:00")
plt.xlim(start_date,end_date)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
