import pandas as pd
import matplotlib.pyplot as plt

file_path = '../spotpriser.csv'
df = pd.read_csv(file_path)

df['hour'] = df['Dato/klokkeslett'].str.extract(r'Kl\. (\d{2})').astype(int)

average_prices_no2 = df.groupby('hour')['NO2'].mean()
average_prices_no2.loc[24] = average_prices_no2.loc[23]

plt.figure(figsize=(10, 6))
plt.step(average_prices_no2.index, average_prices_no2.values, where='post', color='#ff9999', label='Average Price', linewidth='3')
plt.title('Average Electricity Price per Hour')
plt.ylabel('Average Price (NOK/kWh)')
plt.xlabel('Hour of Day')

plt.xticks(range(25))
plt.xlim(0,24)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()