import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error

df = pd.read_csv('ml/new_datasets/allMergedCleaned.csv')

# Split day into its component sections
df['Tidspunkt'] = pd.to_datetime(df['Tidspunkt'])
df['hour'] = df['Tidspunkt'].dt.hour
df['day'] = df['Tidspunkt'].dt.day
df['month'] = df['Tidspunkt'].dt.month

df.loc[df['property_id'] == 10724, 'Verdi'] = (
    df.loc[df['property_id'] == 10724, 'Verdi'] +
    df.loc[df['property_id'] == 10724, 'solar_consumption']
)


# Split datasets into individual buildings
df_4462 = df[df['property_id'] == 4462]
df_4746 = df[df['property_id'] == 4746]
df_10703 = df[df['property_id'] == 10703]
df_10724 = df[df['property_id'] == 10724]

# Create lag features to create temporal understanding
lag_features = ['temperature', 'cloud_fraction', 'precipitation']
dfList = [df_4462, df_4746, df_10703, df_10724]

for df in dfList:
    for feature in lag_features:
        for lag in range(1, 10):
            df[f'lag_{feature}_{lag}'] = df[feature].shift(lag)
    df.dropna(inplace=True)

# Split solar building data into train/test with x and y split
train_features = list(df_4462)
train_features.remove('Tidspunkt')
train_features.remove('Verdi')
train_features.remove('property_id')
train_features.remove('solar_consumption')

x_10724 = df_10724[train_features]
y_10724 = df_10724['solar_consumption']

x_10724_train, x_10724_test, y_10724_train, y_10724_test = train_test_split(x_10724, y_10724, test_size=0.2, random_state=42)

# Setup random forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_10724_train, y_10724_train)

y_10724_pred = model.predict(x_10724_test)

mse_10724 = root_mean_squared_error(y_10724_test, y_10724_pred)
MAPE_10724 = mean_absolute_percentage_error(y_10724_test, y_10724_pred)
print(mse_10724)
print(MAPE_10724)

'''importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
for f in range(x_10724.shape[1]):
    print(f"{f + 1}. Feature {x_10724.columns[indices[f]]} ({importances[indices[f]]})")'''

# Apply the model on the other buildings to predict solar

for building_df in [df_4746, df_10703, df_4462]:
    building_x = building_df[train_features]
    building_df['solar_consumption'] = model.predict(building_x)

combined_df = pd.concat([df_4462, df_4746, df_10703, df_10724])
# Remove lagged features for ease of reading
combined_df = combined_df[['Tidspunkt', 'property_id', 'Verdi', 'temperature', 'wind_speed', 'wind_direction', 'cloud_fraction', 'precipitation', 'solar_consumption']]
combined_df['imported_portion'] = combined_df['Verdi'] - combined_df['solar_consumption']
combined_df.sort_values(by='Tidspunkt', inplace=True)
combined_df.reset_index(drop=True, inplace=True)
#combined_df.set_index('Tidspunkt', inplace=True)

# Save combined df with all predictions in it to file
combined_df.to_csv('ml/new_datasets/combined_with_predictions.csv', index=False)