##### Price Elasticity of Demand #####

##### Laptop and Computer Dataset #####

##### 0. Imports
### Data manipulation 
import pandas               as pd
import numpy                as np

##### 1. Settings
### Pandas Settings
pd.set_option('display.float_format', lambda x: '%.3f' % x)

##### 2. Loading Data

df = pd.read_csv('../../data/processed/price_elasticity_processed_dataset.csv')

##### 3. Main

# filtering for the main merchant
df_best     = df[df['merchant'] == 'Bestbuy.com']

# filtering for the main category
df_best_laptop = df_best[df_best['category_name'] == 'laptop, computer']

# grouping by the important features
df_best_laptop = df_best_laptop.groupby(['name', 'week_number']).agg({'disc_price': 'mean', 'date': 'count'}).reset_index()

# pivoting the price dataframe
x_price = df_best_laptop.pivot(index= 'week_number' , columns= 'name', values='disc_price')
x_price = pd.DataFrame(x_price.to_records())

# pivoting the demand dataframe
y_demand = df_best_laptop.pivot(index= 'week_number' , columns= 'name', values='date')
y_demand = pd.DataFrame(y_demand.to_records())

# if the product is not sold, the price of it stays the same, and we can fill it's missing values with their respective median values. 
# but unlike the price, the demand for the product on those not selling days is 0.
# using median to fill price missing values because median is more robust to price fluctuations.

aux1 = x_price.median()
x_price.fillna(aux1, inplace=True)

# using zeros to fill demand missing values
y_demand.fillna(0, inplace=True)

#### 4. Export
x_price.to_csv('../../data/processed/x_price.csv')
y_demand.to_csv('../../data/processed/y_demand.csv')




