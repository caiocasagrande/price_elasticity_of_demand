##### Price Elasticity of Demand #####

##### 0. Imports
### Data manipulation 
import pandas               as pd
import numpy                as np

### Other libraries
import datetime
import inflection

##### 1. Settings
### Pandas Settings
pd.set_option('display.float_format', lambda x: '%.3f' % x)

##### 2. Loading Data

df_raw = pd.read_csv('../../data/raw/price_elasticity_of_demand.csv')

df = df_raw.copy()

##### 3. Main

# dropping unnecessary columns
df = df.drop(columns={
    'Unnamed: 0', # a second index.
    'Date_imp', # full datetime imputation. other one will be used instead.
    'condition', # all items are considered as 'New'.
    'Imp_count', # informative data. not relevant for this analysis.
    'p_description', # informative data. not relevant for this analysis.
    'currency', # all items are in USD currency.
    'dateAdded', # informative data. not relevant for this analysis.
    'dateSeen', # informative data. not relevant for this analysis.
    'dateUpdated', # informative data. not relevant for this analysis.
    'imageURLs', # informative data. not relevant for this analysis.
    'shipping', # informative data. not relevant for this analysis.
    'sourceURLs', # informative data. not relevant for this analysis.
    'weight', # informative data. not relevant for this analysis.
    'Date_imp_d.1', # duplicated.
    'Zscore_1' # statistical test.
    })

# renaming columns to snake case
old_columns = df.columns.tolist()
snake_case = lambda x: inflection.underscore(x)
new_columns = list(map(snake_case, old_columns))
df.columns = new_columns

# datetime column
df.rename(columns={'date_imp_d': 'date'}, inplace=True)
df['date'] = pd.to_datetime(df['date'])

# is_sale
mapping = {'Yes': 1, 'No': 0}
df['is_sale'] = df['is_sale'].map(mapping)

# dropping columns
df = df.drop(columns={'manufacturer'})

#### 4. Export
df.to_csv('../../data/processed/price_elasticity_processed_dataset.csv')
