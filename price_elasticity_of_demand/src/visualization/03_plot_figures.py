##### Price Elasticity of Demand #####

##### Plotting Figures #####

##### 0. Imports
### Data manipulation 
import pandas               as pd
import numpy                as np

### Data visualization
import seaborn              as sns
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import matplotlib.dates     as mdates

##### 1. Settings
### Pandas Settings
pd.set_option('display.float_format', lambda x: '%.3f' % x)

##### Visualization Settings
mpl.style.use('ggplot')
mpl.rcParams['figure.figsize'] = (20, 5)
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.color'] = 'lightgray'
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.color'] = 'black'
mpl.rcParams['ytick.color'] = 'black'
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.titlesize'] = 25
mpl.rcParams['figure.dpi'] = 100

sns.set_palette('rocket')

##### 2. Functions

def bar_plot(data, x, y, title, xlabel, ylabel, rotation=0, palette='rocket'):
    """
    Summary:

    Args:

    Returns:

    """

    sns.barplot(data=data, x=x, y=y, palette=palette)

    plt.title(title)
    plt.xlabel(xlabel, color='black')
    plt.ylabel(ylabel, color='black')
    plt.xticks(rotation=rotation)
    plt.tick_params(left=False, bottom=False);

    return None

def set_image(title, xlabel, ylabel, rotation=0):
    """
    Summary:

    Args:

    Returns:

    """
    plt.title(title)
    plt.xlabel(xlabel, color='black')
    plt.ylabel(ylabel, color='black')
    plt.xticks(rotation=rotation)
    plt.tick_params(left=False, bottom=False);

    return None


##### 3. Loading Data

df = pd.read_csv('../../data/processed/price_elasticity_processed_dataset.csv')

##### 3. Main

### Creation of sub-dataframes
df_best     = df[df['merchant'] == 'Bestbuy.com']
df_bhp      = df[df['merchant'] == 'bhphotovideo.com']
df_walmart  = df[df['merchant'] == 'Walmart.com']
df_ebay     = df[df['merchant'] == 'ebay.com']

### 0. Distribution of Prices
fig0        = sns.histplot(df['price'], kde=True, stat='density')

fig0        = set_image('Distribution of Prices', 'Prices', 'Density')

plt.savefig('../../images/prices_histogram.png', dpi=150, format='png', bbox_inches='tight')

### 0.0. Distribution of Prices with Filter
aux1        = df.loc[df['price'] <= 2000, :]

fig00       = sns.histplot(aux1['price'], kde=True, stat='density')

fig00       = set_image('Distribution of Prices', 'Prices', 'Density')

plt.savefig('../../images/prices_histogram_filtered.png', dpi=150, format='png', bbox_inches='tight')

### 1. Sales by Merchant

df_aux      = df[['date', 'merchant']].groupby(['merchant']).count().reset_index().sort_values(by='date', ascending=False)

fig1        = bar_plot(df_aux, 'merchant', 'date', 'Total Amount of Sales by Merchant', 'Merchants', 'Total')

plt.savefig('../../images/sales_by_merchant.png', dpi=150, format='png', bbox_inches='tight')

### 2. Sales by Category

df_aux      = df[['date', 'category_name']].groupby(['category_name']).count().reset_index().sort_values(by='date', ascending=False)

fig2        = bar_plot(df_aux, 'category_name', 'date', 'Total Amount of Sales by Category', 'Categories', 'Total', 90, 'flare_r')

plt.savefig('../../images/sales_by_category.png', dpi=150, format='png', bbox_inches='tight')

### 3. Categories by Merchant

# creating canvas
fig3, axes  = plt.subplots(nrows=1, ncols=4)

# plotting each subplot
plt.subplot(1, 4, 1)
aux1        = df_best[['date', 'category_name']].groupby(['category_name']).count().reset_index().sort_values(by='date', ascending=False)
ax1         = bar_plot(aux1.head(15), 'category_name', 'date', 'BestBuy.com', None, None, 90)

plt.subplot(1, 4, 2)
aux2        = df_bhp[['date', 'category_name']].groupby(['category_name']).count().reset_index().sort_values(by='date', ascending=False)
ax2         = bar_plot(aux2.head(15), 'category_name', 'date', 'bhphotvideo.com', None, None, 90)

plt.subplot(1, 4, 3)
aux3        = df_walmart[['date', 'category_name']].groupby(['category_name']).count().reset_index().sort_values(by='date', ascending=False)
ax3         = bar_plot(aux3.head(15), 'category_name', 'date', 'Walmart.com', None, None, 90)

plt.subplot(1, 4, 4)
aux4        = df_ebay[['date', 'category_name']].groupby(['category_name']).count().reset_index().sort_values(by='date', ascending=False)
ax4         = bar_plot(aux4.head(15), 'category_name', 'date', 'ebay.com', None, None, 90)

# main title
fig3.suptitle('Categories by Merchant', fontsize=20)

# # adjusting spacing
fig3.subplots_adjust(top=0.85)

plt.savefig('../../images/categories_by_merchant.png', dpi=150, format='png', bbox_inches='tight')

### 4. Sales by Brand

df_aux      = df[['date', 'brand']].groupby(['brand']).count().reset_index().sort_values(by='date', ascending=False)

fig4        = bar_plot(df_aux.head(25), 'brand', 'date', 'Total Amount of Sales by Brand', 'Brands', 'Total', 90)

plt.savefig('../../images/sales_by_brand.png', dpi=150, format='png', bbox_inches='tight')

### 5. Brands by Merchant

# creating canvas
fig5, axes  = plt.subplots(nrows=1, ncols=4)

# plotting each subplot
plt.subplot(1, 4, 1)
aux1        = df_best[['date', 'brand']].groupby(['brand']).count().reset_index().sort_values(by='date', ascending=False)
ax1         = bar_plot(aux1.head(15), 'brand', 'date', 'BestBuy.com', None, None, 90)

plt.subplot(1, 4, 2)
aux2        = df_bhp[['date', 'brand']].groupby(['brand']).count().reset_index().sort_values(by='date', ascending=False)
ax2         = bar_plot(aux2.head(15), 'brand', 'date', 'bhphotvideo.com', None, None, 90)

plt.subplot(1, 4, 3)
aux3        = df_walmart[['date', 'brand']].groupby(['brand']).count().reset_index().sort_values(by='date', ascending=False)
ax3         = bar_plot(aux3.head(15), 'brand', 'date', 'Walmart.com', None, None, 90)

plt.subplot(1, 4, 4)
aux4        = df_ebay[['date', 'brand']].groupby(['brand']).count().reset_index().sort_values(by='date', ascending=False)
ax4         = bar_plot(aux4.head(15), 'brand', 'date', 'ebay.com', None, None, 90)

# main title
fig5.suptitle('Brands by Merchant', fontsize=20)

# # adjusting spacing
fig5.subplots_adjust(top=0.85)

plt.savefig('../../images/brands_by_merchant.png', dpi=150, format='png', bbox_inches='tight')

### 6. Sales by Day

df_aux      = df[['date', 'day_n']].groupby(['day_n']).count().reset_index().sort_values(by='date', ascending=False)

fig6        = bar_plot(df_aux, 'day_n', 'date', 'Total Amount of Sales by Day of Week', 'Day of week', 'Total')

plt.savefig('../../images/sales_by_day.png', dpi=150, format='png', bbox_inches='tight')

### 7. Days by Merchant

# creating canvas
fig7, axes  = plt.subplots(nrows=1, ncols=4)

# plotting each subplot
plt.subplot(1, 4, 1)
aux1        = df_best[['date', 'day_n']].groupby(['day_n']).count().reset_index().sort_values(by='date', ascending=False)
ax1         = bar_plot(aux1, 'day_n', 'date', 'BestBuy.com', None, None, 90)

plt.subplot(1, 4, 2)
aux2        = df_bhp[['date', 'day_n']].groupby(['day_n']).count().reset_index().sort_values(by='date', ascending=False)
ax2         = bar_plot(aux2, 'day_n', 'date', 'bhphotvideo.com', None, None, 90)

plt.subplot(1, 4, 3)
aux3        = df_walmart[['date', 'day_n']].groupby(['day_n']).count().reset_index().sort_values(by='date', ascending=False)
ax3         = bar_plot(aux3, 'day_n', 'date', 'Walmart.com', None, None, 90)

plt.subplot(1, 4, 4)
aux4        = df_ebay[['date', 'day_n']].groupby(['day_n']).count().reset_index().sort_values(by='date', ascending=False)
ax4         = bar_plot(aux4, 'day_n', 'date', 'ebay.com', None, None, 90)

# main title
fig7.suptitle('Days of Sales by Merchant', fontsize=20)

# # adjusting spacing
fig7.subplots_adjust(top=0.85)

plt.savefig('../../images/days_by_merchant.png', dpi=150, format='png', bbox_inches='tight')

### 8. Sales by Month

df_aux      = df[['date', 'month_n']].groupby(['month_n']).count().reset_index().sort_values(by='date', ascending=False)

fig8        = bar_plot(df_aux, 'month_n', 'date', 'Total Amount of Sales by Month of Year', 'Month of year', 'Total')

plt.savefig('../../images/sales_by_month.png', dpi=150, format='png', bbox_inches='tight')

### 9. Month by Merchant

# creating canvas
fig9, axes  = plt.subplots(nrows=1, ncols=4)

# plotting each subplot
plt.subplot(1, 4, 1)
aux1        = df_best[['date', 'month_n']].groupby(['month_n']).count().reset_index().sort_values(by='date', ascending=False)
ax1         = bar_plot(aux1.head(6), 'month_n', 'date', 'BestBuy.com', None, None, 90)

plt.subplot(1, 4, 2)
aux2        = df_bhp[['date', 'month_n']].groupby(['month_n']).count().reset_index().sort_values(by='date', ascending=False)
ax2         = bar_plot(aux2.head(6), 'month_n', 'date', 'bhphotvideo.com', None, None, 90)

plt.subplot(1, 4, 3)
aux3        = df_walmart[['date', 'month_n']].groupby(['month_n']).count().reset_index().sort_values(by='date', ascending=False)
ax3         = bar_plot(aux3.head(6), 'month_n', 'date', 'Walmart.com', None, None, 90)

plt.subplot(1, 4, 4)
aux4        = df_ebay[['date', 'month_n']].groupby(['month_n']).count().reset_index().sort_values(by='date', ascending=False)
ax4         = bar_plot(aux4.head(6), 'month_n', 'date', 'ebay.com', None, None, 90)

# main title
fig9.suptitle('Main Month Sales by Each Merchant', fontsize=20)

# settings
fig9.subplots_adjust(top=0.85)

plt.savefig('../../images/months_by_merchant.png', dpi=150, format='png', bbox_inches='tight')

### 10. Sales by Week

df_aux      = df[['date', 'week_number']].groupby(['week_number']).count().reset_index()

fig10        = bar_plot(df_aux, 'week_number', 'date', 'Total Amount of Sales by Week of Year', 'Week of year', 'Total', palette='flare')

plt.savefig('../../images/sales_by_week.png', dpi=150, format='png', bbox_inches='tight')

### 11. Week by Merchant

# creating canvas
fig11, axes = plt.subplots(nrows=1, ncols=4)

# plotting each subplot
plt.subplot(1, 4, 1)
aux1        = df_best[['date', 'week_number']].groupby(['week_number']).count().reset_index().sort_values(by='date', ascending=False)
ax1         = bar_plot(aux1.head(6), 'week_number', 'date', 'BestBuy.com', None, None)

plt.subplot(1, 4, 2)
aux2        = df_bhp[['date', 'week_number']].groupby(['week_number']).count().reset_index().sort_values(by='date', ascending=False)
ax2         = bar_plot(aux2.head(6), 'week_number', 'date', 'bhphotvideo.com', None, None)

plt.subplot(1, 4, 3)
aux3        = df_walmart[['date', 'week_number']].groupby(['week_number']).count().reset_index().sort_values(by='date', ascending=False)
ax3         = bar_plot(aux3.head(6), 'week_number', 'date', 'Walmart.com', None, None)

plt.subplot(1, 4, 4)
aux4        = df_ebay[['date', 'week_number']].groupby(['week_number']).count().reset_index().sort_values(by='date', ascending=False)
ax4         = bar_plot(aux4.head(6), 'week_number', 'date', 'ebay.com', None, None)

# main title
fig11.suptitle('Main Week Sales by Each Merchant', fontsize=20)

# settings
fig11.subplots_adjust(top=0.85)

plt.savefig('../../images/weeks_by_merchant.png', dpi=150, format='png', bbox_inches='tight')

### 12. Price and Demand for laptops and computers

## data manipulation

# selecting category
df_best_laptop = df_best[df_best['category_name'] == 'laptop, computer']

# grouping by the important features
df_best_laptop = df_best_laptop.groupby(['name', 'week_number']).agg({'disc_price': 'mean', 'date': 'count'}).reset_index()

# pivoting the price dataframe
x_price = df_best_laptop.pivot(index= 'week_number' , columns= 'name', values='disc_price')
x_price = pd.DataFrame(x_price.to_records())

# pivoting the demand dataframe
y_demand = df_best_laptop.pivot(index= 'week_number' , columns= 'name', values='date')
y_demand = pd.DataFrame(y_demand.to_records())

# imputing missing values
aux1 = x_price.median()
x_price.fillna(aux1, inplace=True)

# using zeros to fill demand missing values
y_demand.fillna(0, inplace=True)

# data manipulation
aux1 = pd.DataFrame(x_price.apply( lambda x: x.median())).reset_index().drop(0, axis=0)
aux1.columns = ['products', 'price']

aux2 = pd.DataFrame(y_demand.apply( lambda x: x.sum())).reset_index().drop(0, axis=0)
aux2.columns = ['products', 'demand']
aux2['products'] = aux2['products'].str.slice(0, 15) + '...' + aux2['products'].str.slice(-45)

## plots
# creating canvas
fig12, axes = plt.subplots(nrows=2, ncols=1)

# plotting each subplot
plt.subplot(2, 1, 1)
ax1 = bar_plot(aux1, 'products', 'price', 'Median Price by Product', None, 'Price', palette='flare')
plt.xticks([])

plt.subplot(2, 1, 2)
ax2 = bar_plot(aux2, 'products', 'demand', 'Demand by Product', None, 'Demand', 90, palette='flare')

plt.savefig('../../images/price_and_demand.png', dpi=150, format='png', bbox_inches='tight')

