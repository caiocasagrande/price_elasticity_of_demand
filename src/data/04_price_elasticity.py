##### Price Elasticity of Demand #####

##### 0. Imports #####

# Data manipulation 
import pandas               as pd
import numpy                as np

# Data visualization
import seaborn              as sns
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import matplotlib.dates     as mdates

# Statistics and Machine learning 
import statsmodels.api      as sm

# Other libraries

import datetime 
import inflection
import warnings
import lxml

##### 1. Settings #####

# Ignoring warnings
warnings.filterwarnings('ignore')

# Pandas Settings
pd.set_option('display.float_format', lambda x: '%.3f' % x)
# pd.set_option('display.max_columns', None)

# Visualization Settings
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

##### 2. Functions #####

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


def crossprice(df_x, df_y, column_name):
    
    # all values from x_price
    aux1 = x_price.copy()

    # values from y_demand, with the same name as the column
    aux1['y_value-' + column_name] = y_demand[column_name]

    multi_xvalues = aux1.loc[:, aux1.columns[1:-1]]
    multi_yvalues = aux1.loc[:, aux1.columns[-1]]

    # obtaining mean values
    mean_xvalues = np.mean(multi_xvalues)
    mean_yvalues = np.mean(multi_yvalues)

    # linear regression
    X       = sm.add_constant(multi_xvalues)
    model   = sm.OLS(multi_yvalues, X, missing='drop')
    result  = model.fit()

    # obtaining results
    results_summary = result.summary()

    # obtaining p-values
    pvalue = result.pvalues

    # creating a dataframe
    results_as_html     = results_summary.tables[1].as_html()
    aux2       = pd.read_html(results_as_html, header=0, index_col=0)[0]

    # adding p-value to the dataframe
    aux2['p_value'] = pvalue

    # resetting index and changing it to 'name'
    aux2.index.name= 'name'
    aux2.reset_index()

    # calculating cross-prices
    aux2['mean'] = mean_xvalues
    aux2['price_elasticity'] = round((aux2.coef)*(aux2['mean']/mean_yvalues), 2)

    aux2 = aux2.reset_index()
    pvalue_siginicant = aux2['p_value']

    # verifying if the price elasticity is significant
    aux2[column_name + 'CPE'] = np.where((pvalue_siginicant > 0.05), 'No Effect', aux2['price_elasticity'])
    aux2 = aux2.dropna()
    return aux2[['name', column_name + 'CPE']]


##### 3. Loading Data #####

df = pd.read_csv('../../data/processed/price_elasticity_processed_dataset.csv')

##### 4. Feature Engineering #####

df_best     = df[df['merchant'] == 'Bestbuy.com']

df_best_laptop = df_best[df_best['category_name'] == 'laptop, computer']

df_best_laptop = df_best_laptop.groupby(['name', 'week_number']).agg({'disc_price': 'mean', 'date': 'count'}).reset_index()

# pivoting the price dataframe
x_price = df_best_laptop.pivot(index= 'week_number' , columns= 'name', values='disc_price')
x_price = pd.DataFrame(x_price.to_records())

# pivoting the demand dataframe
y_demand = df_best_laptop.pivot(index= 'week_number' , columns= 'name', values='date')
y_demand = pd.DataFrame(y_demand.to_records())

# if the product is not sold, the price of it stays the same, and we can fill it's missing values with their respective median values. but unlike the price, the demand for the product on those not selling days is 0.
# using median to fill price missing values because median is more robust to price fluctuations

aux1 = x_price.median()
x_price.fillna(aux1, inplace=True)

# using zeros to fill demand missing values
y_demand.fillna(0, inplace=True)

##### 5. Price Elasticity

# creating a dictionary for output
results_values_laptop = {
    'product_name': [],
    'price_elasticity': [],
    'mean_price': [],
    'std_price': [],
    'mean_quantity': [],
    'std_quantity': [],
    'intercept': [],
    'slope': [],
    'rsquared': [],
    'p_value': []
}

# calculating price elasticity 
for column in x_price.columns[1:]:
    column_points = []

    for i in range(len(x_price[column])):
        column_points.append((x_price[column][i], y_demand[column][i]))
    
    df = pd.DataFrame(list(column_points), columns=['x_price', 'y_demand'])
    
    x_laptop = df['x_price']
    y_laptop = df['y_demand']
    X_laptop = sm.add_constant(x_laptop)
    
    # machine learning model
    model   = sm.OLS(y_laptop, X_laptop)
    results = model.fit()

    if results.f_pvalue < 0.05:
        
        mean_price          = np.mean(x_laptop)
        std_price           = np.std(x_laptop)
        mean_quantity       = np.mean(y_laptop)
        std_quantity        = np.std(y_laptop)
        intercept, slope    = results.params
        rsquared            = results.rsquared
        p_value             = results.f_pvalue

        price_elasticity = slope*(mean_price/mean_quantity)

        results_values_laptop['product_name'].append(column)
        results_values_laptop['price_elasticity'].append(price_elasticity)
        results_values_laptop['mean_price'].append(mean_price)
        results_values_laptop['std_price'].append(std_price)
        results_values_laptop['mean_quantity'].append(mean_quantity)
        results_values_laptop['std_quantity'].append(std_quantity)
        results_values_laptop['intercept'].append(intercept)
        results_values_laptop['slope'].append(slope)
        results_values_laptop['rsquared'].append(rsquared)
        results_values_laptop['p_value'].append(p_value)

# crating the dataframe for output
df_elasticity = pd.DataFrame.from_dict(results_values_laptop)

# creating a ranking column
df_elasticity['ranking'] = df_elasticity.loc[ : ,'price_elasticity'].rank( ascending = True).astype(int)
df_elasticity = df_elasticity.sort_values('ranking', ascending = True).reset_index(drop = True)

### exporting dataframe ###
df_elasticity.to_csv('../../data/processed/df_elasticity.csv', index=False)

### plotting the price elasticity results ###
fig13 = plt.hlines(y = df_elasticity['ranking'], xmin = 0, xmax = df_elasticity['price_elasticity'], alpha = 0.5, linewidth = 3, color = 'blue')

for name, p in zip(df_elasticity['product_name'].str.slice(0, 30) + '...', df_elasticity['ranking']):
    plt.text(4, p, name)

# creating elasticity labels
for x, y, s in zip(df_elasticity['price_elasticity'], df_elasticity['ranking'], df_elasticity['price_elasticity']):
    plt.text(x, y, round(s, 2), horizontalalignment='right' if x < 0 else 'left', 
                                verticalalignment='center', 
                                fontdict={'color':'red' if x < 0 else 'green', 'size':12})
    
fig13 = set_image('Price Elasticity by Product', 'Price elasticity', 'Ranking')

plt.savefig('../../images/price_elasticity_results.png', dpi=150, format='png', bbox_inches='tight')

##### 6. Business Performance #####

revenue_result = {
    'product_name': [],
    'yrly_revenue': [],
    'price_at_risk':[],
    'new_revenue':[],
    'revenue_variation':[],
    'pct_variation':[]
}

for i in range(len(df_elasticity)):
    yrly_mean_price     = x_price[df_elasticity['product_name'][i]].mean()
    yrly_demand         = y_demand[df_elasticity['product_name'][i]].sum()

    # applying a discount of 10% off
    discounted_price    = yrly_mean_price * 0.9
    new_demmand         = df_elasticity['price_elasticity'][i] * (-0.1)

    new_demmand         = yrly_demand + (yrly_demand*new_demmand)

    yrly_revenue        = yrly_mean_price * yrly_demand
    price_risk          = yrly_revenue - (yrly_revenue * 0.9)
    new_revenue         = discounted_price * new_demmand

    revenue_result['product_name'].append(df_elasticity['product_name'][i])
    revenue_result['yrly_revenue'].append(yrly_revenue)
    revenue_result['price_at_risk'].append(price_risk)
    revenue_result['new_revenue'].append(new_revenue)
    revenue_result['revenue_variation'].append(new_revenue - yrly_revenue)
    revenue_result['pct_variation'].append((new_revenue - yrly_revenue)/yrly_revenue)

# from dictionary to dataframe
revenue_result = pd.DataFrame.from_dict(revenue_result)

# a copy to use in the presentation
results = revenue_result.copy()

# formatting as monetary values
format_func = lambda x: '${:,.2f}'.format(x)
results[['yrly_revenue', 'price_at_risk', 'new_revenue', 'revenue_variation']] = results[['yrly_revenue', 'price_at_risk', 'new_revenue', 'revenue_variation']].applymap(format_func)

# formatting the 'pct_variation' column as percentage values
results['pct_variation'] = (results['pct_variation']*100).map('{:.2f}%'.format)

# exporting results
results.to_csv('../../data/processed/business_performance.csv', index=False)

##### 7. Cross-Price Elasticity #####

# creating a dataframe
df_cross = pd.DataFrame()

# calculating cross-prices for each column
for column in x_price.columns[1:]:
    df_cross[['name_of_the_product', column]] = crossprice(x_price, y_demand, column)

# setting the index
# df_cross = df_cross.set_index('name')

# displaying the results
df_cross.to_csv('../../data/processed/df_cross.csv', index=False)
