
##### Price Elasticity of Demand #####

##### 0. Imports #####

### Data manipulation 
import pandas               as pd
import numpy                as np

### Data visualization
import seaborn              as sns
import matplotlib           as mpl
import matplotlib.pyplot    as plt
import matplotlib.dates     as mdates

### Statistics and Machine learning 
import statsmodels.api      as sm

### Other libraries

import streamlit            as st

import datetime 
import inflection
import warnings

from PIL import Image

##### 1. Settings #####

### Ignoring warnings
warnings.filterwarnings('ignore')

### Pandas Settings
pd.set_option('display.float_format', lambda x: '%.3f' % x)

### Visualization Settings
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
mpl.rcParams['figure.dpi'] = 150

sns.set_palette('rocket')

##### 2. Loading Data #####

### Loading Data

## processed datasets 
df              = pd.read_csv('data/processed/price_elasticity_processed_dataset.csv')
df_elasticity   = pd.read_csv('data/processed/df_elasticity.csv')
df_business     = pd.read_csv('data/processed/business_performance.csv')
df_cross        = pd.read_csv('data/processed/df_cross.csv')

df_cross = df_cross.set_index('name_of_the_product')

### Images

fig01    = Image.open('images/sales_by_merchant.png')
fig02    = Image.open('images/sales_by_category.png')
fig03    = Image.open('images/categories_by_merchant.png')
fig04    = Image.open('images/sales_by_brand.png')
fig05    = Image.open('images/brands_by_merchant.png')
fig06    = Image.open('images/sales_by_week.png')
fig07    = Image.open('images/sales_by_day.png')
fig08    = Image.open('images/days_by_merchant.png')
fig09    = Image.open('images/sales_by_month.png')
fig10    = Image.open('images/months_by_merchant.png')
fig12    = Image.open('images/price_and_demand.png')
fig13    = Image.open('images/price_elasticity_results.png')

##### 3. Streamlit App #####

# Set the background color of the Streamlit app
st.set_page_config(layout="wide", page_title="Price Elasticity of Demand", page_icon="📊", 
                   initial_sidebar_state="expanded")

st.sidebar.markdown('# Price Elasticity of Demand')
st.sidebar.markdown("""---""")
st.sidebar.markdown(
    """
    In this fictional project, I aim to explore the price elasticity of demand for laptops and computers. 
    
    Faced with a scarcity of real-world data, my investigation delves into a hypothetical retail environment. 
    The goal is to understand how price variations impact consumer demand for these products, despite limited data availability. 
    The analysis employs fundamental principles of price elasticity to uncover insights that could drive strategic decision-making. 
    
    This project demonstrates the importance of economic modeling and data science in a fictional scenario, offering a brief look into the potential value of such analysis for businesses operating in the real world.
    """
)
st.sidebar.markdown("""---""")
st.sidebar.markdown('Powered by [Caio Casagrande](https://www.linkedin.com/in/caiopc/)')
st.sidebar.markdown('Github [Notebook](https://github.com/caiocasagrande/price_elasticity_of_demand/blob/main/notebooks/price_elasticity.ipynb)')

st.header('Price Elasticity of Demand Project')

tab1, tab2, tab3, tab4 = st.tabs(['Exploratory Data Analysis', 
                                  'Price Elasticity', 
                                  'Business Performance', 
                                  'Crossed Elasticity'])

with tab1:
    st.markdown(
        """
        **Exploratory Data Analysis** main objectives are:

        **Obtain Business Experience** in order to understand the business context and domain knowledge. 
        
        **Validate Business Hypotheses (Insights)** providing a data-driven approach to confirm or challenge what the business believes to be true. This process leads to valuable insights that may guide strategic decisions.
        """
        )

    st.markdown("""---""")

    with st.container():
        st.markdown("### Which merchant sell most?")

        st.image(fig01, use_column_width=True)

    with st.container():
        st.markdown("### What are the best-selling categories?")

        col1, col2 = st.columns(2)
        
        with col1:
            st.image(fig02, use_column_width=True)

        with col2:
            st.image(fig03, use_column_width=True)

    with st.container():
        st.markdown("### What are the best-selling brands overall?")

        col1, col2 = st.columns(2)
        
        with col1:
            st.image(fig04, use_column_width=True)

        with col2:
            st.image(fig05, use_column_width=True)

    with st.container():
        st.markdown("""---""")
        st.markdown("## Analyzing data through time")
        st.markdown("### Which weeks sell most?")

        st.image(fig06, use_column_width=True)

    with st.container():
        st.markdown("### How sales behave over the days of the week?")

        col1, col2 = st.columns(2)
        
        with col1:
            st.image(fig07, use_column_width=True)

        with col2:
            st.image(fig08, use_column_width=True)
    
    with st.container():
        st.markdown("### How sales behave throughout the months of the year?")

        col1, col2 = st.columns(2)
        
        with col1:
            st.image(fig09, use_column_width=True)

        with col2:
            st.image(fig10, use_column_width=True)


with tab2:
    st.markdown(
        """
        ##### The price elasticity of demand will be calculated for the best selling category of the best selling merchant. According to the previous EDA, it will be the *"laptop, computer"* category for *"Bestbuy.com"*.
        """
    )
    st.markdown("""---""")
    st.markdown('##### 1. There are 39 products in the *"laptop, computer"* category for *"Bestbuy.com"*. Their price and demand data are as follows:')
    st.image(fig12, use_column_width=True)

    st.markdown("""---""")
    st.markdown('##### 2. These are the products statistically significant at the *p=.05* level:')
    st.dataframe(df_elasticity, use_container_width=True)
    
    st.markdown("""---""")
    st.markdown('##### 3. Results expressed grafically:')
    st.image(fig13, use_column_width=True)

with tab3:
    st.markdown(
        """
        ##### Business Performance is crucial for optimizing pricing strategies, forecasting revenues, and making data-driven decisions. It helps maximize profits, position the business competitively, and tailor strategies to customer segments with varying price sensitivities.
        """
    )
    st.markdown("#### Results:")
    st.dataframe(df_business, use_container_width=True)
    st.markdown("""---""")
    st.markdown(
        """
        ##### The new revenue provides an increase of **$197,015.78** in annual revenue from latptops and computers.
        > ###### For example, the current revenue for product **"12 MacBook (Mid 2017, Silver)"** is **$12,959.90** annually. 
        > ###### Considering a 10% promotion on the product's value, we could lose something around **$1,295.99**. 
        > ###### However, considering the high price elasticity of demand for the product, the demand for it grows generating an increase of **$74,654.25** in revenue over the year (576.04%).
        > ###### That is, a new revenue of **$87,614.15** annually.
        """
    )

with tab4:
    st.markdown(
        """
        ##### Cross Elasticity of Demand measures the responsiveness of the quantity demanded of one product to changes in the price of another product. 
        > It is calculated as the percentage change in the quantity demanded of one product divided by the percentage change in the price of another product. 
        > A positive cross elasticity indicates that the two goods are substitutes, meaning that an increase in the price of one leads to an increase in the demand for the other, while a negative cross elasticity suggests they are complements, indicating that an increase in the price of one leads to a decrease in the demand for the other. 
        > Cross elasticity is a valuable concept for businesses to understand how products are related in the market and to make informed pricing and marketing decisions.
        """
        )
    st.dataframe(df_cross, use_container_width=True) 
