#loading required libraries
import pandas as pd
import numpy as np

# loading the library for the visualisation
import streamlit as st

# Visualisations
import matplotlib.pyplot as plt

# Correlation test
from scipy import stats
from scipy.stats import pearsonr

# RMSE
from sklearn.metrics import mean_squared_error,r2_score

# Splitting the data
from sklearn.model_selection import train_test_split

# Linear Regression library
from sklearn.linear_model import LinearRegression

# Statitistical Tests - Z-Test
from statsmodels.stats.weightstats import ztest as ztest

# OLS Regression
import statsmodels.api as sm

# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor


#Suppressing warnings
import warnings
warnings.filterwarnings("ignore")

MODIS = pd.read_csv(r'D:\Data Science\Nigeria Proj\Intermediate\GEE Data\Nigeria_LGAs_NDVI.csv')
VIIRS = pd.read_csv(r'D:\Data Science\Nigeria Proj\Intermediate\GEE Data\VIIRS_Admin2.csv')


def resampling_monthly(data1:pd.DataFrame)  -> pd.DataFrame:

    data1['date'] = [x.replace('_','-') for x in data1['imageId']]
    data1['date'] = pd.to_datetime(data1['date'])
    data1.set_index('date', inplace=True)
    data1.drop(['system:index','.geo','imageId'], inplace=True, axis=1)
    data1 = data1.resample('M').mean()
    return data1


MODIS_monthly = resampling_monthly(MODIS)
VIIRS_monthly = resampling_monthly(VIIRS)


MODIS_2012_2023=  MODIS_monthly[MODIS_monthly.index >= '2012-01-17'] 
MODIS_2000 = MODIS_monthly[MODIS_monthly.index < '2012-01-17']

st.title('MODIS AND VIIRS  NDIVI COMPARISON')


selected_region = st.sidebar.selectbox('Select a LGA:', options=MODIS_2012_2023.columns)

def correlation_VIIRS_MODIS(data1:pd.DataFrame,data2:pd.DataFrame) -> str:
    """
    Here we are going to be carrying out correlation analysis between the different MODIS and VIIRS LGAs
    """
    
    data = pd.concat([data1[selected_region], data2[selected_region]], axis=1)

    corrs = data.corr()

    corrs.columns = [f'{data1[selected_region]}_MODIS',f"{data1[selected_region]}_VIIRS"]

    return corrs.iloc[1][0]

def regression(x:pd.DataFrame,y:pd.DataFrame,x_val:pd.DataFrame)->float:
    """
    This function performs a simple linear regression analysis on the MODIS and VIIRS data sets seeing how well the VIIRS data can 
    be used to explain the discontinued MODIS data
    """
    X_train, y_train, X_test, y_test = x.iloc[-int(0.7*x.shape[0]):], y.iloc[-int(0.7*y.shape[0]):],x.iloc[:-int(0.7*x.shape[0])],y.iloc[:-int(0.7*x.shape[0])]
    reg = LinearRegression()
    reg.fit(np.array(X_train[selected_region]).reshape(-1,1),np.array(y_train[selected_region]).reshape(-1,1))
    y_pred = reg.predict(np.array(X_test[selected_region]).reshape(-1,1))
    y_val = reg.predict(np.array(x_val[selected_region]).reshape(-1,1))
    df = pd.DataFrame({'MODIS 2000-2012':x_val[selected_region],'VIIRS 2000-2012':y_val.reshape(len(y_val),)})
    return  np.sqrt(mean_squared_error(y_pred,y_test[selected_region])), reg.score(np.array(X_train[selected_region]).reshape(-1,1), np.array(y_train[selected_region]).reshape(-1,1)),df


def polynomial_regression(x:pd.DataFrame,y:pd.DataFrame,x_val:pd.DataFrame):
    
    X_train, y_train, X_test, y_test = x.iloc[-int(0.7*x.shape[0]):], y.iloc[-int(0.7*y.shape[0]):],x.iloc[:-int(0.7*x.shape[0])],y.iloc[:-int(0.7*x.shape[0])]
    degree = 3
    polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    polyreg.fit(np.array(X_train[selected_region]).reshape(-1,1),np.array(y_train[selected_region]).reshape(-1,1))
    scores = polyreg.score(np.array(X_train[selected_region]).reshape(-1,1),np.array(y_train[selected_region]).reshape(-1,1))
    y_val = polyreg.predict(np.array(x_val[selected_region]).reshape(-1,1))
    df = pd.DataFrame({'MODIS 2000-2012':x_val[selected_region],'VIIRS 2000-2012':y_val.reshape(len(y_val),)})

    return scores, df

def random_forest(x:pd.DataFrame,y:pd.DataFrame,x_val:pd.DataFrame):
    X_train, y_train, X_test, y_test = x.iloc[-int(0.7*x.shape[0]):], y.iloc[-int(0.7*y.shape[0]):],x.iloc[:-int(0.7*x.shape[0])],y.iloc[:-int(0.7*x.shape[0])]
    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(np.array(X_train[selected_region]).reshape(-1,1),np.array(y_train[selected_region]).reshape(-1,1))
    scores = rf_model.score(np.array(X_train[selected_region]).reshape(-1,1), np.array(y_train[selected_region]).reshape(-1,1))
    y_val = rf_model.predict(np.array(x_val[selected_region]).reshape(-1,1))
    df = pd.DataFrame({'MODIS 2000-2012':x_val[selected_region],'VIIRS 2000-2012':y_val.reshape(len(y_val),)})
    
    return scores, df

poly = polynomial_regression(MODIS_2012_2023,VIIRS_monthly, MODIS_2000)

rf = random_forest(MODIS_2012_2023,VIIRS_monthly,MODIS_2000)

linear_regs = regression(MODIS_2012_2023,VIIRS_monthly, MODIS_2000)

st.write("The objective of this analysisis to carry out backdating of the VIIRS data using the MODIS data provided")
st.header('Data and Methodology')
st.write("In this process we will first create temporal plots of the different regions showing the MODIS and VIIRS comparison")
MODIS_filtered_df = MODIS_2012_2023[[selected_region]]
VIIRS_filtered_df = VIIRS_monthly[[selected_region]]
df = pd.concat([MODIS_filtered_df, VIIRS_filtered_df], axis=1)
df.columns = [f'{selected_region}_MODIS', f'{selected_region}_VIIRS']
###PLotting the charts
# fig, ax = plt.subplots()
# MODIS_filtered_df.plot(kind='line', ax=ax)
# VIIRS_filtered_df.plot(kind='line', ax=ax)
# ax.set_title(f'Temporal Distribution  of NDVI for {selected_region} over time for  MODIS & VIIRS')
# ax.set_ylabel('NDVI')
# ax.set_xlabel('Date')
# st.pyplot(fig)
st.subheader(f'Temporal Distribution  of NDVI for {selected_region} over time for  MODIS & VIIRS')
st.line_chart(df, color=['#39FF14','#FF3131'])

st.subheader(f"Creating Correlation")

corrs = correlation_VIIRS_MODIS(MODIS_2012_2023, VIIRS_monthly)

colors = ''
if corrs>= 0.98:
    colors+='#4EE35A'
elif corrs<=0.5:
    colors+='#DC143C' 
elif 0.5<corrs<0.98:
    colors+='#FFA500'



st.write(f"We would like to then carryout a Pearson correlation analysis to see whether simply imputing the MODIS data to the missing VIIRS data")

st.markdown(f'The correlation between MODIS and VIIRS for {selected_region} is <span style="color:{colors}; font-size:35px;">{corrs}</span>', unsafe_allow_html=True)

# Linear Regression
st.header("Using Linear Regression for Backdating")
colors1 = ''
if linear_regs[1]>= 0.98:
    colors+='#4EE35A'
elif linear_regs[1]<=0.5:
    colors+='#DC143C' 
elif 0.5<linear_regs[1]<0.98:
    colors+='#FFA500'

st.markdown(f'The Adjusted Rsquared of the Linear Regression between MODIS and VIIRS for {selected_region} is <span style="color:{colors1}; font-size:35px;">{linear_regs[1]}</span> and the RMSE is {linear_regs[0]}', unsafe_allow_html=True)
st.subheader(f'Backcasting Temporal Distribution of NDVI for {selected_region} over time for  MODIS & VIIRS using Linear Regression')
st.line_chart(linear_regs[2], color=['#39FF14','#FF3131'])

### Polynomial Regression

colors2 = ''
if poly[0]>= 0.98:
    colors+='#4EE35A'
elif poly[0]<=0.5:
    colors+='#DC143C' 
elif 0.5<poly[0]<0.98:
    colors+='#FFA500'

st.header("Using Polynomial Regression for Backdating")
st.markdown(f'The Rsquared of the Polynomial Regression between  MODIS and VIIRS for {selected_region} is <span style="color:{colors}; font-size:35px;">{poly[0]}</span> and the RMSE is {poly[0]}', unsafe_allow_html=True)

st.subheader(f'Backcasting Temporal Distribution of NDVI for {selected_region} over time for  MODIS & VIIRS using Polynomial Regression')
st.line_chart(poly[1], color=['#39FF14','#FF3131'])


## Random Forest Regressor
st.header("Using Random Forest Regression for Backdating")
st.markdown(f'The Rsquared of the Random Forest Regression between  MODIS and VIIRS for {selected_region} is <span style="color:{colors}; font-size:35px;">{rf[0]}</span> and the RMSE is {rf[0]}', unsafe_allow_html=True)

st.subheader(f'Backcasting Temporal Distribution of NDVI for {selected_region} over time for  MODIS & VIIRS using Polynomial Regression')
st.line_chart(rf[1], color=['#39FF14','#FF3131'])



