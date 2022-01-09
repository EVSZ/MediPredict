import streamlit as st
import pandas as pd
import numpy as np
import plotly as px
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import datetime
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts
import plotly.express as px

st.set_page_config(layout="wide")

def return_name(df, cnk):
    return df[df['CNK'].isin([cnk])]['Product name'].iloc[0]

################################################################################### IMPORTING AND PREPARING ********************************************************
# import main dataset
pharma_df = pd.read_csv('./cleaned_pharma_df.csv')
pharma_df = pharma_df[['CNK', 'Product name', 'Supply date', 'Pharmacy']]
pharma_df['Supply date'] = pharma_df['Supply date'].astype('datetime64[ns]')

# extract most sold medicines, in descending order
most_sold_df = pharma_df.groupby(by=['CNK'])['Supply date'].count().sort_values(ascending=False).reset_index().rename(columns={'Supply date':'Amount sold'})

# extract the monthly sales per cnk, sorted by date
pharma_df['Month'] = pharma_df['Supply date'].dt.month
pharma_df['Year'] = pharma_df['Supply date'].dt.year
pharma_monthly_df = pharma_df.groupby(['Pharmacy', 'Year', 'Month', 'CNK']).count()[['Product name']].reset_index().rename(columns={'Product name':'Amount sold'})

################################################################################### MACHINE LEARNING AND MODELING******************************************************
def machine(cnk, pharmacy):
    if((pharma_monthly_df[pharma_monthly_df['CNK'] == cnk].count() > 50).any() == True):
        # calculate the best hyperparameters for the SVR model
        #print(f'Started at:  {datetime.now().strftime("%H:%M:%S")}')   
        if(pharmacy == 'All Pharmacies'):
            cnk_df = pharma_monthly_df[(pharma_monthly_df['CNK']==cnk)].groupby(['CNK', 'Year', 'Month']).sum().reset_index()
        else:
            cnk_df = pharma_monthly_df[(pharma_monthly_df['CNK']==cnk) & (pharma_monthly_df['Pharmacy']==pharmacy)].groupby(['CNK', 'Year', 'Month']).sum().reset_index()
        
        X = cnk_df[['Year', 'Month']]
        Y = cnk_df['Amount sold']

        cnk_size = X.shape[0]

        X_sample = X.sample(n=cnk_size, random_state=1) 
        Y_sample = Y.sample(n=cnk_size, random_state=1) # random_state must be the same in both datasets to match X and Y

        C = [0.1, 1, 10, 100, 1000]
        gamma = [1, 0.1, 0.01, 0.001, 0.0001,'auto']

        svr_linear = {'C': C, 
                    'kernel': ['linear']} 
        svr_poly =   {'C': C,
                    'gamma': gamma, 
                    'degree': [2, 3, 4],
                    'kernel': ['poly']}
        svr_others = {'C': C,
                    'gamma': gamma, 
                    'kernel': ['rbf', 'sigmoid']}  
        param_grid = [svr_linear, svr_others]

        model_svr_gs = GridSearchCV(estimator=SVR(), param_grid=param_grid, n_jobs=-1)
        model_svr_gs.fit(X_sample, Y_sample)

        # train the model
        X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.33, random_state=0)

        model_svr = model_svr_gs.best_estimator_
        mySVR = model_svr.fit(X_train, Y_train)

        # predict the sales
        fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(20, 5))
        ax = ax.flatten()
        index=0

        # Graph 1

        X_extended = X
        Y_extended = Y
        for month in range(0, months_ahead):
            X_extended = X_extended.append({'Year':(2021 + int(month/12)), 'Month':(month%12 + 1)}, ignore_index=True)
            Y_extended = Y_extended.append(pd.Series([np.nan]))
        Y_extended = Y_extended.reset_index(drop=True).rename('Amount sold')
        Y_pred = model_svr.predict(X_extended)
        pred = pd.DataFrame({'Predicted': Y_pred})
        data = pd.concat([X_extended, Y_extended], axis=1).reindex(X_extended.index)
        comp = pd.concat([data, pred], axis=1).reindex(data.index)
        comp = comp.sort_values(by=['Year', 'Month']).set_index(['Year', 'Month'])
        comp.plot(ax=ax[index], title=f'Monthly sales compared to model prediction \n (CNK {cnk})')
        fig_comp = comp.reset_index()
        fig_comp['Date'] = fig_comp[fig_comp.columns[0:2]].apply(lambda x: '-'.join(x.dropna().astype(str)),axis=1)
        fig1 = px.line(fig_comp, x='Date', y=['Amount sold', 'Predicted'], title=f'Sales')
        index += 1

        # Graph 2
        cum_sales = comp.cumsum()
        cum_sales.plot(ax=ax[index], title=f'Cumulative sales on a monthly basis')
        fig_cum_sales = cum_sales.reset_index()
        fig_cum_sales['Date'] = fig_cum_sales[fig_cum_sales.columns[0:2]].apply(lambda x: '-'.join(x.dropna().astype(str)),axis=1)
        fig2 = px.line(fig_cum_sales, x='Date', y=['Amount sold', 'Predicted'], title=f'Cumulative sales')
        index += 1

        # Graph 3
        delta_df = (comp['Predicted'] - comp['Amount sold']).cumsum().reset_index().rename(columns={0:'Delta'})
        delta_df = delta_df.iloc[0:X.shape[0]]

        X_delta = delta_df[['Year', 'Month']]
        Y_delta = delta_df['Delta']
        sample_size = X_delta.shape[0]
        X_sample = X_delta.sample(n=sample_size, random_state=1) 
        Y_sample = Y_delta.sample(n=sample_size, random_state=1) # random_state must be the same in both datasets to match X and Y
        C = [0.1, 1, 10, 100, 1000, 10000]
        gamma = [1, 0.1, 0.01, 0.001, 0.0001,'auto']
        svr_linear = {'C': C, 
                    'kernel': ['linear']} 
        svr_poly =   {'C': C,
                    'gamma': gamma, 
                    'degree': [1, 2, 3, 4],
                    'kernel': ['poly']}
        svr_others = {'C': C,
                    'gamma': gamma, 
                    'kernel': ['rbf', 'sigmoid']}  
        param_grid = [svr_linear, svr_others]
        model_svr_gs = GridSearchCV(estimator=SVR(), param_grid=param_grid, n_jobs=-1)
        model_svr_gs.fit(X_sample, Y_sample)
        X_train, X_test, Y_train, Y_test = tts(X_delta, Y_delta, test_size=0.33, random_state=0)
        model_svr = model_svr_gs.best_estimator_
        model_svr.fit(X_train, Y_train)

        Y_extended = Y_delta
        for month in range(0, months_ahead): 
            Y_extended = Y_extended.append(pd.Series([np.nan]))
        Y_extended = Y_extended.reset_index(drop=True).rename('Amount sold')

        Y_pred = model_svr.predict(X_extended)
        comp = pd.DataFrame({'Actual delta': Y_extended, 'Predicted': Y_pred})
        comp = pd.concat([X_extended, comp], axis=1).set_index(['Year', 'Month'])    
        comp.plot(ax=ax[index], title=f'Cumulative sales delta prediction \n (CNK {cnk})')
        fig_comp = comp.reset_index()
        fig_comp['Date'] = fig_comp[fig_comp.columns[0:2]].apply(lambda x: '-'.join(x.dropna().astype(str)),axis=1)
        fig3 = px.line(fig_comp, x='Date', y=['Actual delta', 'Predicted'], title=f'Delta of cumulative sales')
        index += 1

        # Graph 4
        cum_sales['Predicted'] -= comp['Predicted']
        cum_sales.plot(ax=ax[index], title=f'Adjusted cumulative sales delta prediction')
        fig_cum_sales = cum_sales.reset_index()
        fig_cum_sales['Date'] = fig_cum_sales[fig_cum_sales.columns[0:2]].apply(lambda x: '-'.join(x.dropna().astype(str)),axis=1)
        fig4 = px.line(fig_cum_sales, x='Date', y=['Amount sold', 'Predicted'],
        title=f'Adjusted delta of cumulative sales')
        index += 1
        
        acc = mySVR.score(X, Y)

        st.header(f'Predictions for {return_name(pharma_df, int(cnk))} with accuracy of {acc * 100:.2f}%')
        
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)
        st.plotly_chart(fig3)
        st.plotly_chart(fig4)

        cum_sales['Buying advice (rounded up)'] = cum_sales['Predicted']
        cum_sales['Compared to last month (%)'] = np.nan
        for i in range(1, cum_sales.shape[0]):
            cum_sales.iat[i, 2] = cum_sales.iat[i, 1] - cum_sales.iat[i-1, 1]
            cum_sales.iat[i, 3] = (cum_sales.iat[i, 2] - cum_sales.iat[i-1, 2])/cum_sales.iat[i-1, 2] * 100
        cum_sales = cum_sales.round({'Compared to last month (%)': 2})
        cum_sales['Buying advice (rounded up)'] = cum_sales['Buying advice (rounded up)'].apply(np.ceil).astype('int64')
        cum_sales = cum_sales[['Buying advice (rounded up)', 'Compared to last month (%)']].reset_index().sort_values(by=['Year', 'Month'], ascending=False)

        st.header('Buying advice:')

        cum_sales
    else:
        st.header(f'The data concerning {return_name(pharma_df, int(cnk))} is insufficient for prediction')

        pharma_monthly_df[pharma_monthly_df['CNK'] == cnk]

################################################################################### Displaying ******************************************************
cnk = st.sidebar.text_input(
    'Please input a CNK number for medicine data retreival',
    key='cnk_search',
    value=1414333
)

st.session_state.months = list(range(1, 36))
st.session_state.pharmacy = ['All Pharmacies', 'A1', 'A2', 'A3', 'A5', 'A6']

months_ahead = st.sidebar.selectbox(
    'Please select range of prediction in Months',
    st.session_state.months
)

#st.session_state.months_ahead = st.sidebar.selectbox(
#    'Please select range of prediction in Months',
#    st.session_state.months
#)

st.session_state.current_pharmacy = st.sidebar.selectbox(
    'Please select your pharmacy',
    st.session_state.pharmacy
)

st.sidebar.button('Search', key='button_search', on_click=machine, args=(int(cnk), st.session_state.current_pharmacy))
