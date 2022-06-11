import numpy as np
import pandas as pd
#import plotly.express as px
#import seaborn as sns
#import matplotlib.pyplot as plt
#from scipy import stats
import streamlit as st
#from pytrends.request import TrendReq
import yfinance as yf



st.write("""

## Hello, here is a programme in which you can write any ticker of existing company and see some data related to this.
""")

st.write("""
### Please write a ticker of interest to you (for example AAPL, GOOGL, and etc):
""")

symbol = st.text_input("Ticker")

st.write("""
### And dates required:
""")

start = st.date_input("Start Date")
close = st.date_input("Close Date")

#symbol = 'AAPL'
#start = '2015-5-5'
#close = '2022-5-5'

data = yf.Ticker(symbol)
df = data.history(period = '1d', start = start, end = close)

st.write(
    "You can get some information about chosen company. What would you like to know about?"
)

info = data.info

quest = st.selectbox("Pick one", ["Business Summary", "Location", "Popularity", "All information"])

if quest == "All information":
    for key, value in info.items():
        st.write(key, ":", value)
elif quest == "Business Summary":
    st.write(info['longBusinessSummary'])
elif quest == "Location":
    st.write(info['country'])
#elif quest == "Popularity":
#    st.write()

st.write("""
### First of all, you can see close prices of this stock
""")

st.line_chart(df.Close)

st.write("""
### Its volume
""")

st.line_chart(df.Volume)

st.write("""
### And dividends
""")

st.line_chart(df.Dividends)

st.write("""
### Chose one of them and we will try to build its trend
""")

choice = st.radio ("Pick one", ['Close Price', 'Volume', 'Dividends'])






