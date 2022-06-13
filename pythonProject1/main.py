import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.write("""

## Hello, here is a programme in which you can write any ticker of existing company and see some data related to this.
""")

st.write("""
## Please write a ticker of interest to you (for example AAPL, GOOGL, and etc):
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
df = data.history(period='1d', start=start, end=close)

st.write(
    "You can get some usefull information about chosen company and its country. What would you like to know about?"
)

info = data.info
quest = st.selectbox("Pick one", ["Location and Country", "Business Summary", "All information"])

if quest == "All information":
    for key, value in info.items():
        st.write(key, ":", value)
elif quest == "Business Summary":
    st.write(info['longBusinessSummary'])
elif quest == "Location and Country":
    #browser = webdriver.Chrome(ChromeDriverManager().install())
    url = "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.csv"
    #browser.get(ref)
    data_country = pd.read_csv(url)
    data_country = data_country[data_country['Country Name'] == info['country']]
    mean_gdp = data_country[['Value']].mean()
    data_country = data_country.reset_index()
    country_code = data_country['Country Code'][0]
    #st.write(country_code)
    entry = "http://api.worldbank.org/v2/country/" + country_code + "?format=json"
    params = {'format': 'json'}
    r = requests.get(entry)
    r.encoding = 'utf-8-sig'
    capital = r.json()[1][0]['capitalCity']
    st.write("Your company locates in ", info['country'], " with a capital ", capital, ". Its average GDP for last 60 years is ", str(mean_gdp['Value']), "$")
    st.write("Here you can see the full data about it")
    del data_country['index']
    del data_country['Country Code']
    data_country.columns = ['Country Name', 'Year', 'GDP']
    st.write(data_country)

st.write("""
## Let's take a look at some data, that can be found at <<Yahoo Finance>>, and try to construct predictions using machine learning
""")

choice = st.radio("Pick one", ['Close Price', 'Volume', 'Dividends'])

if choice == 'Close Price':
    df1 = df.Close
    st.line_chart(df1)
    st.write(
        " Linear regression uses 10% of normalized data in order to build trendline and approximately predict future values"
    )
    st.write(
        "There is a comparison of actual prices and ones predicted by our model"
    )
    data = df1
    data = data.reset_index()
    data['index'] = data.index
    x = data
    X = pd.DataFrame(x, columns= ['index', 'Close'])
    x = X.iloc[:, :-1].values
    y = X.iloc[:, 1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    model = LinearRegression()
    model = model.fit(x_train, y_train)
    predict = model.predict(x_test)
    df_predict = pd.DataFrame({'Real Close Price' : y_test, 'Predicted Close Price' : predict})
    #df_predict.index = df.index
    st.write(df_predict)
    st.line_chart(df_predict)
elif choice == 'Volume' :
    df2 = df.Volume
    st.line_chart(df2)
    st.write(
        " Linear regression uses 30% of normalized data in order to build trendline and approximately predict future values"
    )
    st.write(
        "There is a comparison of actual Volumes and ones predicted by our model"
    )
    data = df2
    data = data.reset_index()
    data['index'] = data.index
    x = data
    X = pd.DataFrame(x, columns=['index', 'Volume'])
    x = X.iloc[:, :-1].values
    y = X.iloc[:, 1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    model = LinearRegression()
    model = model.fit(x_train, y_train)
    predict = model.predict(x_test)
    df_predict = pd.DataFrame({'Real Volume': y_test, 'Predicted Volume': predict})
    # df_predict.index = df.index
    st.write(df_predict)
    st.line_chart(df_predict)
    st.write(
        "We can see that Volume is predicted much worse due to unpredictability of purchases before dividends are paid"
    )
elif choice == 'Dividends':
    st.write(
        "There is histogram of accumulated dividends per year"
    )
    df = df.Dividends
    data = df.resample('Y').sum()
    data = data.reset_index()
    data['Year'] = data['Date'].dt.year
    st.bar_chart(data)

st.write(
    "### It is could be very interesting to count correlation between Close Price and Volume of stock, since we know that they should be closely related to each other."
)

data = yf.Ticker(symbol)
df = data.history(period='1d', start=start, end=close)

matrix = pd.DataFrame({'Close' : []})
for e in ['Close', 'Volume']:
    avg = np.mean(np.array(df[e]))
    avg_2 = np.mean(np.array(df[e]) ** 2)
    vrc = avg_2 - avg ** 2
    matrix[e] = [avg, avg_2, vrc]

matrix.index = ['E(x)', 'E(x^2)', 'Var(x)']
a = []
avg_g = []
avg_2_g = []
vrc_g = []

for i in range(len(df['Close'])):
    z = df['Close'][:i+1]
    avg_g.append(np.mean(np.array(z)))
    avg_2_g.append(np.mean(np.array(z) ** 2))
    vrc_g.append(np.mean(np.array(z) ** 2) - np.mean(np.array(z)) ** 2)

graph = pd.DataFrame({'Average': avg_g, 'Variance': vrc_g})
f = px.bar(graph, x = df.index, y = ['Average'], barmode = "overlay")
st.plotly_chart(f)

cov = np.sum((np.array(df['Close'])-matrix['Close']['E(x)']) * (np.array(df['Volume'])-matrix['Volume']['E(x)'])) / len(df['Close'])
corr = cov/((matrix['Close']['Var(x)'] ** 0.5)*(matrix['Volume']['Var(x)'] ** 0.5))

st.write(
    "### Using formulas from Maths we have just found correlation between our variables is ", str(corr)
)

st.write(
    "# Thank you for your attention"
)
