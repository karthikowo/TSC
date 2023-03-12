import streamlit as st
import plotly.express as px
import pickle
import plotly.graph_objects as go

try:
    imports = pickle.load(open('exports.pkl','rb'))
    df = imports[-1]
    resid = imports[-4]
    seasonality = imports[-5]
    trend = imports[-6]

    st.write("Decomposition of the Time Series Data.")

    fig1 = px.line(df, x="point_timestamp", y="point_value", title="Original Dataset Plot")
    st.plotly_chart(fig1)

    fig3 = px.line(trend,title="Trend")
    st.plotly_chart(fig3)

    fig4 = px.line(seasonality, title="Seasonality")
    st.plotly_chart(fig4)

    fig5 = px.line(resid, title="Residuals")
    st.plotly_chart(fig5)


except FileNotFoundError:
    st.write("Please wait.")
