#importing lib.
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from streamlit_option_menu import option_menu
from xgboost import XGBRFRegressor
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
#import pandas_profiling 
from streamlit_pandas_profiling import st_profile_report



# Set up a style for a plain rectangle box with padding
title_style = "border: 1px solid #ddd; border-radius: 4px; padding: 5px; background-color: #22222F;Margin-top:-10px;"
#Set Page title
st.set_page_config(page_title="Strength Forecast", layout = 'wide', page_icon = 'lo.jpg', initial_sidebar_state = 'auto')

#setting page theme

# Define custom dark theme CSS
dark_theme_css = """
    <style>
        body {
            background-color: #1E1E1E;
            color: #FFFFFF;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background-color: #1E1E1E;
        }
        .css-17e1w1i {
            background-color: #272727;
        }
    </style>
"""

# Inject dark theme CSS into the app
st.markdown(dark_theme_css, unsafe_allow_html=True)

# Use st.markdown to format the title with the specified style
st.markdown(f'<div style="{title_style}"><h2 style="text-align:center; font-weight: bold;">Concrete Strength Predictions</h2></div>', unsafe_allow_html=True)

with st.sidebar:
    nav = option_menu(
        menu_title="Main Menu",
        options=['Home','Prediction','Exploratory data analysis','Support'],
        icons=["house","bounding-box","graph-up-arrow","book"],
        menu_icon="cast",
        default_index=0,
        )
    

if nav=='Home':
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-photo/gradient-bokeh-digital-business-wallpaper_53876-110796.jpg?w=1060&t=st=1703702796~exp=1703703396~hmac=bd209d0e218aeceaf7e1ad9497188fb5c96c285f01ea64f10f50276e8a6054d3");
    background-size: cover;
    }

    [data-testid="stHeader"]{
        background:rgba(0,0,0,0)
    }
    </style>
    '''
    df=pd.read_csv("Concrete-strength_data.csv")
    st.markdown("#### Dataset Description")

    st.write(
        """
        - **Cement:** The quantity of cement in the mix.
        - **Blast Furnace Slag:** Amount of blast furnace slag, a byproduct of iron production.
        - **Fly Ash:** Quantity of fly ash, a residue from burning coal.
        - **Water:** The volume of water in the mix.
        - **Superplasticizer:** Presence of a chemical additive to improve fluidity.
        - **Coarse Aggregate:** Amount of coarse aggregate, typically gravel or crushed stone.
        - **Fine Aggregate:** Quantity of fine aggregate, usually sand.
        - **Age (days):** The curing time of the concrete in days.
        - **Concrete Strength:** The compressive strength of the concrete sample.

        """
        )
    st.markdown("#### Dataset Features Distribution")
    labels = df.columns[:-1]  # Exclude the last column (Concrete Strength) for the donut chart
    values = df.iloc[0, :-1]  # Take the first row as an example
    fig = px.pie(df, names=labels, values=values, hole=0.4, color_discrete_sequence=px.colors.qualitative.Set1)

    # Customize layout for transparent background
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',  # Set background color to transparent
        plot_bgcolor='rgba(0,0,0,0)'    # Set plot area background color to transparent
    )

    # Display the interactive donut chart using st.plotly_chart
    st.plotly_chart(fig)

    st.markdown("#### Dataset Overview")

    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.dataframe(df)




if nav=='Prediction':
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-photo/gradient-bokeh-digital-business-wallpaper_53876-110796.jpg?w=1060&t=st=1703702796~exp=1703703396~hmac=bd209d0e218aeceaf7e1ad9497188fb5c96c285f01ea64f10f50276e8a6054d3");
    background-size: cover;
    }

    [data-testid="stHeader"]{
        background:rgba(0,0,0,0)
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        a= st.number_input("Cement", placeholder="Type a number...",min_value=1.0, max_value=1000.0, value=None, step=None)
        b = st.number_input("Blast Furnace Slag", placeholder="Type a number...",min_value=1.0, max_value=1000.0, value=None, step=None)
        c= st.number_input("Fly Ash", placeholder="Type a number...",min_value=1.0, max_value=1000.0, value=None, step=None)
        d = st.number_input("Water", placeholder="Type a number...",min_value=1.0, max_value=1000.0, value=None, step=None)
   
    with col2:
        e= st.number_input("Coarse Aggregate", placeholder="Type a number...",min_value=1.0, max_value=1000.0, value=None, step=None)
        f = st.number_input("Fine Aggregate", placeholder="Type a number...",min_value=1.0, max_value=1000.0, value=None, step=None)
        g = st.number_input("Age (days)", placeholder="Type a number...",min_value=1.0, max_value=1000.0, value=None, step=None)

    # Convert input values to float, handling None values
    def convert_to_float(value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    a = convert_to_float(a)
    b = convert_to_float(b)
    c = convert_to_float(c)
    d = convert_to_float(d)
    e = convert_to_float(e)
    f = convert_to_float(f)
    g = convert_to_float(g)
    
    #mapping the input and predciting the pattern
    ml1=open('model_regressor.pkl','rb')
    classifier=pickle.load(ml1)

    mi=open('scaling_mi.pkl','rb')
    scaling=pickle.load(mi)

    input_x=np.array([a,b,c,d,e,f,g],dtype=float).reshape(1,7)

    scale=scaling.transform(input_x)
    pred=classifier.predict(scale)[0]

    if st.button("Predict"):
        st.success(f"so, by selecting the component in the specific percentage Cement {a}% ,Blast Furnace Slag {b}% ,Fly Ash {c}% ,Water {d}% ,Coarse Aggregate {e}% ,Fine Aggregate {f}% and Age(days) {g}%  so , Machine learning model  generate Predicted strength equal to {np.round(pred)}% ")

if nav=='Exploratory data analysis':
    page_bg_img = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url("https://img.freepik.com/free-photo/gradient-bokeh-digital-business-wallpaper_53876-110796.jpg?w=1060&t=st=1703702796~exp=1703703396~hmac=bd209d0e218aeceaf7e1ad9497188fb5c96c285f01ea64f10f50276e8a6054d3");
    background-size: cover;
    }

    [data-testid="stHeader"]{
        background:rgba(0,0,0,0)
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    st.header("Pandas Profiling report")
    #pr = df2.profile_report()
    #pr.to_file("Analysis.html")
    #pr.to_file("Analysis.json")
    #st_profile_report(pr)
    # Load your Pandas Profiling HTML report
    html_report_path = "Analysis.html"

    # Render the HTML report in Streamlit
    with open(html_report_path, "r", encoding="utf-8") as file:
        html_content = file.read()
        # Embed the HTML content in Streamlit
        st.components.v1.html(html_content, height=2000, scrolling=True)

if nav=="Support":
    st.title("Support")
    link = "[Visit Github for futher query](https://github.com/Priyanshu-Ganwani09)"
    st.markdown(link, unsafe_allow_html=True)
