# Sistema de deteccao de diabetes

import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from PIL import Image

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title('Sistema de Detecção de Diabetes')

data_file = 'diabetes.csv'
df = pd.read_csv(data_file)

atributos = list(df.columns.values)
atributos.remove('Outcome')

st.sidebar.title("Informe os dados")


# guardar os valores de min, max e média
atributos_valores = {}
for atributo in atributos:
    minimo, media, maximo = float(df[atributo].min()), float(df[atributo].mean()), float(df[atributo].max())
    atributos_valores[atributo] = {"min": minimo, "media": media, "max": maximo }


with st.sidebar:
    with st.form(key='my_form'):

        atributo = 'Pregnancies'
        Pregnancies = st.number_input("Filhos", min_value=0.0, max_value=10.0, value=atributos_valores[atributo]['media'],step=1.0,format="%.0f")
        
        atributo = 'Glucose'
        Glucose = st.number_input("Glicose", atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],step=1.0,format="%.0f")
        
        atributo = 'BloodPressure'
        BloodPressure = st.number_input("Pressão arterial", atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],step=1.0,format="%.0f")
        
        atributo = 'SkinThickness'
        SkinThickness = st.number_input("Espessura da pele", atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],step=1.0,format="%.0f")
        
        atributo = 'Insulin'
        Insulin = st.number_input("Insulina", atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],step=1.0,format="%.0f")
        
        atributo = 'BMI'
        bmi  = st.number_input("IMC", atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],step=1.0,format="%.1f")
        
        atributo = 'DiabetesPedigreeFunction'
        DiabetesPedigreeFunction = st.number_input(atributo, atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],step=0.01,format="%.3f")
        
        atributo = 'Age'
        Age = st.number_input("Idade", atributos_valores[atributo]['min'], atributos_valores[atributo]['max'], value=atributos_valores[atributo]['media'],step=1.0,format="%.0f")
        
        predict_button = st.form_submit_button(label='Prever')



# Pagina pricipal
arquivo_modelo = 'ModeloXGBoost.pkl'
with open(arquivo_modelo, 'rb') as f:
    modelo = pickle.load(f)

def previsao_classificao_diabetes(modelo, Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                    Insulin, bmi, DiabetesPedigreeFunction, Age):

    new_X = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                    Insulin, bmi, DiabetesPedigreeFunction, Age])
    
    
    classificacao_diabetes_XGBoost = modelo.predict(new_X.reshape(1, -1) )[0]

    classificacao_diabetes = classificacao_diabetes_XGBoost

    return classificacao_diabetes


imagem = 'diabetes.jpeg'
image = Image.open(imagem)
st.image(image, width=600)

if predict_button:
    classificaco_diabetes  = previsao_classificao_diabetes(modelo, Pregnancies, Glucose, BloodPressure, SkinThickness, 
                                    Insulin, bmi, DiabetesPedigreeFunction, Age)

    st.markdown('## Detecção diabetes (sim-nao): ' + \
                str(classificaco_diabetes == 1 and "Sim" or "Não" ) )