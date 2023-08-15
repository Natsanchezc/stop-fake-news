import streamlit as st
import pandas as pd
from models import classifier

st.title("Introduzca una noticia")

noticia = st.text_area('', 'Ej: La continuidad del Instituto Nacional Electoral...')

st.write("\n\n")
st.write("\n\n")

st.subheader("Le diremos a qué periódico pertenece:")
st.write("\n\n")


realidad, prediccion = classifier([noticia])

if st.button('Clasificar noticia'):
    if len(realidad) > 0:
        st.write('Realidad: ', realidad)
    else:
        st.write('No disponemos de los datos de esta noticia. Sin embargo, diríamos que pertenece al periódico:')
    st.write('Predicción: ', prediccion)


