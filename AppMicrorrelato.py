#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jesús Eduardo Oliva Abarca
"""

from IPython.core.display import HTML
import streamlit as st
import pandas as pd
import numpy as np
import texthero as hero
import spacy
import matplotlib.pyplot as plt
from spacy import displacy

st.set_option('deprecation.showPyplotGlobalUse', False)
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
st.set_option('deprecation.showPyplotGlobalUse', False)

@st.cache(persist= True, suppress_st_warning= True)
def carga(archivo):
    datos = pd.read_csv('Datos/' + archivo, index_col= 0)
    return datos

@st.cache(persist= True, suppress_st_warning= True, allow_output_mutation=True)
def carga_modelo_tareas():
    modelo = spacy.load('es_core_news_sm')
    return modelo

@st.cache(persist= True, suppress_st_warning= True, allow_output_mutation=True)
def carga_modelo_clasificacion():
    modelo = spacy.load('modelo_bow')
    return modelo

nlp_es_tareas = carga_modelo_tareas()
nlp_es_clasificacion = carga_modelo_clasificacion()

def main():
    st.title('Caso de estudio: microficciones en *Twitter*')
    opcion = st.sidebar.selectbox(label= 'Selecciona una opción',
                                  options= ['Bienvenida', 'Análisis Exploratorio (EDA: Exploratory Data Analysis)',
                                            'Aplicaciones básicas de procesamiento de lenguaje natural',
                                            'Sistema de clasificación textual multiclase'])
    if opcion == 'Bienvenida':
       bienvenida()
    elif opcion == 'Análisis Exploratorio (EDA: Exploratory Data Analysis)':
        eda()
    elif opcion == 'Aplicaciones básicas de procesamiento de lenguaje natural':
        procesamiento()
    elif opcion == 'Sistema de clasificación textual multiclase':
        sistema_clasificacion()

def bienvenida():
    st.markdown("""Esta aplicación web ha sido desarrollada por Jesús Eduardo Oliva Abarca, como parte de un proyecto general de investigación 
    que parte del enfoque de la analítica cultural de Lev Manovich, el cual aborda las aplicaciones de las herramientas, métodos y técnicas
    de la ciencia de datos para el estudio de conjuntos de datos culturales masivos.
    En esta aplicación, el usuario puede examinar tres clases de microtextos extraídos del sitio de microblogging *Twitter*, mediante el
    uso de su API (Interfaz de Programación de Aplicaciones: https://developer.twitter.com/en). Los textos recabados se han catalogado como noticias, microficciones, y 
    frases o reflexiones. Aunque el estudio se centra en el análisis de las características de los *tweets* ficcionales, se recolectaron
    también microtextos noticiosos y de reflexiones para desarrollar un sistema automatizado de clasificación textual.""")
    st.markdown("""El propósito de esta aplicación es ofrecer a las interesadas e interesados una herramienta para examinar y clasificar
                diferentes clases de textos. El modelo en que se basa el sistema de clasificación requiere perfeccionarse para incrementar
                su precisión, no obstante, en su estado actual puede ofrecer información relevante para el análisis de textos digitales.""")
    st.markdown("""Los datos empleados para esta aplicación son tratados con todo respeto y confidencialidad. Un último aspecto a señalar 
                es que el corpus de textos aquí presentado fue elaborado durante el año 2020, hasta el mes de septiembre.
    Cualquier duda o comentario: 
        
    jeduardo.oliv@gmail.com""")
    
    st.markdown('https://github.com/JEOA-1981')
    st.markdown('https://www.linkedin.com/in/jes%C3%BAs-eduardo-oliva-abarca-78b615157/')

def eda():
    with st.beta_expander(label= 'Descripción de la sección', expanded= True):
        st.subheader('En esta sección, el usuario puede examinar y visualizar los conjuntos de datos que conforman al corpus de este estudio.')
        
    opcion_general = st.sidebar.radio('Selecciona el conjunto de datos, o el corpus general para su análisis', 
                                      options= ['Conjuntos de datos separados', 'Corpus general'])
        
    if  opcion_general == 'Conjuntos de datos separados':
        opcion_clase = st.selectbox(label= 'Seleciona uno de los conjuntos de datos disponibles', 
                             options= ['Microficciones', 'Noticias', 'Frases/Reflexiones'])
        if opcion_clase == 'Microficciones':
                datos = carga(archivo= 'microficciones.csv')
                st.dataframe(data= datos)
        elif opcion_clase == 'Noticias':
                datos = carga(archivo= 'noticias.csv')
                st.dataframe(data= datos)
        elif opcion_clase == 'Frases/Reflexiones':
                datos = carga(archivo= 'frases_pensamientos.csv')
                st.dataframe(data= datos)
    
    elif opcion_general == 'Corpus general':
        opcion_corpus = st.sidebar.radio(label= 'Seleciona el corpus sin procesar, o ya preprocesado', 
                             options= ['corpus', 'corpus preprocesado'], key= 1)
        if opcion_corpus == 'corpus':
            datos = carga(archivo= 'corpus.csv')
            st.dataframe(data= datos)
        elif opcion_corpus == 'corpus preprocesado':
            datos = carga(archivo= 'corpus_preprocesado.csv')
            st.dataframe(data= datos)
            if st.button(label= 'Generar nube de palabras'):
                nube = hero.wordcloud(datos['Texto limpio'])
                st.pyplot(nube)
            if st.button(label= 'Frecuencia de palabras', key= 1):
                        fig, ax = plt.subplots()
                        ax = hero.top_words(datos['Texto limpio'], normalize=False).head(n= 30).plot(kind= 'barh',
                                                                                                                    color= ['C0', 'C1', 'C2',
                                                                                                                            'C3', 'C4', 'C5', 
                                                                                                                            'C6', 'C7', 'C8',
                                                                                                                            'C9'])
                        st.pyplot(fig)
                
            
            opcion_tipo = st.selectbox(label= 'Seleciona la clase de microtexto ya preprocesado', 
                         options= ['Microficción', 'Noticia', 'Frase/Pensamiento'])
            if opcion_tipo == 'Microficción':
                datos_seleccionados = datos.loc[datos[opcion_tipo] == True]
                st.dataframe(data= datos_seleccionados)
                opcion_uno, opcion_dos = st.beta_columns(2)
                with opcion_uno:
                    if st.checkbox(label= 'Nube de palabras', key= 1):
                        nube = hero.wordcloud(datos_seleccionados['Texto limpio'])
                        st.pyplot(nube)
                with opcion_dos:
                    if st.checkbox(label= 'Frecuencia de palabras', key= 2):
                        fig, ax = plt.subplots()
                        ax = hero.top_words(datos_seleccionados['Texto limpio'], normalize=False).head(n= 30).plot(kind= 'barh',
                                                                                                                    color= ['C0', 'C1', 'C2',
                                                                                                                            'C3', 'C4', 'C5', 
                                                                                                                            'C6', 'C7', 'C8',
                                                                                                                            'C9'])
        
                        st.pyplot(fig)
            elif opcion_tipo == 'Noticia':
                datos_seleccionados = datos.loc[datos[opcion_tipo] == True]
                st.dataframe(data= datos_seleccionados)
                opcion_uno, opcion_dos = st.beta_columns(2)
                with opcion_uno:
                    if st.checkbox(label= 'Nube de palabras', key= 1):
                        nube = hero.wordcloud(datos_seleccionados['Texto limpio'])
                        st.pyplot(nube)
                with opcion_dos:
                    if st.checkbox(label= 'Frecuencia de palabras', key= 2):
                        fig, ax = plt.subplots()
                        ax = hero.top_words(datos_seleccionados['Texto limpio'], normalize=False).head(n= 30).plot(kind= 'barh',
                                                                                                                    color= ['C0', 'C1', 'C2',
                                                                                                                            'C3', 'C4', 'C5', 
                                                                                                                            'C6', 'C7', 'C8',
                                                                                                                            'C9'])
        
                        st.pyplot(fig)
            elif opcion_tipo == 'Frase/Pensamiento':
                datos_seleccionados = datos.loc[datos[opcion_tipo] == True]
                st.dataframe(data= datos_seleccionados)
                opcion_uno, opcion_dos = st.beta_columns(2)
                with opcion_uno:
                    if st.checkbox(label= 'Nube de palabras', key= 1):
                        nube = hero.wordcloud(datos_seleccionados['Texto limpio'])
                        st.pyplot(nube)
                with opcion_dos:
                    if st.checkbox(label= 'Frecuencia de palabras', key= 2):
                        fig, ax = plt.subplots()
                        ax = hero.top_words(datos_seleccionados['Texto limpio'], normalize=False).head(n= 30).plot(kind= 'barh',
                                                                                                                    color= ['C0', 'C1', 'C2',
                                                                                                                            'C3', 'C4', 'C5', 
                                                                                                                            'C6', 'C7', 'C8',
                                                                                                                            'C9'])
        
                        st.pyplot(fig)

def procesamiento():
    with st.beta_expander(label= 'Descripción de la sección', expanded= True):
        st.subheader('En esta sección, el usuario puede revisar algunas de las aplicaciones elementales del procesamiento de lenguaje natural.')
        
    
    diccionario_pos = {
    'ADJ': 'adjetivo',
    'ADP': 'adposición',
    'ADV': 'adverbio',
    'AUX': 'verbo auxiliar',
    'CONJ': 'conjunción coordinante',
    'DET': 'determinador',
    'INTJ': 'interjección',
    'NOUN': 'sustantivo',
    'NUM': 'numero',
    'PART': 'partícula',
    'PRON': 'pronombre',
    'PROPN': 'nombre propio',
    'PUNCT': 'punctuación',
    'SCONJ': 'conjunción subordinante',
    'SYM': 'símbolo',
    'VERB': 'verbo',
    'X': 'otro'
}
    
    with st.beta_expander('Etiquetado de partes del discurso (POS tagging: Part of Speech)'):
        st.info('El etiquetado de partes del discuso (POS tagging: Part of Speech), consiste en la identificación de las funciones gramaticales que cumple cada palabra en un texto.')
        texto = st.text_area('Introduce un texto aquí y presiona Cmd y Enter/Ctrl y Enter para visualizar el etiquetado')
        doc = nlp_es_tareas(texto)
        for i in doc:
            st.info(i.text + ': ' + i.pos_)
        if st.button('Mostrar claves de etiquetado'):
            st.write(diccionario_pos)
    with st.beta_expander('Similitud entre textos'):
        st.info('La similitud entre textos se "computa" mediante la identificación de semejanzas léxicas y semánticas entre frases aisladas o textos extensos.')
        texto_01 = st.text_area(label= 'Introduce el primer texto aquí:', key= 1)
        texto_02 = st.text_area(label= 'Introduce el segundo texto aquí:', key= 2)
        doc_01 = nlp_es_tareas(texto_01)
        doc_02 = nlp_es_tareas(texto_02)
        if st.button(label= 'Calcula la similitud entre textos'):
            st.info('El porcentaje de similitud entre ambos textos es de {:.0%}'.format(doc_01.similarity(doc_02)))
    with st.beta_expander('Visualización de dependencias sintácticas'):
        st.info('La visualización de dependencias sintácticas permite observar de manera gráfica las relaciones estructurales entre palabras, por ejemplo, la relación de determinación entre un sustantivo y un adjetivo')
        texto = st.text_area(label= 'Introduce un texto aquí y presiona Cmd y Enter/Ctrl y Enter para ejecutar la visualización', key= 3)
        doc = nlp_es_tareas(texto)
        #if st.button(label= 'Generar visualización', key= 1):
        html = displacy.render(doc, style= 'dep')
        html = html.replace('\n\n', '\n')
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html= True)
    with st.beta_expander('Visualización de entidades nombradas (NER)'):
        st.info('La visualización de entidades nombradas consiste en el reconocimiento de palabras que se refieren a objetos o sujetos reales, lo que posibilita corroborar, teóricamente, el nivel de factibilidad de un texto, mediante la exploración de referentes "reales" en un texto, tales como personajes históricos, lugares, organizaciones, etc.',)
        texto = st.text_area(label= 'Introduce un texto aquí y presiona Cmd y Enter/Ctrl y Enter para realizar la visualización', key= 4)
        doc = nlp_es_tareas(texto)
        #if st.button(label= 'Generar visualización', key= 2):
        html = displacy.render(doc, style= 'ent')
        html = html.replace('\n\n', '\n')
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html= True)

def sistema_clasificacion():
    st.markdown('En esta sección, puedes obtener información sobre la posible catalogación de un texto, según sea una microficción, una noticia o una reflexión')
    explicacion = """La clasificación de textos se realizó "entrenando" un modelo de clasificación multiclase, esto es, una tipología basada en más de dos etiquetas (a diferencia de un modelo binario)
                    mutuamente excluyentes. En este caso, se clasificaron los *tweets* como "microficción", de "Noticia", o de "Frase/reflexión".
                    La clasificación en el Procesamiento de Lenguaje Natural se basa en el cómputo de atributos comunes en los textos a catalogar. En el caso
                    particular del modelo aquí presentado, la distinción entre microtextos ficcionales, noticiosos o de reflexión se basa en la presencia (o no presencia) de
                    determinadas características lingüísticas (como por ejemplo, el total de verbos conjugados en formas impersonales o personales, las similitudes semánticas entre textos, la frecuencia de un tipo o tipos de partículas gramaticales, etc.). Cabe 
                    señalar que el modelo no tiene la precisión deseada al momento, no obstante, se puede mejorar mediante sucesivos entrenamientos."""
    st.sidebar.markdown('Cataloga el texto introducido según un modelo de clasificación multiclase')
    st.info('Escribe o copia y pega un fragmento de texto, y presiona Cmd y Enter/Ctrl y Enter para obtener la clasificación')
    texto = st.text_area(label= 'Introduce un texto aquí:')
    doc = nlp_es_clasificacion(texto)
    if round(doc.cats['noticia']) > 0:
        st.info('Noticia')
    elif round(doc.cats['microficción']) > 0:
        st.info('Micfroficción')
    elif round(doc.cats['frase']) > 0:
        st.info('Frase/reflexión')
    else:
        st.info('El texto no ha podido ser clasificado')
    with st.beta_expander('Explicación'):
        st.markdown(explicacion)

if __name__ == '__main__':
    main()















