import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Perfilador educativo de textos", layout="wide")

@st.cache_data
def cargar_datos():
    return pd.read_csv("datos.csv")

datos = cargar_datos()

COLUMNAS_OBJETIVO = ["edad", "sexo", "extraversion", "diligencia", "simpatia", "neurosis", "franqueza"]

def entrenar(textos, etiquetas):
    modelo = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clasificador", LogisticRegression(max_iter=1000))
    ])
    modelo.fit(textos, etiquetas)
    return modelo

@st.cache_resource
def entrenar_todos(df):
    return {col: entrenar(df["texto"], df[col]) for col in COLUMNAS_OBJETIVO}

@st.cache_resource
def evaluar_todos(df):
    resultados = {}
    for col in COLUMNAS_OBJETIVO:
        X_train, X_test, y_train, y_test = train_test_split(
            df["texto"], df[col], test_size=0.3, random_state=42
        )
        modelo = entrenar(X_train, y_train)
        y_pred = modelo.predict(X_test)
        cv = cross_val_score(
            entrenar(df["texto"], df[col]), df["texto"], df[col], cv=3
        )
        resultados[col] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "cv_mean": cv.mean(),
            "cv_std": cv.std(),
            "matriz": confusion_matrix(y_test, y_pred),
            "clases": modelo.classes_,
            "reporte": classification_report(y_test, y_pred, zero_division=0),
        }
    return resultados

def palabras_clave(modelo, n=10):
    vec = modelo.named_steps["tfidf"]
    clf = modelo.named_steps["clasificador"]
    nombres = vec.get_feature_names_out()
    resultado = {}
    for i, clase in enumerate(clf.classes_):
        pesos = clf.coef_[i] if len(clf.classes_) > 2 else clf.coef_[0]
        top = sorted(zip(nombres, pesos), key=lambda x: x[1], reverse=True)[:n]
        resultado[clase] = [w for w, _ in top]
    return resultado

modelos = entrenar_todos(datos)
evaluaciones = evaluar_todos(datos)

st.title("Perfilador educativo de textos")
st.write("Aplicación didáctica con datos sintéticos. No evalúa personas reales.")

pestanas = st.tabs(["Análisis", "Métricas", "Palabras clave", "Análisis de opinión"])

with pestanas[0]:
    texto = st.text_area("Escribe una auto-descripción ficticia:")
    if st.button("Analizar"):
        if texto.strip():
            st.subheader("Resultado")
            for col in COLUMNAS_OBJETIVO:
                st.write(f"**{col.capitalize()}:**", modelos[col].predict([texto])[0])
            with st.expander("Ver probabilidades por clase"):
                for col in COLUMNAS_OBJETIVO:
                    probs = modelos[col].predict_proba([texto])[0]
                    clases = modelos[col].classes_
                    st.write(f"**{col.capitalize()}:**")
                    for c, p in zip(clases, probs):
                        st.write(f"  - {c}: {p:.2%}")
        else:
            st.warning("Escribe un texto.")

with pestanas[1]:
    st.subheader("Exactitud y validación cruzada")
    resumen = pd.DataFrame({
        col: {
            "Accuracy (test)": f"{evaluaciones[col]['accuracy']:.2%}",
            "CV media": f"{evaluaciones[col]['cv_mean']:.2%}",
            "CV desv. std": f"{evaluaciones[col]['cv_std']:.2%}",
        }
        for col in COLUMNAS_OBJETIVO
    }).T
    st.dataframe(resumen)

    st.subheader("Matrices de confusión")
    for col in COLUMNAS_OBJETIVO:
        st.write(f"**{col.capitalize()}**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            evaluaciones[col]["matriz"], annot=True, fmt="d", cmap="Blues",
            xticklabels=evaluaciones[col]["clases"],
            yticklabels=evaluaciones[col]["clases"], ax=ax
        )
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Reportes de clasificación detallados"):
        for col in COLUMNAS_OBJETIVO:
            st.write(f"**{col.capitalize()}**")
            st.code(evaluaciones[col]["reporte"])

with pestanas[2]:
    st.subheader("Palabras con mayor peso por clasificador")
    for col in COLUMNAS_OBJETIVO:
        st.write(f"**{col.capitalize()}**")
        top = palabras_clave(modelos[col], n=10)
        for clase, palabras in top.items():
            st.write(f"  - *{clase}*: {', '.join(palabras)}")

with pestanas[3]:
    st.subheader("Análisis de opinión (versión preliminar)")
    texto_op = st.text_area("Texto para analizar opinión:", key="opinion")
    if st.button("Analizar opinión"):
        if texto_op.strip():
            positivas = ["bien", "bueno", "excelente", "encanta", "feliz", "gusta", "genial", "maravilloso"]
            negativas = ["mal", "malo", "odio", "triste", "aburrido", "pésimo", "terrible", "enfada"]
            t = texto_op.lower()
            pos = sum(t.count(p) for p in positivas)
            neg = sum(t.count(n) for n in negativas)
            if pos > neg:
                st.success(f"Opinión positiva (pos={pos}, neg={neg})")
            elif neg > pos:
                st.error(f"Opinión negativa (pos={pos}, neg={neg})")
            else:
                st.info(f"Opinión neutra o ambigua (pos={pos}, neg={neg})")
        else:
            st.warning("Escribe un texto.")