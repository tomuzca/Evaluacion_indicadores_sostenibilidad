import streamlit as st
import pandas as pd
import numpy as np
import datetime
import io
import matplotlib.pyplot as plt
from matplotlib import cm
import os

# ---------- CONFIG ----------
CSV_PATH = os.path.join(os.path.dirname(__file__), "dimensiones.csv")
APP_TITLE = "Evaluador de Sostenibilidad Agrícola"
DATE_FMT = "%Y-%m-%d"
# ----------------------------

# ---- Configuración de página ----
st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---- Pantalla de bienvenida ----
st.title(APP_TITLE)
st.write(
    """
    Bienvenido/a al **Evaluador de Sostenibilidad Agrícola** 🌿  
    Este formulario le permitirá registrar las prácticas agrícolas de su predio y obtener un **diagnóstico visual**
    del nivel de sostenibilidad, representado en un gráfico radial.

    Complete las preguntas, y al final podrá **descargar sus resultados** (Excel y gráfico PNG con fecha).
    """
)
st.divider()

# ---- Cargar archivo CSV de indicadores ----
@st.cache_data
def load_indicators(path: str) -> pd.DataFrame:
    """
    Carga el archivo de indicadores desde la ruta especificada.
    Verifica la existencia del archivo y las columnas requeridas.
    Devuelve un DataFrame listo para su uso en la aplicación.
    """
    if not os.path.exists(path):
        st.stop()  # detiene la app si no encuentra el archivo
    df = pd.read_csv(path)

    # Validación de columnas requeridas
    expected_cols = {"dimension", "indicador", "opciones", "puntajes"}
    missing = expected_cols - set(df.columns)
    if missing:
        st.error(f"El archivo CSV no tiene las columnas requeridas: {', '.join(missing)}")
        st.stop()

    return df

# Intentar cargar el CSV (sin mostrar mensajes)
try:
    df_ind = load_indicators(CSV_PATH)
except Exception as e:
    st.error("Ocurrió un error al cargar el archivo de indicadores. "
             "Verifica el CSV antes de continuar.")
    st.exception(e)
    st.stop()


# ---- Función para parsear opciones ----
def parse_row_options(options_str, puntajes_str):
    opts = [o.strip() for o in options_str.split(";")] if pd.notna(options_str) else []
    pts = [p.strip() for p in puntajes_str.split(";")] if pd.notna(puntajes_str) else []
    pairs = []
    for i in range(min(len(opts), len(pts))):
        try:
            score = int(pts[i])
        except:
            import re
            m = re.match(r"(\d+)", pts[i])
            score = int(m.group(1)) if m else 0
        pairs.append((score, opts[i]))
    return pairs

# ---- Preparar dataframe ----
df_ind["parsed_options"] = df_ind.apply(
    lambda r: parse_row_options(r["opciones"], r["puntajes"]), axis=1
)
df_ind["max_indicador"] = df_ind["parsed_options"].apply(
    lambda x: max([p for p, _ in x]) if x else 0
)

# ---- PRESERVAR ORDEN ORIGINAL DE DIMENSIONES ----
# Crear una lista con el orden original de aparición de las dimensiones
dimension_order = df_ind["dimension"].drop_duplicates().tolist()

# ---- Información del usuario ----
st.subheader("Datos del evaluador")
col1, col2, col3 = st.columns(3)
with col1:
    productor = st.text_input("Nombre del productor / agricultor")
with col2:
    predio = st.text_input("Nombre del predio")
with col3:
    fecha_input = st.date_input("Fecha de la evaluación", value=datetime.date.today())

demo_mode = st.checkbox("Modo demostración (rellenar con valores de ejemplo)", value=False)
st.divider()

# ---- Formulario ----
st.subheader("Formulario de indicadores")
responses = {}

with st.form("formulario_indicadores"):
    # Iterar sobre las dimensiones en el orden original del CSV
    for dim in dimension_order:
        group = df_ind[df_ind["dimension"] == dim]
        st.markdown(f"### 🌱 {dim}")
        for _, row in group.iterrows():
            opciones = [f"{p} — {t}" for p, t in row["parsed_options"]]
            default_idx = len(opciones) - 1 if demo_mode and opciones else 0
            seleccion = st.radio(row["indicador"], opciones, index=default_idx, key=row.name)
            puntaje = int(seleccion.split("—")[0].strip())
            responses[row["indicador"]] = {"dimension": dim, "puntaje": puntaje}
    enviado = st.form_submit_button("Enviar evaluación")

# ---- Cálculo de resultados ----
if enviado:
    st.success("Evaluación registrada correctamente ✅")

    df_resp = pd.DataFrame.from_dict(responses, orient="index").reset_index()
    df_resp = df_resp.rename(columns={"index": "indicador"})

    subtotal = df_resp.groupby("dimension")["puntaje"].sum().reset_index()
    maximos = df_ind.groupby("dimension")["max_indicador"].sum().reset_index()
    resumen = subtotal.merge(maximos, on="dimension", how="left")
    resumen["valor_normalizado"] = (resumen["puntaje"] / resumen["max_indicador"]) * 100

    # ---- ORDENAR RESUMEN SEGÚN ORDEN ORIGINAL ----
    resumen["dimension"] = pd.Categorical(resumen["dimension"], categories=dimension_order, ordered=True)
    resumen = resumen.sort_values("dimension").reset_index(drop=True)

    st.subheader("Resumen por dimensión")
    st.dataframe(resumen)

    # ---- Gráfico radial ----
    st.subheader("Gráfico radial de sostenibilidad")

    dim_names = resumen["dimension"].tolist()
    values = resumen["valor_normalizado"].tolist()

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    theta = np.linspace(0, 2 * np.pi, len(dim_names), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    theta = np.concatenate((theta, [theta[0]]))

    # Colores: marrón (bajo) → verde (alto)
    colors = [cm.get_cmap("YlGn")(v / 100) for v in values]
    for i in range(len(values) - 1):
        ax.plot([theta[i], theta[i + 1]], [values[i], values[i + 1]], color=colors[i], linewidth=2)

    ax.fill(theta, values, color="green", alpha=0.2)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(dim_names, fontsize=7)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"])
    ax.set_ylim(0, 100)
    ax.set_title("Nivel de sostenibilidad por dimensión", pad=30)
    st.pyplot(fig)

    # ---- Función para sanitizar nombres de archivo ----
    def sanitize_filename(text):
        """Elimina o reemplaza caracteres no válidos en nombres de archivo"""
        if not text:
            return ""
        # Reemplazar espacios por guiones bajos y eliminar caracteres especiales
        import re
        text = text.strip()
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'[-\s]+', '_', text)
        return text

    # ---- Construir nombre de archivo ----
    fecha_str = fecha_input.strftime(DATE_FMT)
    
    # Sanitizar nombres
    productor_clean = sanitize_filename(productor) if productor else ""
    predio_clean = sanitize_filename(predio) if predio else ""
    
    # Construir nombre base del archivo
    nombre_partes = []
    if productor_clean:
        nombre_partes.append(productor_clean)
    if predio_clean:
        nombre_partes.append(predio_clean)
    if not nombre_partes:
        nombre_partes.append("evaluacion")
    nombre_partes.append(fecha_str)
    
    nombre_base = "_".join(nombre_partes)
    
    xlsx_filename = f"{nombre_base}.xlsx"
    png_filename = f"{nombre_base}.png"

    # ---- Excel (XLSX) ----
    out_xlsx = io.BytesIO()
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
        # Crear hoja con información general
        info_df = pd.DataFrame({
            'Campo': ['Productor/Agricultor', 'Predio', 'Fecha de Evaluación'],
            'Valor': [productor or 'No especificado', predio or 'No especificado', fecha_str]
        })
        info_df.to_excel(writer, sheet_name='Información', index=False)
        
        # Crear hoja con resumen de resultados
        resumen.to_excel(writer, sheet_name='Resumen por Dimensión', index=False)
        
        # Crear hoja con respuestas detalladas
        df_resp.to_excel(writer, sheet_name='Respuestas Detalladas', index=False)
    
    out_xlsx.seek(0)
    st.download_button(
        "📥 Descargar resultados (Excel)", 
        data=out_xlsx.getvalue(), 
        file_name=xlsx_filename, 
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # ---- PNG ----
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.download_button("📷 Descargar gráfico (PNG)", data=buf, file_name=png_filename, mime="image/png")

    st.info(f"Los archivos incluyen el nombre del productor, predio y la fecha en el nombre: {nombre_base}")