import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load model
pipeline = joblib.load("netflix_cluster_pipeline.pkl")
features = joblib.load("netflix_features.pkl")

st.set_page_config(
    page_title="Netflix Content Segmentation",
    page_icon="🎬",
    layout="wide"
)

# CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827, #1f2937);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 45px;
    font-weight: 800;
    color: #ef4444;
}

.sub-title {
    text-align: center;
    color: #d1d5db;
    font-size: 18px;
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 0 25px rgba(239,68,68,0.25);
    margin-bottom: 20px;
}

.result-box {
    background: linear-gradient(135deg, #ef4444, #7f1d1d);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 25px;
    font-weight: bold;
}

label, p, div {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🎬 Netflix Content Segmentation</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Unsupervised Machine Learning Clustering App</div>", unsafe_allow_html=True)

st.write("")

# Sidebar
st.sidebar.title("🎛 Input Features")

input_data = {}

for col in features:
    input_data[col] = st.sidebar.number_input(
        label=col,
        value=0.0,
        step=1.0
    )

input_df = pd.DataFrame([input_data])

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📌 User Input Data")
    st.dataframe(input_df, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🤖 Cluster Prediction")

    if st.button("Predict Cluster"):
        cluster = pipeline.predict(input_df)[0]

        st.markdown(
            f"<div class='result-box'>Predicted Cluster: {cluster}</div>",
            unsafe_allow_html=True
        )

        if cluster == 0:
            st.info("Cluster 0: Popular or high-performing Netflix content")
        elif cluster == 1:
            st.info("Cluster 1: Average-performing Netflix content")
        elif cluster == 2:
            st.info("Cluster 2: Long-duration or feature-heavy content")
        else:
            st.info("Cluster 3: Niche or low-engagement content")

    st.markdown("</div>", unsafe_allow_html=True)

# Upload clustered data
st.markdown("---")
st.subheader("📊 Cluster Analysis Dashboard")

uploaded_file = st.file_uploader("Upload netflix_clustered_data.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.dataframe(data.head(), use_container_width=True)

    if "Cluster" in data.columns:
        fig = px.histogram(
            data,
            x="Cluster",
            title="Cluster Distribution",
            color="Cluster"
        )
        st.plotly_chart(fig, use_container_width=True)

        numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns

        selected_col = st.selectbox("Select Feature for Analysis", numeric_cols)

        fig2 = px.box(
            data,
            x="Cluster",
            y=selected_col,
            color="Cluster",
            title=f"{selected_col} by Cluster"
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption("Netflix Content Segmentation using K-Means Clustering")