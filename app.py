import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from scipy import stats


st.set_page_config(page_title="Distribution Fitting Tool", layout="wide")

st.markdown("""
<style>
    body {font-family: 'Arial', sans-serif;}
    .main {background-color: #fafafa;}
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-size: 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4c8bf5 !important;
        color: white !important;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Advanced Distribution Fitting Webapp")


dist_list = {
    "Normal": stats.norm,
    "Gamma": stats.gamma,
    "Exponential": stats.expon,
    "Weibull": stats.weibull_min,
    "Lognormal": stats.lognorm,
    "Beta": stats.beta,
    "Uniform": stats.uniform,
    "Pareto": stats.pareto,
    "Rayleigh": stats.rayleigh,
    "Chi-square": stats.chi2,
    "F": stats.f,
    "T": stats.t,
    "Cauchy": stats.cauchy,
    "Laplace": stats.laplace,
    "Logistic": stats.logistic,
    "Gumbel": stats.gumbel_r,
    "Inverse Gamma": stats.invgamma,
    "Nakagami": stats.nakagami,
    "Rice": stats.rice,
    "Powerlaw": stats.powerlaw,
}


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📥 Data Input", "⚙ Automatic Fit", "🛠 Manual Fit",
     "🏆 Best Distribution", "📤 Export"]
)

with tab1:
    st.header("📥 Enter or Upload Data")

    input_method = st.radio("Choose Input Method:", ["Manual Entry", "Upload CSV"])

    if input_method == "Manual Entry":
        txt = st.text_area("Enter numbers separated by commas",
                           "1,2,3,4,5,6")
        try:
            data = np.array([float(x) for x in txt.split(",")])
        except:
            st.error("Invalid format.")
            st.stop()
    else:
        uploaded = st.file_uploader("Upload CSV File")
        if uploaded:
            df = pd.read_csv(uploaded)
            data = df.iloc[:, 0].dropna().to_numpy()
        else:
            st.warning("Upload a file to continue.")
            st.stop()

    st.subheader("📌 Quick Data Preview")
    st.write(data[:20])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(data, "k.")
    ax.set_title("Raw Data Scatter Plot")
    st.pyplot(fig)


def compute_errors(data, dist_obj):
    params = dist_obj.fit(data)
    dist = dist_obj(*params)

    
    hist_counts, hist_bins = np.histogram(data, bins=30, density=True)
    hist_centers = (hist_bins[:-1] + hist_bins[1:]) / 2

    errors = np.abs(hist_counts - dist.pdf(hist_centers))
    return params, np.mean(errors)


with tab2:
    st.header("⚙ Automatic Distribution Fit")

    selected = st.selectbox("Choose Distribution", list(dist_list.keys()))
    dist_obj = dist_list[selected]

    bins = st.slider("Histogram Bins", 10, 100, 30)

    params = dist_obj.fit(data)
    fitted = dist_obj(*params)

 
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(data, bins=bins, density=True, alpha=0.6, color="orange")
    x = np.linspace(min(data), max(data), 300)
    ax.plot(x, fitted.pdf(x), "b", lw=2)
    ax.set_title(f"{selected} Fit")
    st.pyplot(fig)

   
    _, avg_err = compute_errors(data, dist_obj)
    st.success(f"📉 Average Error: **{avg_err:.5f}**")

    st.write("**Fitted Parameters:**")
    st.write(params)


with tab3:
    st.header("🛠 Manual Parameter Adjustment")

    dist_name = st.selectbox("Distribution", list(dist_list.keys()), key="manual_dist")
    dist_obj = dist_list[dist_name]
    auto_params = dist_obj.fit(data)

    st.write("Auto Parameters:", auto_params)

    sliders = []
    for i, p in enumerate(auto_params):
        sliders.append(
            st.slider(f"Parameter {i+1}",
                      p * 0.2, p * 3, p)
        )

    manual_dist = dist_obj(*sliders)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.hist(data, bins=30, density=True, alpha=0.6, color="orange")
    x = np.linspace(min(data), max(data), 300)
    ax2.plot(x, manual_dist.pdf(x), "r--", lw=2)
    ax2.set_title("Manual Adjustment Fit")
    st.pyplot(fig2)


with tab4:
    st.header("🏆 Best Fitting Distribution")

    results = []
    for name, d in dist_list.items():
        try:
            params, err = compute_errors(data, d)
            results.append((name, err, params))
        except:
            pass

    results.sort(key=lambda x: x[1])
    best_name, best_err, best_params = results[0]

    st.success(f"Best Fit: **{best_name}** (Error = {best_err:.5f})")
    st.write("Parameters:", best_params)


with tab5:
    st.header("📤 Export Tools")

    
    df_params = pd.DataFrame({
        "Distribution": [selected],
        "Parameters": [params]
    })

    csv = df_params.to_csv(index=False).encode("utf-8")
    st.download_button("Download Parameters CSV", csv, "params.csv", "text/csv")

   
    img = BytesIO()
    fig.savefig(img, format="png")
    st.download_button("Download Fit Plot (PNG)", img.getvalue(), "fit.png", "image/png")

st.info("You can now upload `app.py` to LEARN. For bonus marks, deploy on Streamlit Cloud.")

    
    
    
    