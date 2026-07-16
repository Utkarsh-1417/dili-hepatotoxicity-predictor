"""
DILI Hepatotoxicity Prediction App
------------------------------------
Streamlit web app for predicting hepatotoxicity (Drug-Induced Liver Injury)
from molecular SMILES strings, using a trained SVM model.

Run locally with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from fpdf import FPDF

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="DILI Hepatotoxicity Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS — clinical-modern theme
# ============================================================
st.markdown("""
<style>
    /* Overall background */
    .stApp {
        background-color: #F7F9FA;
    }

    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #0B5563 0%, #147A8C 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(11, 85, 99, 0.25);
    }
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2.1rem;
        margin-bottom: 0.3rem;
        font-weight: 700;
    }
    .main-header p {
        color: #D6ECEF;
        font-size: 1rem;
        margin: 0;
    }

    /* Result cards */
    .result-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-left: 6px solid #147A8C;
        margin-bottom: 1rem;
    }

    /* Risk badges */
    .badge-toxic {
        background-color: #FDECEC;
        color: #B3261E;
        border: 1.5px solid #B3261E;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }
    .badge-safe {
        background-color: #E8F5E9;
        color: #1B7A3D;
        border: 1.5px solid #1B7A3D;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
        display: inline-block;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0B5563;
    }
    section[data-testid="stSidebar"] * {
        color: #F0F7F8 !important;
    }

    /* Metric boxes */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Buttons */
    .stButton > button, .stDownloadButton > button {
        background-color: #147A8C;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #0B5563;
        color: white;
    }

    /* Footer disclaimer */
    .disclaimer {
        background-color: #FFF8E1;
        border-left: 5px solid #F5A623;
        padding: 0.9rem 1.2rem;
        border-radius: 8px;
        font-size: 0.88rem;
        color: #6B5300;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL ARTIFACTS (cached so it only loads once)
# ============================================================
@st.cache_resource
def load_artifacts():
    imputer = joblib.load("artifacts/imputer.pkl")
    scaler = joblib.load("artifacts/scaler.pkl")
    pca = joblib.load("artifacts/pca.pkl")
    final_model = joblib.load("artifacts/final_model_svm.pkl")
    fp_cols = joblib.load("artifacts/fp_cols.pkl")
    return imputer, scaler, pca, final_model, fp_cols

imputer, scaler, pca, final_model, fp_cols = load_artifacts()
chooser = LargestFragmentChooser()
imputer_expected_cols = list(imputer.feature_names_in_)
scaler_expected_cols = list(scaler.feature_names_in_)

# ============================================================
# PREDICTION PIPELINE (mirrors training exactly)
# ============================================================
def predict_hepatotoxicity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {'status': 'error', 'message': 'Invalid SMILES string — could not be parsed.'}

    mol = chooser.choose(mol)
    canonical_smiles = Chem.MolToSmiles(mol)

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_arr = np.zeros((2048,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
    fp_dict = {f'Morgan_{i}': fp_arr[i] for i in range(2048)}

    desc_values = {}
    for name, func in Descriptors._descList:
        try:
            desc_values[name] = func(mol)
        except Exception:
            desc_values[name] = np.nan

    row = {**fp_dict, **desc_values}
    X_new = pd.DataFrame([row])
    X_new = X_new[imputer_expected_cols]
    X_new = X_new.replace([np.inf, -np.inf], np.nan)

    X_new_imp = pd.DataFrame(imputer.transform(X_new), columns=imputer_expected_cols)
    if 'Ipc' in X_new_imp.columns:
        X_new_imp = X_new_imp.drop(columns=['Ipc'])

    X_desc_scaled = pd.DataFrame(
        scaler.transform(X_new_imp[scaler_expected_cols]),
        columns=scaler_expected_cols
    )
    fp_present = [c for c in fp_cols if c in X_new_imp.columns]
    X_scaled = pd.concat([X_new_imp[fp_present].reset_index(drop=True), X_desc_scaled], axis=1)

    X_pca = pca.transform(X_scaled)
    prediction = final_model.predict(X_pca)[0]
    probability = final_model.predict_proba(X_pca)[0, 1]

    return {
        'status': 'success',
        'input_smiles': smiles,
        'canonical_smiles': canonical_smiles,
        'mol': mol,
        'prediction': 'Hepatotoxic' if prediction == 1 else 'Non-hepatotoxic',
        'probability_hepatotoxic': round(float(probability), 4)
    }


def predict_batch(smiles_list):
    results = []
    for smi in smiles_list:
        r = predict_hepatotoxicity(str(smi).strip())
        if r['status'] == 'success':
            results.append({
                'Input_SMILES': smi,
                'Canonical_SMILES': r['canonical_smiles'],
                'Prediction': r['prediction'],
                'Probability_Hepatotoxic': r['probability_hepatotoxic'],
                'Status': 'OK'
            })
        else:
            results.append({
                'Input_SMILES': smi,
                'Canonical_SMILES': None,
                'Prediction': None,
                'Probability_Hepatotoxic': None,
                'Status': r['message']
            })
    return pd.DataFrame(results)

# ============================================================
# PDF REPORT GENERATION
# ============================================================
def generate_pdf_report(df, title="Hepatotoxicity Batch Prediction Report"):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(11, 85, 99)
    pdf.cell(0, 12, title, ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    valid_df = df[df['Status'] == 'OK']
    n_total = len(df)
    n_valid = len(valid_df)
    n_toxic = (valid_df['Prediction'] == 'Hepatotoxic').sum() if n_valid else 0
    n_safe = n_valid - n_toxic
    pct_toxic = (n_toxic / n_valid * 100) if n_valid else 0

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, "Summary Statistics", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Total compounds submitted: {n_total}", ln=True)
    pdf.cell(0, 7, f"Successfully processed: {n_valid}", ln=True)
    pdf.cell(0, 7, f"Predicted Hepatotoxic: {n_toxic} ({pct_toxic:.1f}%)", ln=True)
    pdf.cell(0, 7, f"Predicted Non-hepatotoxic: {n_safe} ({100 - pct_toxic:.1f}%)", ln=True)
    if n_total - n_valid > 0:
        pdf.cell(0, 7, f"Invalid / unparseable entries: {n_total - n_valid}", ln=True)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Detailed Results", ln=True)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(20, 122, 140)
    pdf.set_text_color(255, 255, 255)
    col_widths = [70, 35, 35, 45]
    headers = ["SMILES", "Prediction", "Probability", "Status"]
    for w, h in zip(col_widths, headers):
        pdf.cell(w, 8, h, border=1, fill=True)
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(30, 30, 30)
    for _, row in df.iterrows():
        smi = str(row['Input_SMILES'])[:38]
        pred = str(row['Prediction']) if pd.notna(row['Prediction']) else "-"
        prob = f"{row['Probability_Hepatotoxic']:.3f}" if pd.notna(row['Probability_Hepatotoxic']) else "-"
        status = str(row['Status'])[:28]

        if pred == "Hepatotoxic":
            pdf.set_text_color(179, 38, 30)
        elif pred == "Non-hepatotoxic":
            pdf.set_text_color(27, 122, 61)
        else:
            pdf.set_text_color(120, 120, 120)

        pdf.cell(col_widths[0], 7, smi, border=1)
        pdf.cell(col_widths[1], 7, pred, border=1)
        pdf.cell(col_widths[2], 7, prob, border=1)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(col_widths[3], 7, status, border=1)
        pdf.ln()

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, "Disclaimer: This tool provides a computational screening estimate based on a "
                          "machine learning model (SVM, external validation MCC ~0.33). It is intended for "
                          "research and educational purposes only and is not a substitute for experimental "
                          "toxicology testing or clinical judgment.")

    return bytes(pdf.output(dest='S'))

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🧪 About")
    st.markdown(
        "This app predicts **Drug-Induced Liver Injury (DILI)** risk "
        "from a molecule's SMILES string using a trained machine learning model."
    )
    st.markdown("---")
    st.markdown("**What is DILI?**")
    st.markdown(
        "Drug-Induced Liver Injury (DILI) is liver damage caused by "
        "medications, supplements, or chemical compounds. It is one of "
        "the most common reasons drugs fail in clinical trials or get "
        "withdrawn from the market after approval."
    )
    st.markdown(
        "The liver is the body's main site of drug metabolism, which "
        "makes it especially vulnerable — some compounds are converted "
        "into reactive, toxic byproducts during this process. DILI can "
        "range from mild, reversible enzyme elevations to severe liver "
        "failure in rare cases."
    )
    st.markdown(
        "Predicting DILI risk early, from a molecule's structure alone, "
        "can help prioritize safer candidates before costly experimental "
        "testing begins."
    )
    st.markdown("---")
    st.markdown("Built as a learning/portfolio project in cheminformatics + ML.")

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🧪 DILI Hepatotoxicity Predictor</h1>
    <p>Predict Drug-Induced Liver Injury risk from molecular structure (SMILES)</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# TABS
# ============================================================
tab1, tab2 = st.tabs(["🔬 Single Molecule", "📄 Batch Upload (CSV)"])

# ---------------- SINGLE MOLECULE TAB ----------------
with tab1:
    col_input, col_example = st.columns([3, 1])
    with col_input:
        smiles_input = st.text_input(
            "Enter a SMILES string",
            placeholder="e.g. CC(=O)NC1=CC=C(O)C=C1  (Paracetamol)"
        )
    with col_example:
        st.markdown("<br>", unsafe_allow_html=True)
        example = st.selectbox("Or try an example:", [
            "", "Aspirin", "Paracetamol", "Ethanol", "Caffeine"
        ])

    example_smiles = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "Ethanol": "CCO",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    }
    if example and not smiles_input:
        smiles_input = example_smiles[example]
        st.info(f"Using example: **{example}** → `{smiles_input}`")

    if st.button("🔍 Predict Hepatotoxicity", key="single_predict"):
        if not smiles_input.strip():
            st.warning("Please enter a SMILES string first.")
        else:
            with st.spinner("Analyzing molecular structure..."):
                result = predict_hepatotoxicity(smiles_input.strip())

            if result['status'] == 'error':
                st.error(result['message'])
            else:
                colA, colB = st.columns([1, 1.4])

                with colA:
                    img = Draw.MolToImage(result['mol'], size=(350, 350))
                    st.image(img, caption="Molecular structure")

                with colB:
                    badge_class = "badge-toxic" if result['prediction'] == "Hepatotoxic" else "badge-safe"
                    icon = "⚠️" if result['prediction'] == "Hepatotoxic" else "✅"
                    st.markdown(f"""
                    <div class="result-card">
                        <p style="color:#555; margin-bottom:4px;">Prediction</p>
                        <span class="{badge_class}">{icon} {result['prediction']}</span>
                        <p style="color:#555; margin-top:18px; margin-bottom:4px;">Canonical SMILES</p>
                        <code>{result['canonical_smiles']}</code>
                    </div>
                    """, unsafe_allow_html=True)

                    prob = result['probability_hepatotoxic']
                    st.markdown("**Probability of Hepatotoxicity**")
                    st.progress(prob)
                    st.markdown(f"<h3 style='color:#0B5563;'>{prob*100:.1f}%</h3>", unsafe_allow_html=True)

                st.markdown("""
                <div class="disclaimer">
                ⚠️ <b>Disclaimer:</b> This is a computational screening estimate from a machine learning model
                (external validation MCC ≈ 0.33). It is intended for research/educational purposes only and
                should not be used as the sole basis for safety decisions.
                </div>
                """, unsafe_allow_html=True)

# ---------------- BATCH UPLOAD TAB ----------------
with tab2:
    st.markdown("Upload a CSV file with a column named **`SMILES`** containing one molecule per row.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            input_df = None

        if input_df is not None:
            if 'SMILES' not in input_df.columns:
                st.error("CSV must contain a column named 'SMILES'.")
            else:
                st.success(f"Loaded {len(input_df)} compounds.")
                if st.button("🔍 Run Batch Prediction", key="batch_predict"):
                    progress_bar = st.progress(0, text="Processing compounds...")
                    smiles_list = input_df['SMILES'].tolist()

                    results = []
                    for i, smi in enumerate(smiles_list):
                        r = predict_hepatotoxicity(str(smi).strip())
                        if r['status'] == 'success':
                            results.append({
                                'Input_SMILES': smi,
                                'Canonical_SMILES': r['canonical_smiles'],
                                'Prediction': r['prediction'],
                                'Probability_Hepatotoxic': r['probability_hepatotoxic'],
                                'Status': 'OK'
                            })
                        else:
                            results.append({
                                'Input_SMILES': smi,
                                'Canonical_SMILES': None,
                                'Prediction': None,
                                'Probability_Hepatotoxic': None,
                                'Status': r['message']
                            })
                        progress_bar.progress((i + 1) / len(smiles_list), text=f"Processing {i+1}/{len(smiles_list)}...")

                    results_df = pd.DataFrame(results)
                    st.session_state['batch_results'] = results_df
                    progress_bar.empty()

    if 'batch_results' in st.session_state:
        results_df = st.session_state['batch_results']
        valid_df = results_df[results_df['Status'] == 'OK']

        n_total = len(results_df)
        n_valid = len(valid_df)
        n_toxic = (valid_df['Prediction'] == 'Hepatotoxic').sum() if n_valid else 0
        n_safe = n_valid - n_toxic

        st.markdown("### Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total submitted", n_total)
        m2.metric("Successfully processed", n_valid)
        m3.metric("Predicted Hepatotoxic", f"{n_toxic} ({n_toxic/n_valid*100:.1f}%)" if n_valid else "0")
        m4.metric("Predicted Non-hepatotoxic", f"{n_safe} ({n_safe/n_valid*100:.1f}%)" if n_valid else "0")

        st.markdown("### Results Table")
        st.dataframe(results_df, use_container_width=True)

        col_csv, col_pdf = st.columns(2)
        with col_csv:
            csv_bytes = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Results (CSV)",
                data=csv_bytes,
                file_name=f"hepatotoxicity_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        with col_pdf:
            pdf_bytes = generate_pdf_report(results_df)
            st.download_button(
                "⬇️ Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"hepatotoxicity_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )

        st.markdown("""
        <div class="disclaimer">
        ⚠️ <b>Disclaimer:</b> These are computational screening estimates from a machine learning model
        (external validation MCC ≈ 0.33). Intended for research/educational purposes only.
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# PERMANENT FOOTER — Laboratory Verification Disclaimer
# ============================================================
st.markdown("---")
st.markdown("""
<div class="disclaimer" style="margin-top: 1.5rem;">
🧫 <b>Important — Laboratory Verification Required:</b> All predictions from this tool are
computational estimates based on a machine learning model trained on historical data. They are
<b>not a substitute for experimental toxicology testing</b>. Any compound flagged here — toxic or
non-toxic — should be verified through appropriate in vitro or in vivo laboratory studies before
any real-world safety, clinical, or regulatory decision is made. This tool is intended solely for
early-stage research screening and educational purposes.
</div>
""", unsafe_allow_html=True)
