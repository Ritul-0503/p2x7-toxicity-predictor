import streamlit as st
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ── Page configuration ─────────────────────────────────────────────
st.set_page_config(
    page_title="P2X7 Toxicity Predictor",
    page_icon="🧬",
    layout="centered"
)

# ── Load the trained model (cached so it only loads once) ──────────
@st.cache_resource
def load_model():
    return joblib.load('best_model_p2x7.pkl')

model = load_model()

# ── Fingerprint conversion function (same as training pipeline) ────
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp), mol

# ── App title and description ───────────────────────────────────────
st.title("🧬 P2X7 Toxicity Predictor")
st.write(
    "Enter a molecule's **SMILES string** below to predict whether it is "
    "**Active** (potent against P2X7) or **Inactive** (weak binder)."
)
st.caption(
    "Don't have a SMILES string? Look up any compound on "
    "[PubChem](https://pubchem.ncbi.nlm.nih.gov) and copy its Canonical SMILES."
)

# ── Example molecules for quick testing ─────────────────────────────
examples = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}
selected_example = st.selectbox(
    "Or pick an example molecule:",
    ["-- Type your own below --"] + list(examples.keys())
)

default_smiles = "" if selected_example == "-- Type your own below --" else examples[selected_example]

# ── Input box ─────────────────────────────────────────────────────
smiles_input = st.text_input(
    "SMILES string:",
    value=default_smiles,
    placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O"
)

# ── Predict button ────────────────────────────────────────────────
if st.button("🔬 Predict Activity", type="primary"):
    if not smiles_input.strip():
        st.warning("⚠️ Please enter a SMILES string first.")
    else:
        fp, mol = smiles_to_fingerprint(smiles_input.strip())

        if fp is None:
            st.error(
                "❌ Invalid SMILES string. Please check the format and try again.\n\n"
                "Tip: copy directly from PubChem to avoid typos."
            )
        else:
            # Run prediction
            fp_reshaped = fp.reshape(1, -1)
            label = model.predict(fp_reshaped)[0]
            proba = model.predict_proba(fp_reshaped)[0]

            # ── FIX: convert numpy float32 -> native Python float ──
            # st.progress() and st.metric() reject numpy float types
            label = int(label)
            confidence = float(proba[label]) * 100.0

            # Display molecule structure
            col1, col2 = st.columns([1, 1])
            with col1:
                img = Draw.MolToImage(mol, size=(300, 300))
                st.image(img, caption="Molecule Structure")

            with col2:
                if label == 1:
                    st.success("### 🔴 ACTIVE\n(Potent against P2X7)")
                else:
                    st.info("### 🟢 INACTIVE\n(Weak binder)")

                st.metric("Confidence", f"{confidence:.1f}%")
                st.progress(float(confidence / 100.0))  # explicit native float

            st.caption(
                "⚠️ This is a computational prediction based on a machine learning model "
                "trained on experimental data. It is intended for research screening "
                "purposes only and should not replace laboratory validation."
            )

# ── Footer ────────────────────────────────────────────────────────
st.divider()
st.caption("Built with a Random Forest / XGBoost model trained on P2X7 bioactivity data using Morgan Fingerprints.")
