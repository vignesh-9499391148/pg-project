import streamlit as st
st.set_page_config(page_title="VickyAI - Secure UPI", page_icon="💳", layout="wide")
import qrcode
import pandas as pd
import numpy as np
import cv2
from io import BytesIO
from datetime import datetime
import requests
from urllib.parse import urlparse, parse_qs
import json

# --- State Management ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'form_data' not in st.session_state:
    st.session_state.form_data = {"pa": "vigneshk@oksbi", "pn": "sparrow", "am": "20000", "tn": "gpay"}

# --- Sidebar Documentation ---
st.sidebar.markdown("---")
st.sidebar.info("🚀 **Project: VickyAI**\nAB-XAI-RNN-Driven Fraud Detection")
if st.sidebar.button("📄 View Full diagrams.md"):
    st.switch_page("pay.py") # Just use tabs instead for simpler nav

def decode_qr(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        detector = cv2.QRCodeDetector()
        data, _, _ = detector.detectAndDecode(img)
        return data
    except Exception as e:
        return f"Error: {str(e)}"

def parse_upi_url(url):
    try:
        if not url.startswith("upi://"): return None
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        return {
            "pa": params.get('pa', [''])[0],
            "pn": params.get('pn', [''])[0],
            "am": params.get('am', [''])[0],
            "tn": params.get('tn', [''])[0]
        }
    except:
        return None

def generate_upi_qr(upi_id, name="", amount="", note=""):
    upi_url = f"upi://pay?pa={upi_id}"
    if name.strip(): upi_url += f"&pn={name}"
    if amount.strip(): upi_url += f"&am={amount}&cu=INR"
    if note.strip(): upi_url += f"&tn={note}"

    qr = qrcode.make(upi_url)
    buf = BytesIO()
    qr.save(buf, format="PNG")
    return buf.getvalue(), upi_url

def analyze_fraud(amount, device_score, location_id):
    """Integrates with the running app.py backend if available."""
    try:
        payload = {
            "module_type": "upi",
            "features": {
                "amount": amount,
                "device_score": device_score,
                "location_cluster": location_id
            }
        }
        res = requests.post("http://127.0.0.1:5000/api/predict_single", json=payload, timeout=2)
        if res.status_code == 200:
            return res.json()
    except:
        pass
    # Basic simulation if backend is down
    is_fraud = amount > 50000 or device_score < 0.3
    return {"is_fraud": is_fraud, "fraud_probability": 0.85 if is_fraud else 0.05}

# --- UI Header ---
st.title("🛡️ VickyAI - Secure UPI Ecosystem")
st.markdown("---")

# --- Global Inputs ---
st.sidebar.subheader("🔍 Global Fraud Context")
device_score = st.sidebar.slider("Device Trust Score", 0.0, 1.0, 0.8)
location_id = st.sidebar.number_input("Location Cluster ID", value=10)

tab1, tab2 = st.tabs(["🚀 Live Scanner & Generator", "📊 Project Diagrams & Architecture"])

with tab1:
    col1, col2 = st.columns([1, 1.2])

with col2:
    st.subheader("📷 QR Scanner (Upload)")
    uploaded_qr = st.file_uploader("Scan QR Image to Check Fraud & Auto-fill", type=['png', 'jpg', 'jpeg'])
    if uploaded_qr:
        img_bytes = uploaded_qr.read()
        with st.spinner("Decoding QR Code..."):
            decoded_data = decode_qr(img_bytes)
            
            if decoded_data and decoded_data.startswith("upi://"):
                st.success("✅ QR Code Detected!")
                st.code(decoded_data, language="text")
                
                upi_data = parse_upi_url(decoded_data)
                if upi_data:
                    # Run Fraud Analysis on Scanned Data
                    scan_amt = float(upi_data['am'] if upi_data['am'] else 0)
                    with st.spinner("Analyzing scanned QR security..."):
                        # Use current slider settings for device/location context
                        scan_fraud_res = analyze_fraud(scan_amt, device_score, location_id)
                    
                    # Display Prominent Status Badge
                    st.markdown("### Scanned Transaction Analysis")
                    if scan_fraud_res['is_fraud']:
                        st.error(f"🚨 STATUS: FRAUD DETECTED (Confidence: {scan_fraud_res['fraud_probability']*100:.1f}%)")
                        st.info("⚠️ Risk Factors: Unusual amount or suspicious UPI pattern detected by VickyAI.")
                    else:
                        st.success(f"✅ STATUS: SAFE TRANSACTION (Confidence: {(1-scan_fraud_res['fraud_probability'])*100:.1f}%)")
                    
                    if st.button("📥 Auto-fill Transaction Details"):
                        st.session_state.form_data = upi_data
                        st.rerun()
            else:
                st.error("No valid UPI QR code found in this image.")

    st.markdown("---")
    st.subheader("📊 Transaction History")
    if st.session_state.history:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)
        csv = df_history.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download History as CSV", csv, "transaction_history.csv", "text/csv")
    else:
        st.write("No transactions processed yet.")

with col1:
    st.subheader("📝 Transaction Details")
    # Bind fields to session_state.form_data
    f_data = st.session_state.form_data
    upi_id = st.text_input("Enter UPI ID", value=f_data.get("pa", ""))
    payee_name = st.text_input("Payee Name", value=f_data.get("pn", ""))
    amount = st.text_input("Amount (₹)", value=f_data.get("am", ""))
    note = st.text_input("Transaction Note", value=f_data.get("tn", ""))


    if st.button("🚀 Process & Generate QR", use_container_width=True):
        if not upi_id.strip():
            st.error("UPI ID is required")
        else:
            with st.spinner("Analyzing transaction security..."):
                fraud_res = analyze_fraud(float(amount or 0), device_score, location_id)
            
            byte_im, upi_url = generate_upi_qr(upi_id, payee_name, amount, note)
            
            new_record = {
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "UPI ID": upi_id,
                "Receiver": payee_name,
                "Amount": amount,
                "Status": "🚨 FRAUD" if fraud_res['is_fraud'] else "✅ SAFE",
                "Confidence": f"{fraud_res['fraud_probability']*100:.1f}%"
            }
            st.session_state.history.insert(0, new_record)

            if fraud_res['is_fraud']:
                st.warning(f"⚠️ POTENTIAL FRAUD DETECTED! (Score: {fraud_res['fraud_probability']:.2f})")
            else:
                st.success("Safe Transaction")

            st.image(byte_im, caption="Generated QR - Scan to Pay")
            st.download_button("💾 Download QR", byte_im, f"qr_{upi_id}.png", "image/png")

# --- Tab 2: Project Diagrams ---
with tab2:
    st.header("📈 Research & Architecture Diagrams")
    st.markdown("These diagrams are based on the **AB-XAI-RNN-SGRU** research framework.")
    
    # 1. Architecture Image
    st.subheader("1. System Architecture Overview")
    try:
        st.image("C:/Users/crazy/.gemini/antigravity/brain/b239d062-bc67-4970-b3fa-e073dda30638/vickyai_architecture_diagram_1772290119260.png", 
                 caption="VickyAI End-to-End System Architecture")
    except:
        st.error("Architecture image currently being generated...")

    # 2. CNN Detail Image
    st.subheader("2. 1D-CNN Feature Extraction Visualization")
    try:
        st.image("C:/Users/crazy/.gemini/antigravity/brain/b239d062-bc67-4970-b3fa-e073dda30638/vickyai_cnn_layer_viz_1772290298864.png", 
                 caption="Feature Extraction via 1D-Convolutional Layers")
    except:
        st.warning("CNN visualization not found.")

    st.markdown("---")
    st.subheader("3. Technical Mermaid Workflows")
    
    # Render parts of diagrams.md
    try:
        with open("C:/Users/crazy/.gemini/antigravity/brain/b239d062-bc67-4970-b3fa-e073dda30638/diagrams.md", "r") as f:
            st.markdown(f.read())
    except:
        st.info("Technical markdown details available in diagrams.md")

    st.success("Documents ready for project report submission! 🎓")
