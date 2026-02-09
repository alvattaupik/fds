import streamlit as st
import pandas as pd
import joblib 
import numpy as np
import time
import json
import base64
from streamlit_option_menu import option_menu

# ==========================================
# 0. FUNGSI UTILITAS (LOGO)
# ==========================================
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

# Pastikan path file gambar benar
img_base64 = get_base64_of_bin_file('assets/images/jaga logo.jpg')

img_profile_base64 = get_base64_of_bin_file('assets/images/profile.jpg')

# ==========================================
# 1. KONFIGURASI HALAMAN & THEME
# ==========================================
st.set_page_config(
    page_title="JAGA | Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #fcfcfd;
    }

    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #f0f0f5;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px rgba(0,0,0,0.05);
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4F46E5 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }

    .status-card {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .fraud-bg {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border-left: 8px solid #e53e3e;
    }
    .safe-bg {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border-left: 8px solid #38a169;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    folder_path = 'models/v1_2/' 
    try:
        assets = {
            'preprocessor': joblib.load(f'{folder_path}preprocessor.pkl'),
            'iso_forest': joblib.load(f'{folder_path}iso_forest_layer.pkl'),
            'xgb_model': joblib.load(f'{folder_path}model_fraud_xgb.pkl'),
        }
        with open(f'{folder_path}model_metadata.json', 'r') as f:
            assets['metrics'] = json.load(f)
        return assets
    except Exception as e:
        st.error(f"Gagal memuat model: {e}") # Munculkan pesan error di UI
        return None
    
    
# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    # Logo Header Custom
    st.markdown(f"""
        <div style='
            text-align: center; 
            padding: 25px 20px; 
            background: linear-gradient(180deg, rgba(59, 130, 246, 0.15) 0%, rgba(17, 24, 39, 0) 100%);
            border-radius: 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        '>
            <div style='
                display: flex; 
                justify-content: center; 
                margin-bottom: 15px;
            '>
                <div style='
                    background: white; 
                    padding: 10px; 
                    border-radius: 20px; 
                    box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
                '>
                    <img src="data:image/png;base64,{img_base64}" width="60" style="border-radius: 5px;">
                </div>
            </div>
            <h1 style='
                font-family: "Inter", sans-serif;
                color: white; 
                margin: 0; 
                letter-spacing: 4px; 
                font-size: 24px;
                font-weight: 800;
                text-shadow: 0 2px 10px rgba(59, 130, 246, 0.5);
            '>JAGA</h1>
            <p style='
                color: #94a3b8; 
                font-size: 11px; 
                text-transform: uppercase; 
                letter-spacing: 1.5px;
                margin-top: 5px;
                font-weight: 600;
            '>Hybrid Fraud Shield</p>
        </div>
    """, unsafe_allow_html=True)
    
    selected_page = option_menu(
        menu_title=None,
        options=["Deteksi Fraud", "Penjelasan Model", "Tentang Dataset", "About Me"],
        icons=["shield-check", "cpu", "database", "person-badge"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "transparent"},
            "icon": {"color": "#94a3b8", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"5px", "--hover-color": "#1f2937"},
            "nav-link-selected": {"background-color": "#3b82f6"},
        }
    )

# ==========================================
# 4. LOGIKA HALAMAN
# ==========================================

# ------------------------------------------
# HALAMAN 1: DETEKSI FRAUD (DASHBOARD)
# ------------------------------------------
if selected_page == "Deteksi Fraud":
    st.title("🛡️ Dashboard Deteksi Fraud")
    st.markdown("Simulasikan transaksi di bawah ini untuk melihat prediksi keamanan secara *real-time*.")
    
    # Grid untuk Metrics Utama
    if assets and 'metrics' in assets:
        full_metadata = assets['metrics']
        m = full_metadata['metrics']
        cols = st.columns(3)
        cols[0].metric("Precision", f"{m['precision']:.1%}", "High")
        cols[1].metric("Recall", f"{m['recall']:.1%}", "Sensitivity")
        cols[2].metric("F1-Score", f"{m['f1_score']:.1%}", "Balanced")
    
    st.divider()

    # --- INPUT FORM DI HALAMAN UTAMA (GRID SYSTEM) ---
    with st.container(border=True):
        st.subheader("📝 Input Parameter Transaksi")
        
        with st.form("transaction_form"):
            # Baris 1: Informasi Umum (3 Kolom)
            col_info1, col_info2, col_info3 = st.columns(3)
            
            with col_info1:
                st.markdown("##### 🕒 Waktu & Jenis")
                hour_val = st.slider("Jam Transaksi", 0, 23, 12, help="Jam berapa transaksi dilakukan?")
                type_trans = st.selectbox("Tipe Transaksi", ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"])
            
            with col_info2:
                st.markdown("##### 📤 Detail Pengirim")
                old_org = st.number_input("Saldo Awal Pengirim", min_value=0.0, value=5000.0, format="%.2f")
                new_org = st.number_input("Saldo Akhir Pengirim", min_value=0.0, value=3500.0, format="%.2f")
            
            with col_info3:
                st.markdown("##### 📥 Detail Penerima")
                old_dest = st.number_input("Saldo Awal Penerima", min_value=0.0, value=0.0, format="%.2f")
                new_dest = st.number_input("Saldo Akhir Penerima", min_value=0.0, value=1500.0, format="%.2f")

            st.markdown("---")
            
            # Baris 2: Nominal & Tombol (Layout Asimetris)
            c_amount, c_button = st.columns([2, 1])
            
            with c_amount:
                 amount = st.number_input("💵 Nominal Transaksi (USD)", min_value=0.0, value=1500.0, step=100.0, format="%.2f")
            
            with c_button:
                st.write("") # Spacer layout
                st.write("") 
                submit = st.form_submit_button("🔍 ANALISIS RISIKO", use_container_width=True)

    # Logika Prediksi
    if submit and assets:
        with st.status("Melakukan pemindaian keamanan...", expanded=True) as status:
            st.write("Mengekstrak fitur transaksi...")
            time.sleep(0.4)
            st.write("Menghitung skor anomali (Isolation Forest)...")
            time.sleep(0.4)
            st.write("Klasifikasi risiko final (XGBoost)...")
            
            # Data Prep
            input_df = pd.DataFrame({
                'step': [hour_val], 'type': [type_trans], 'amount': [amount],
                'oldbalanceOrg': [old_org], 'newbalanceOrig': [new_org],
                'oldbalanceDest': [old_dest], 'newbalanceDest': [new_dest]
            })
            
            # Feature Engineering sederhana
            input_df['hour'] = input_df['step'] % 24
            input_df['errorBalanceOrig'] = input_df['newbalanceOrig'] + input_df['amount'] - input_df['oldbalanceOrg']
            input_df['errorBalanceDest'] = input_df['oldbalanceDest'] + input_df['amount'] - input_df['newbalanceDest']

            # Predict
            X_preped = assets['preprocessor'].transform(input_df)
            anomaly_score = assets['iso_forest'].decision_function(X_preped)
            X_hybrid = np.column_stack((X_preped, anomaly_score))
            xgb_proba = assets['xgb_model'].predict_proba(X_hybrid)[:, 1][0]
            
            status.update(label="Analisis Selesai!", state="complete", expanded=False)

       # --- HASIL VISUAL ---
        if xgb_proba > 0.8:
            status_title = "🚨 BAHAYA: Transaksi Fraud Terdeteksi"
            status_desc = "Sistem mendeteksi indikator penipuan yang sangat kuat."
            css_class = "fraud-bg"
            text_color = "#c53030"
            recommendation = "⛔ REKOMENDASI: BLOKIR TRANSAKSI OTOMATIS"
        elif xgb_proba > 0.5:
            status_title = "⚠️ PERINGATAN: Transaksi Mencurigakan"
            status_desc = "Probabilitas fraud cukup tinggi, namun perlu verifikasi manual."
            css_class = "fraud-bg" # Bisa buat class baru 'warning-bg' jika mau warna oranye
            text_color = "#d97706" # Warna Amber/Oranye
            recommendation = "✋ REKOMENDASI: TAHAN & LAKUKAN PENINJAUAN MANUAL (MANUAL REVIEW)"
        else:
            status_title = "✅ AMAN: Transaksi Valid"
            status_desc = "Pola transaksi terlihat normal dan sesuai profil nasabah."
            css_class = "safe-bg"
            text_color = "#2f855a"
            recommendation = "👍 REKOMENDASI: IZINKAN TRANSAKSI"

        # --- 1. KARTU STATUS UTAMA ---
        st.markdown(f"""
            <div class="status-card {css_class}">
                <h2 style='color:{text_color}; margin:0;'>{status_title}</h2>
                <p style='color:{text_color}; font-size:16px; margin-top:5px;'>{status_desc}</p>
                <hr style='border-top: 1px solid {text_color}; opacity: 0.3;'>
                <p style='color:{text_color}; font-weight:bold; font-size:14px;'>{recommendation}</p>
            </div>
        """, unsafe_allow_html=True)

        # --- 2. DETAIL ANALISIS (GRID 2 KOLOM) ---
        res_col1, res_col2 = st.columns(2)
        
        # KOLOM KIRI: Skor Model
        with res_col1:
            st.markdown("### 📊 Skor Risiko Model")
            
            # Metric Probabilitas XGBoost
            st.metric(
                label="Probabilitas Fraud (XGBoost)", 
                value=f"{xgb_proba:.1%}",
                delta="Sangat Berisiko" if xgb_proba > 0.8 else ("Perlu Waspada" if xgb_proba > 0.5 else "Aman"),
                delta_color="inverse"
            )
            st.progress(float(xgb_proba))
            
            # Metric Anomaly Isolation Forest
            anom_val = anomaly_score[0]
            st.metric(
                label="Skor Anomali (Isolation Forest)", 
                value=f"{anom_val:.4f}",
                help="Semakin negatif skornya, semakin aneh/langka data transaksi ini dibandingkan data historis.",
                delta="Anomaly Terdeteksi" if anom_val < 0 else "Pola Wajar",
                delta_color="off" if anom_val >= 0 else "inverse" 
            )

        # KOLOM KANAN: Penjelasan Teknis (Explainability)
        with res_col2:
            st.markdown("### 💡 Temuan Teknis (Why?)")
            
            # Cek Error Balance (Selisih Saldo) - Ini fitur paling krusial di PaySim
            err_orig = input_df['errorBalanceOrig'].values[0]
            err_dest = input_df['errorBalanceDest'].values[0]
            
            with st.container(border=True):
                # 1. Analisis Saldo Pengirim
                if abs(err_orig) > 0.01:
                    st.error(f"❌ **Ketidakcocokan Saldo Pengirim:** Terdapat selisih **${err_orig:,.2f}** antara mutasi saldo dan jumlah transfer. Ini indikator kuat fraud/pengambilalihan akun.")
                else:
                    st.success("✅ **Validasi Saldo Pengirim:** Perhitungan saldo awal, jumlah transfer, dan saldo akhir konsisten.")
                
                # 2. Analisis Saldo Penerima
                if new_dest == 0 and amount > 0:
                     st.warning("⚠️ **Penerima Mencurigakan:** Uang dikirim tapi saldo akhir penerima tetap 0 (kemungkinan *Cash Out* langsung).")
                elif old_dest == 0 and new_dest > 0:
                    st.info("ℹ️ **Akun Baru/Kosong:** Penerima memiliki saldo awal 0.")
            
            # 3. Analisis Jam Transaksi
            if hour_val < 6:
                st.caption(f"🕒 **Waktu:** Transaksi dilakukan dini hari (Jam {hour_val}:00). Pola ini sering diasosiasikan dengan aktivitas bot.")
            error_val = input_df['errorBalanceOrig'].values[0]
            if abs(error_val) > 0.1:
                st.error(f"Ditemukan ketidaksesuaian saldo: ${error_val:,.2f}")
            else:
                st.success("Logika saldo pengirim konsisten.")

# ------------------------------------------
# HALAMAN 2: PENJELASAN MODEL
# ------------------------------------------
elif selected_page == "Penjelasan Model":
    st.title("📚 Arsitektur Sistem JAGA ")
    
    st.markdown("""
        <div style="background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%); padding: 30px; border-radius: 20px; color: white; margin-bottom: 30px;">
            <h3 style="margin:0;">Mengapa Menggunakan Pendekatan Hybrid?</h3>
            <p style="opacity: 0.9; font-size: 16px; margin-top: 10px;">
                Penipuan transaksi (fraud) seringkali memiliki pola yang sangat cerdik dan terus berubah. JAGA menggabungkan 
                <b>Unsupervised Learning</b> untuk mendeteksi anomali baru (Zero-day fraud) dan <b>Supervised Learning</b> 
                untuk akurasi tinggi pada pola yang sudah dikenal.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("🔄 Alur Kerja Data (Model Pipeline)")
    
    
    st.markdown("""
    <div style="background-color: #f1f5f9; padding: 20px; border-radius: 15px; border: 1px solid #e2e8f0;">
        <p style="color: #475569; font-size: 14px;">
            <b>Input Transaksi</b> → <b>Preprocessing</b> (Scaling & Encoding) → <b>Layer 1: Isolation Forest</b> (Skoring Anomali) → 
            <b>Layer 2: XGBoost</b> (Klasifikasi Final) → <b>Hasil Prediksi</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
            <div style="border: 1px solid #e2e8f0; padding: 20px; border-radius: 15px; height: 100%;">
                <img src="https://img.icons8.com/fluency/64/search-property.png"/>
                <h4 style="margin-top: 15px;">Layer 1: Isolation Forest</h4>
                <p style="font-size: 14px; color: #64748b;"><b>Spesialisasi: Deteksi Kejanggalan</b></p>
                <p style="font-size: 14px; line-height: 1.6;">
                    Model ini bekerja tanpa label. Ia mengasumsikan bahwa transaksi fraud adalah <b>langka</b> dan <b>berbeda</b> secara statistik.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown("""
            <div style="border: 1px solid #e2e8f0; padding: 20px; border-radius: 15px; height: 100%;">
                <img src="https://img.icons8.com/fluency/64/flash-on.png"/>
                <h4 style="margin-top: 15px;">Layer 2: XGBoost Classifier</h4>
                <p style="font-size: 14px; color: #64748b;"><b>Spesialisasi: Akurasi & Presisi</b></p>
                <p style="font-size: 14px; line-height: 1.6;">
                    Algoritma Gradient Boosting yang sangat kuat. Ia menerima input fitur transaksi beserta <b>skor dari Layer 1</b> untuk klasifikasi akhir yang tajam.
                </p>
            </div>
        """, unsafe_allow_html=True)

# ------------------------------------------
# HALAMAN 3: TENTANG DATASET
# ------------------------------------------
elif selected_page == "Tentang Dataset":
    st.title("📂 Dataset & Transparansi")
    
    # 1. HEADER & SUMBER DATA
    st.markdown("""
        <div style="background-color: white; padding: 20px; border-radius: 15px; border-left: 5px solid #3b82f6; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <h4 style="margin-top:0;">📌 PaySim: Synthetic Financial Datasets</h4>
            <p style="color: #64748b; margin-bottom: 10px;">
                Model ini dilatih menggunakan dataset <b>PaySim</b>, sebuah simulasi transaksi uang seluler yang dibuat berdasarkan log transaksi nyata dari layanan keuangan di Afrika. 
                Tujuannya adalah untuk mengisi kekosongan dataset publik terkait penipuan keuangan.
            </p>
            <a href="https://www.kaggle.com/datasets/ealaxi/paysim1" target="_blank" style="text-decoration: none;">
                <button style="background-color: #f1f5f9; color: #334155; border: none; padding: 8px 15px; border-radius: 5px; cursor: pointer; font-size: 12px;">
                    🔗 Lihat Sumber Data (Kaggle)
                </button>
            </a>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")

    # 2. KEY METRICS (STATISTIK DATASET)
    # Angka ini adalah fakta dari dataset PaySim asli (6.3 juta baris)
    st.subheader("📊 Statistik Dataset Asli")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("Total Transaksi", "6,362,620", help="Jumlah baris total dalam dataset.")
    with col_m2:
        st.metric("Total Fraud", "8,213", help="Jumlah transaksi yang dilabeli sebagai penipuan.")
    with col_m3:
        st.metric("Rasio Fraud", "0.13%", help="Sangat tidak seimbang (Imbalanced Data).")
    with col_m4:
        st.metric("Durasi Simulasi", "30 Hari", help="744 steps (jam).")

    st.divider()

    # 3. VISUALISASI DISTRIBUSI (Hardcoded Data for Performance)
    # Kita tidak perlu load CSV 500MB, cukup visualisasikan hasil EDA-nya.
    c1, c2 = st.columns([1.5, 1])
    
    with c1:
        st.subheader("🔍 Dimana Fraud Bersembunyi?")
        st.caption("Distribusi Tipe Transaksi vs Kejadian Fraud")
        
        # Data distribusi PaySim (Approximation)
        fraud_dist_data = pd.DataFrame({
            "Tipe Transaksi": ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"],
            "Jumlah Transaksi": [2237500, 532909, 2151495, 1399284, 41432],
            "Jumlah Fraud": [4116, 4097, 0, 0, 0] # Fraud HANYA ada di Cash_out & Transfer
        })
        
        # Menggunakan Bar Chart Streamlit
        st.bar_chart(
            fraud_dist_data.set_index("Tipe Transaksi")[["Jumlah Fraud"]],
            color="#ef4444" # Warna merah untuk fraud
        )
        st.info("💡 **Insight:** Fraud hanya ditemukan pada tipe transaksi **TRANSFER** dan **CASH_OUT**.")

    with c2:
        st.subheader("📖 Kamus Data (Feature Dictionary)")
        # Tabel Data yang lebih lengkap
        df_desc = pd.DataFrame([
            {"Fitur": "step", "Deskripsi": "Unit waktu (1 step = 1 jam). Total 744 steps (30 hari)."},
            {"Fitur": "type", "Deskripsi": "Jenis transaksi (CASH-IN, OUT, DEBIT, PAYMENT, TRANSFER)."},
            {"Fitur": "amount", "Deskripsi": "Jumlah uang yang ditransaksikan dalam mata uang lokal."},
            {"Fitur": "oldbalanceOrg", "Deskripsi": "Saldo pengirim sebelum transaksi dimulai."},
            {"Fitur": "newbalanceOrig", "Deskripsi": "Saldo pengirim setelah transaksi selesai."},
            {"Fitur": "oldbalanceDest", "Deskripsi": "Saldo penerima sebelum transaksi."},
            {"Fitur": "isFraud", "Deskripsi": "Target variable (1 = Fraud, 0 = Aman)."}
        ])
        st.dataframe(
            df_desc, 
            hide_index=True, 
            use_container_width=True,
            column_config={
                "Fitur": st.column_config.TextColumn("Nama Fitur", width="medium"),
                "Deskripsi": st.column_config.TextColumn("Penjelasan", width="large"),
            }
        )

    # 4. PENJELASAN TANTANGAN
    st.write("")
    with st.expander("🧐 Mengapa Dataset ini Menantang?", expanded=False):
        st.markdown("""
        1.  **Imbalanced Class (Ketimpangan Data):**
            Hanya **0.13%** data yang merupakan Fraud. Jika model menebak semua transaksi "Aman", akurasinya tetap 99.87%, tapi gagal mendeteksi penjahat. Oleh karena itu, kami menggunakan metrik **F1-Score** dan **Recall**, bukan hanya Akurasi.
        
        2.  **Pola Pengurasan Saldo:**
            Banyak fraudster melakukan pemindahan dana dan segera melakukan *Cash Out*. Fitur `newbalanceOrig` seringkali menjadi **0** pada kasus fraud.
        
        3.  **Jumlah Uang:**
            Fraud tidak selalu bernilai besar. Namun, dalam dataset ini, transaksi fraud cenderung mengosongkan rekening korban.
        """)


# ------------------------------------------
# HALAMAN 4: ABOUT ME (Profile Page)
# ------------------------------------------
elif selected_page == "About Me":
    # Header Section dengan Gradient Background
    st.markdown("""
        <div style='background: linear-gradient(120deg, #1e3a8a, #3b82f6); padding: 30px; border-radius: 15px; color: white; text-align: center; margin-bottom: 30px;'>
            <h1 style='margin:0; font-size: 32px;'>👨‍💻 Creator Profile</h1>
            <p style='margin-top:5px; opacity:0.9;'>Meet the mind behind JAGA System</p>
        </div>
    """, unsafe_allow_html=True)

    # Layout Kolom: Foto di Kiri, Bio di Kanan
    col_profile, col_desc = st.columns([1, 2.5], gap="large")
    
    with col_profile:
        # Menampilkan foto profil menggunakan variabel img_profile_base64
        # Pastikan ada huruf 'f' sebelum tanda kutip triple agar variabel terbaca
        st.markdown(f"""
            <style>
            .profile-container {{
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .profile-img {{
                width: 220px;
                height: 220px;
                object-fit: cover; /* Memastikan wajah tetap di tengah dan tidak gepeng */
                border-radius: 50%;
                border: 5px solid #ffffff;
                box-shadow: 0 10px 20px rgba(0,0,0,0.15);
                transition: transform 0.3s ease;
            }}
            .profile-img:hover {{
                transform: scale(1.05);
            }}
            </style>
            <div class="profile-container">
                <img src="data:image/jpeg;base64,{img_profile_base64}" class="profile-img">
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        

    with col_desc:
        # Kartu Bio Utama
        st.markdown("""
            <div style="background-color: white; padding: 30px; border-radius: 15px; border: 1px solid #f1f5f9; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
                <h2 style="color: #1e293b; margin-bottom: 5px; font-weight: 800;">Alvat Taupik Hidayat</h2>
                <p style="color: #3b82f6; font-weight: 600; font-size: 16px; margin-top:0;">🚀 System Analyst, Data Scientist & AI Researcher</p>
                <hr style="border: 0; border-top: 1px solid #e2e8f0; margin: 15px 0;">
                
            
            <div data-testid="stMarkdownContainer" class="st-emotion-cache-2fgyt4 e1t8ru6f0"><div style="background-color: rgb(241, 245, 249); padding: 20px; border-radius: 15px; border: 1px solid rgb(226, 232, 240);">
                <p style="color: rgb(71, 85, 105); font-size: 14px;">
                  Halo! Saya adalah Sistem Analis di balik Arsitektur JAGA.
                  Fokus utama saya adalah menerjemahkan kebutuhan bisnis dan risiko operasional menjadi rancangan sistem teknologi yang terstruktur,
                  terukur, dan berkelanjutan. 
                  Dengan pendekatan analitis dan berbasis data, saya merancang solusi yang mampu mengolah data mentah menjadi sistem cerdas yang dapat diandalkan 
                  dalam pengambilan keputusan—khususnya pada ranah Financial Fraud Detection.
                </p>
            </div></div>



                   
        
            </div>
        """, unsafe_allow_html=True)

        # Tech Stack Section (Tampilan Badges Keren)
        st.markdown("### 🛠️ Tech Stack & Tools")
        st.markdown("""
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                <span style="background-color: #e0f2fe; color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px;">🐍 Python</span>
                <span style="background-color: #f0fdf4; color: #15803d; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px;">🤖 Scikit-Learn</span>
                <span style="background-color: #fff7ed; color: #c2410c; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px;">⚡ XGBoost</span>
                <span style="background-color: #fef2f2; color: #b91c1c; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px;">🎈 Streamlit</span>
                <span style="background-color: #faf5ff; color: #7e22ce; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px;">🐼 Pandas</span>
                <span style="background-color: #f8fafc; color: #475569; padding: 6px 14px; border-radius: 20px; font-weight: 600; font-size: 14px;">📊 Matplotlib</span>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    
    # Contact Section dengan Kartu Hover
    st.subheader("📬 Let's Connect")
    st.markdown("Tertarik diskusi tentang Data Science atau kolaborasi? Hubungi saya di:")
    
    # Custom CSS untuk kartu kontak
    st.markdown("""
    <style>
    .contact-card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
        text-decoration: none;
        color: inherit;
        display: block;
    }
    .contact-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #3b82f6;
    }
    .icon {
        font-size: 24px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Grid Kontak
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("""
            <a href="https://www.linkedin.com/in/alvattaupik/" target="_blank" style="text-decoration:none;">
                <div class="contact-card">
                    <div class="icon" style="color:#0077b5;">💼</div>
                    <div style="font-weight:bold; color:#1e293b;">LinkedIn</div>
                    <div style="font-size:12px; color:#64748b;">Connect Professionally</div>
                </div>
            </a>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
            <a href="https://github.com/alvattaupik" target="_blank" style="text-decoration:none;">
                <div class="contact-card">
                    <div class="icon" style="color:#333;">💻</div>
                    <div style="font-weight:bold; color:#1e293b;">GitHub</div>
                    <div style="font-size:12px; color:#64748b;">Check My Code</div>
                </div>
            </a>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown("""
            <a href="mailto:alvattaufik@gmail.com" target="_blank" style="text-decoration:none;">
                <div class="contact-card">
                    <div class="icon" style="color:#ea4335;">📧</div>
                    <div style="font-weight:bold; color:#1e293b;">Email</div>
                    <div style="font-size:12px; color:#64748b;">alvattaufik@gmail.com</div>
                </div>
            </a>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown(
        "<div style='text-align: center; color: #94a3b8; font-size: 14px;'>"
        "© 2026 JAGA System. Built with ❤️ by Alvat Taupik Hidayat."
        "</div>", 
        unsafe_allow_html=True
    )