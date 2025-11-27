import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import warnings
import re

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="⚽ Dashboard Tattica V30", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# --- TITOLO ---
st.title("💎 Dashboard Analisi Calcio V30")
st.markdown("""
**Strumento di Analisi Tattica e Statistica**
*Statistiche Gol • Previsioni Poisson HT/FT • Ritmi di Gioco (Kaplan-Meier)*
""")
st.divider()

# ==========================================
# 1. CARICAMENTO DATI
# ==========================================
with st.sidebar:
    st.header("📂 Dati & Configurazione")
    uploaded_file = st.file_uploader("Carica il file (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file is None:
    st.info("👈 Carica un file dal menu laterale per iniziare.")
    st.stop()

@st.cache_data
def load_data(file):
    try:
        # Tenta lettura robusta
        try:
            # Legge prima riga per capire separatore
            line = file.readline().decode('latin1')
            file.seek(0) 
            sep = ';' if line.count(';') > line.count(',') else ','
            
            # Carica
            df = pd.read_csv(file, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False)
        except:
            df = pd.read_excel(file)

        # --- PULIZIA CRITICA COLONNE ---
        # 1. Converte in stringa, rimuove spazi, tutto maiuscolo
        df.columns = df.columns.astype(str).str.strip().str.upper()
        
        # 2. RIMUOVE DUPLICATI (Mantiene la prima occorrenza)
        df = df.loc[:, ~df.columns.duplicated()]

        # 3. Mappatura Nomi Standard
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY'],
            'CASA': ['CASA', 'HOME', 'TEAM1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TEAM2']
        }
        
        for target, candidates in col_map.items():
            # Se la colonna target non c'è già
            if target not in df.columns:
                for cand in candidates:
                    if cand in df.columns:
                        df.rename(columns={cand: target}, inplace=True)
                        break
        
        # 4. Pulizia Celle (Trim spazi)
        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # 5. Crea ID Lega
        if 'PAESE' in df.columns and 'LEGA' in df.columns:
            df['ID_LEGA'] = df['PAESE'] + " - " + df['LEGA']
        elif 'LEGA' in df.columns:
            df['ID_LEGA'] = df['LEGA']
            
        return df

    except Exception as e:
        st.error(f"Errore caricamento: {e}")
        return pd.DataFrame()

df = load_data(uploaded_file)

if df.empty:
    st.error("File vuoto o non valido.")
    st.stop()

st.sidebar.success(f"✅ Caricato: {len(df)} righe")

# ==========================================
# 2. SELEZIONE MATCH (CASCATA)
# ==========================================
col1, col2, col3 = st.columns(3)

with col1:
    if 'ID_LEGA' in df.columns:
        leghe = sorted(df['ID_LEGA'].unique())
        sel_lega = st.selectbox("🏆 Seleziona Campionato", leghe)
    else:
        st.error("Colonna LEGA non trovata.")
        st.stop()

# Filtra dataframe
df_league = df[df['ID_LEGA'] == sel_lega].copy()
teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())

with col2:
    sel_home = st.selectbox("🏠 Squadra Casa", teams, index=0)

with col3:
    idx_away = 1 if len(teams) > 1 else 0
    sel_away = st.selectbox("✈️ Squadra Ospite", teams, index=idx_away)

# ==========================================
# 3. ENGINE DI ANALISI
# ==========================================
if st.button("🚀 AVVIA ANALISI", type="primary"):
    st.divider()
    st.subheader(f"⚔️ Analisi: {sel_home} vs {sel_away}")
    
    intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    
    def get_minutes(val):
        if pd.isna(val): return []
        # Pulisce tutto tranne numeri
        s = str(val).replace(',', ' ').replace(';', ' ').replace('.', ' ').replace('"', '').replace("'", "")
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
        res = []
        for x in nums:
            try:
                n = int(float(x))
                if 0 <= n <= 130: res.append(n)
            except: pass
        return res

    # Colonne
    c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else 'GOALMINCASA'
    c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else 'GOALMINOSPITE'

    # Accumulatori
    goals_h = {'FT': 0, 'HT': 0, '2T': 0, 'S_FT': 0, 'S_HT': 0, 'S_2T': 0}
    goals_a = {'FT': 0, 'HT': 0, '2T': 0, 'S_FT': 0, 'S_HT': 0, 'S_2T': 0}
    match_h, match_a = 0, 0
    times_h, times_a, times_league = [], [], []
    
    # Heatmap specifica per le due squadre
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        # Media Lega (KM)
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))

        # --- POPOLA HEATMAP (Solo per le due squadre) ---
        if h in stats_match:
            for m in min_h: 
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[h]['F'][intervals[idx]] += 1
            for m in min_a: 
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[h]['S'][intervals[idx]] += 1 
        
        if a in stats_match:
            for m in min_a: 
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[a]['F'][intervals[idx]] += 1
            for m in min_h: 
                idx = min(5, (m-1)//15)
                if m > 45 and m <= 60 and idx < 3: idx = 3
                stats_match[a]['S'][intervals
