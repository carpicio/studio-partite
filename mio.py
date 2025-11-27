import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from scipy.stats import poisson
import warnings
import re

# Configurazione Pagina
st.set_page_config(page_title="⚽ Dashboard Tattica V29", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Titolo
st.title("💎 Dashboard Analisi Calcio V29")
st.markdown("**Fix Errori: Gestione Squadre con 0 Gol e Caricamento Robusto**")
st.divider()

# ==========================================
# 1. CARICAMENTO DATI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("📂 Configurazione")
    uploaded_file = st.file_uploader("Carica il file (CSV/Excel)", type=['csv', 'xlsx'])

if uploaded_file is None:
    st.info("👈 Per iniziare, carica il file dei dati dalla barra laterale.")
    st.stop()

@st.cache_data
def load_dataset(file):
    try:
        # Tenta lettura CSV con separatore automatico
        try:
            line = file.readline().decode('latin1')
            file.seek(0) 
            sep = ';' if line.count(';') > line.count(',') else ','
            
            df = pd.read_csv(file, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False, header=None)
        except:
            df = pd.read_excel(file, header=None)

        # Pulizia Header
        header = df.iloc[0].astype(str).str.strip().str.upper().tolist()
        seen = {}
        unique_header = []
        for col in header:
            if col in seen:
                seen[col] += 1
                unique_header.append(f"{col}.{seen[col]}")
            else:
                seen[col] = 0
                unique_header.append(col)
                
        df = df.iloc[1:].copy()
        df.columns = unique_header
        
        # Mappatura Colonne
        col_map = {
            'GOALMINH': ['GOALMINH', 'GOALMINCASA', 'MINUTI_CASA'],
            'GOALMINA': ['GOALMINA', 'GOALMINOSPITE', 'MINUTI_OSPITE'],
            'LEGA': ['LEGA', 'LEAGUE', 'DIVISION'],
            'PAESE': ['PAESE', 'COUNTRY'],
            'CASA': ['CASA', 'HOME', 'TEAM1'],
            'OSPITE': ['OSPITE', 'AWAY', 'TEAM2']
        }
        
        for target, candidates in col_map.items():
            if target not in df.columns:
                for candidate in candidates:
                    found = next((c for c in df.columns if c == candidate), None)
                    if found:
                        df.rename(columns={found: target}, inplace=True)
                        break
        
        # Pulizia dati stringa
        for c in ['PAESE', 'LEGA', 'CASA', 'OSPITE']:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()

        # Crea ID Lega
        if 'PAESE' in df.columns:
            df['ID_LEGA'] = df['PAESE'] + " - " + df['LEGA']
        else:
            df['ID_LEGA'] = df['LEGA']
            
        return df

    except Exception as e:
        st.error(f"Errore critico nel caricamento: {e}")
        return pd.DataFrame()

df = load_dataset(uploaded_file)

if df.empty:
    st.error("Il file caricato è vuoto o non valido.")
    st.stop()

st.sidebar.success(f"✅ File caricato: {len(df)} righe.")

# ==========================================
# 2. SELEZIONE MATCH
# ==========================================
col_sel1, col_sel2, col_sel3 = st.columns(3)

with col_sel1:
    leghe = sorted(df['ID_LEGA'].unique())
    sel_lega = st.selectbox("🏆 Seleziona Campionato", leghe)

# Filtra dataframe per lega
df_league = df[df['ID_LEGA'] == sel_lega].copy()
teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())

with col_sel2:
    sel_home = st.selectbox("🏠 Squadra Casa", teams, index=0)

with col_sel3:
    idx_away = 1 if len(teams) > 1 else 0
    sel_away = st.selectbox("✈️ Squadra Ospite", teams, index=idx_away)

# ==========================================
# 3. ENGINE DI ANALISI
# ==========================================
if st.button("🚀 AVVIA ANALISI MATCH", type="primary"):
    st.divider()
    st.subheader(f"⚔️ Analisi Tattica: {sel_home} vs {sel_away}")
    
    intervals = ['0-15', '16-30', '31-45', '46-60', '61-75', '76-90']
    
    def get_minutes(val):
        if pd.isna(val): return []
        s = str(val).replace('"', '').replace("'", "").replace('.', ' ').replace(',', ' ').replace(';', ' ')
        res = []
        for x in s.split():
            if x.isdigit():
                n = int(x)
                if 0 <= n <= 130: res.append(n)
        return res

    c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else 'GOALMINCASA'
    c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else 'GOALMINOSPITE'

    # Accumulatori
    goals_h = {'FT': 0, 'HT': 0, '2T': 0, 'S_FT': 0, 'S_HT': 0, 'S_2T': 0}
    goals_a = {'FT': 0, 'HT': 0, '2T': 0, 'S_FT': 0, 'S_HT': 0, 'S_2T': 0}
    match_h, match_a = 0, 0
    times_h, times_a, times_league = [], [], []
    
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        # Dati Lega (per Media KM) - FIX ERROR QUI
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))

        # --- POPOLA HEATMAP ---
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
                stats_match[a]['S'][intervals[idx]] += 1 

        # --- DATI STATISTICI ---
        if h == sel_home:
            match_h += 1
            goals_h['FT'] += len(min_h)
            goals_h['HT'] += len([x for x in min_h if x <= 45])
            goals_h['2T'] += len([x for x in min_h if x > 45])
            
            goals_h['S_FT'] += len(min_a)
            goals_h['S_HT'] += len([x for x in min_a if x <= 45])
            goals_h['S_2T'] += len([x for x in min_a if x > 45])
            
            if min_h: times_h.append(min(min_h)) # FIX ERROR
        
        if a == sel_away:
            match_a += 1
            goals_a['FT'] += len(min_a)
            goals_a['HT'] += len([x for x in min_a if x <= 45])
            goals_a['2T'] += len([x for x in min_a if x > 45])
            
            goals_a['S_FT'] += len(min_h)
            goals_a['S_HT'] += len([x for x in min_h if x <= 45])
            goals_a['S_2T'] += len([x for x in min_h if x > 45])
            
            if min_a: times_a.append(min(min_a)) # FIX ERROR

    # --- CALCOLI MEDIE ---
    def safe_div(n, d): return n / d if d > 0 else 0

    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_s_ft = safe_div(goals_h['S_FT'], match_h)
    avg_h_s_ht = safe_div(goals_h['S_HT'], match_h)

    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_s_ft = safe_div(goals_a['S_FT'], match_a)
    avg_a_s_ht = safe_div(goals_a['S_HT'], match_a)

    # --- VISUALIZZAZIONE MEDIE ---
    st.write("### 📊 Medie Gol")
    c1, c2 = st.columns(2)
    with c1:
        st.info(f"**🏠 {sel_home}** ({match_h} match)\n\n"
                f"**1°T:** {avg_h_ht:.2f} F | {avg_h_s_ht:.2f} S\n\n"
                f"**FIN:** {avg_h_ft:.2f} F | {avg_h_s_ft:.2f} S")
    with c2:
        st.error(f"**✈️ {sel_away}** ({match_a} match)\n\n"
                 f"**1°T:** {avg_a_ht:.2f} F | {avg_a_s_ht:.2f} S\n\n"
                 f"**FIN:** {avg_a_ft:.2f} F | {avg_a_s_ft:.2f} S")

    st.divider()

    # --- POISSON HT/FT ---
    exp_h_ft = (avg_h_ft + avg_a_s_ft) / 2
    exp_a_ft = (avg_a_ft + avg_h_s_ft) / 2
    exp_h_ht = (avg_h_ht + avg_a_s_ht) / 2
    exp_a_ht = (avg_a_ht + avg_h_s_ht) / 2

    def calc_poisson_probs(lam_h, lam_a):
        probs = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                probs[i][j] = poisson.pmf(i, lam_h) * poisson.pmf(j, lam_a)
        p1 = np.sum(np.tril(probs, -1))
        px = np.sum(np.diag(probs))
        p2 = np.sum(np.triu(probs, 1))
        
        pu25 = 0
        for i in range(6):
            for j in range(6):
                if i+j <= 2: pu25 += probs[i][j]
        return p1, px, p2, pu25

    p1_ft, px_ft, p2_ft, pu25_ft = calc_poisson_probs(exp_h_ft, exp_a_ft)
    p1_ht, px_ht, p2_ht, _ = calc_poisson_probs(exp_h_ht, exp_a_ht) # 1X2 HT

    prob_00_ht = poisson.pmf(0, exp_h_ht) * poisson.pmf(0, exp_a_ht)
    prob_u15_ht = prob_00_ht + (poisson.pmf(1, exp_h_ht) * poisson.pmf(0, exp_a_ht)) + (poisson.pmf(0, exp_h_ht) * poisson.pmf(1, exp_a_ht))

    def to_odd(p): return round(1/p, 2) if p > 0 else 99.00

    st.subheader("🎲 Previsioni & Quote Implicite")
    c1, c2, c3 = st.columns(3)
    c1.metric("1X2 Finale", f"1: {p1_ft*100:.0f}%", f"Quota: {to_odd(p1_ft)}")
    c1.caption(f"X: {px_ft*100:.0f}% (@{to_odd(px_ft)}) | 2: {p2_ft*100:.0f}% (@{to_odd(p2_ft)})")
    
    c2.metric("O/U 2.5 FT", f"Over: {(1-pu25_ft)*100:.0f}%", f"@{to_odd(1-pu25_ft)}")
    c2.caption(f"Under: {pu25_ft*100:.0f}% (@{to_odd(pu25_ft)})")
    
    c3.metric("Speciale 1°T", f"0-0: {prob_00_ht*100:.0f}%", f"@{to_odd(prob_00_ht)}")
    c3.caption(f"Under 1.5 HT: {prob_u15_ht*100:.0f}% (@{to_odd(prob_u15_ht)})")

    st.divider()

    # --- GRAFICI ---
    tab1, tab2, tab3 = st.tabs(["📉 Ritmo Gol (Kaplan-Meier)", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti"])

    with tab1:
        if times_h or times_a: # Basta che UNA delle due abbia dati
            fig, ax = plt.subplots(figsize=(10, 5))
            kmf_h = KaplanMeierFitter()
            kmf_a = KaplanMeierFitter()
            kmf_l = KaplanMeierFitter()
            
            if times_h:
                kmf_h.fit(times_h, label=f'{sel_home} (1° Gol)')
                kmf_h.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            
            if times_a:
                kmf_a.fit(times_a, label=f'{sel_away} (1° Gol)')
                kmf_a.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
                
            if times_league:
                kmf_l.fit(times_league, label='Media Campionato')
                kmf_l.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')

            # Mediana (solo se ci sono dati)
            try:
                if times_h:
                    med_h = kmf_h.median_survival_time_
                    plt.text(5, 0.1, f"1° Gol {sel_home}: ~{med_h:.0f}'", color='blue', fontweight='bold')
                if times_a:
                    med_a = kmf_a.median_survival_time_
                    plt.text(5, 0.05, f"1° Gol {sel_away}: ~{med_a:.0f}'", color='red', fontweight='bold')
            except: pass

            plt.axhline(y=0.5, color='green', linestyle=':', label='Mediana (50%)')
            plt.title(f'Tempo al 1° Gol (Probabilità 0-0)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("Dati insufficienti per il grafico KM (Nessun gol segnato da entrambe le squadre).")

    # Heatmaps
    rows_f = []
    rows_s = []
    for t in [sel_home, sel_away]:
        d = stats_match[t]
        rows_f.append({**{'SQUADRA': t}, **d['F']})
        rows_s.append({**{'SQUADRA': t}, **d['S']})
    
    df_f = pd.DataFrame(rows_f).set_index('SQUADRA')
    df_s = pd.DataFrame(rows_s).set_index('SQUADRA')

    with tab2:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df_f[intervals], annot=True, cmap="Greens", fmt="d", cbar=False, ax=ax)
        plt.title("Densità Gol Fatti")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=ax)
        plt.title("Densità Gol Subiti")
        st.pyplot(fig)
