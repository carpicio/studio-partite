import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from scipy.stats import poisson
import warnings
import re
import os

# Configurazione Pagina
st.set_page_config(page_title="⚽ Football Analytics Pro", layout="wide", page_icon="⚽")
warnings.simplefilter(action='ignore', category=FutureWarning)

# Titolo e Intestazione
st.title("💎 Dashboard Analisi Calcio Pro")
st.markdown("""
**Strumento Professionale di Analisi Statistica e Predittiva**
*Statistiche 1°T/2°T/Finale • Modelli Poisson HT/FT • Ritmi di Gioco Kaplan-Meier*
""")
st.divider()

# ==========================================
# 1. CARICAMENTO DATI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.header("📂 Configurazione Dati")
    
    # Opzione 1: Carica dal PC
    uploaded_file = st.file_uploader("Carica il file (CSV o Excel)", type=['csv', 'xlsx'])
    
    # Opzione 2: Usa file di default (se presente nella repo)
    default_file = 'eng_tot_1.csv'
    use_default = False
    
    # Verifica esistenza file default
    if uploaded_file is None:
        if os.path.exists(default_file):
            st.success(f"File predefinito trovato: {default_file}")
            use_default = True
        else:
            st.warning(f"File predefinito '{default_file}' non trovato. Carica un file per iniziare.")

if uploaded_file is None and not use_default:
    st.stop()

@st.cache_data
def load_data(file_path_or_buffer, is_path=False):
    try:
        # Tenta lettura CSV con separatore automatico
        try:
            if is_path:
                with open(file_path_or_buffer, 'r', encoding='latin1', errors='replace') as f:
                    line = f.readline()
                    sep = ';' if line.count(';') > line.count(',') else ','
                file_source = file_path_or_buffer
            else:
                # Legge prima riga dal buffer
                line = file_path_or_buffer.readline().decode('latin1')
                file_path_or_buffer.seek(0) 
                sep = ';' if line.count(';') > line.count(',') else ','
                file_source = file_path_or_buffer
            
            df = pd.read_csv(file_source, sep=sep, encoding='latin1', on_bad_lines='skip', low_memory=False, header=None)
        except:
            # Fallback Excel
            df = pd.read_excel(file_path_or_buffer, header=None)

        # Pulizia Header
        # Prende la prima riga come header
        header = df.iloc[0].astype(str).str.strip().str.upper().tolist()
        
        # Rendi unici i nomi delle colonne
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
        
        # Mappatura Colonne (Case Insensitive)
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
                    # Cerca se esiste una colonna che contiene il candidato
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

# Caricamento effettivo
if uploaded_file:
    df = load_data(uploaded_file)
elif use_default:
    df = load_data(default_file, is_path=True)

if df.empty:
    st.error("Il file caricato è vuoto o non valido.")
    st.stop()

st.sidebar.success(f"✅ Dati caricati: {len(df)} righe")

# ==========================================
# 2. SELEZIONE MATCH (CASCATA)
# ==========================================
col_sel1, col_sel2, col_sel3 = st.columns(3)

# 1. Seleziona Lega
leghe = sorted(df['ID_LEGA'].unique())
with col_sel1:
    sel_lega = st.selectbox("🏆 Seleziona Campionato", leghe)

# Filtra dataframe per lega
df_league = df[df['ID_LEGA'] == sel_lega].copy()

# 2. Seleziona Squadre (Casa)
teams = sorted(pd.concat([df_league['CASA'], df_league['OSPITE']]).unique())
with col_sel2:
    sel_home = st.selectbox("🏠 Squadra Casa", teams, index=0)

# 3. Seleziona Squadre (Ospite)
with col_sel3:
    # Default diverso da casa
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
        # Pulisce tutto tranne i numeri
        s = str(val).replace(',', ' ').replace(';', ' ').replace('.', ' ').replace('"', '').replace("'", "")
        res = []
        for x in s.split():
            if x.isdigit():
                n = int(x)
                if 0 <= n <= 130: res.append(n)
        return res

    # Cerca colonne minuti (con fallback posizionale se i nomi non matchano)
    c_h = 'GOALMINH' if 'GOALMINH' in df_league.columns else df_league.columns[21]
    c_a = 'GOALMINA' if 'GOALMINA' in df_league.columns else df_league.columns[22]

    # Accumulatori
    goals_h = {'FT': 0, 'HT': 0, '2T': 0, 'S_FT': 0, 'S_HT': 0, 'S_2T': 0}
    goals_a = {'FT': 0, 'HT': 0, '2T': 0, 'S_FT': 0, 'S_HT': 0, 'S_2T': 0}
    match_h, match_a = 0, 0
    times_h, times_a, times_league = [], [], []
    
    # Stats per Heatmap (specifiche per le due squadre)
    stats_match = {
        sel_home: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}},
        sel_away: {'F': {i:0 for i in intervals}, 'S': {i:0 for i in intervals}}
    }

    for _, row in df_league.iterrows():
        h, a = row['CASA'], row['OSPITE']
        min_h = get_minutes(row.get(c_h))
        min_a = get_minutes(row.get(c_a))
        
        # Dati Lega (per Media KM)
        if min_h: times_league.append(min(min_h))
        if min_a: times_league.append(min(min_a))

        # --- POPOLA HEATMAP (Solo se squadre match) ---
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

        # --- DATI STATISTICI E CONTEGGIO ---
        if h == sel_home:
            match_h += 1
            goals_h['FT'] += len(min_h)
            goals_h['HT'] += len([x for x in min_h if x <= 45])
            goals_h['2T'] += len([x for x in min_h if x > 45])
            
            goals_h['S_FT'] += len(min_a)
            goals_h['S_HT'] += len([x for x in min_a if x <= 45])
            goals_h['S_2T'] += len([x for x in min_a if x > 45])
            
            if min_h: times_h.append(min(min_h))
        
        if a == sel_away:
            match_a += 1
            goals_a['FT'] += len(min_a)
            goals_a['HT'] += len([x for x in min_a if x <= 45])
            goals_a['2T'] += len([x for x in min_a if x > 45])
            
            goals_a['S_FT'] += len(min_h)
            goals_a['S_HT'] += len([x for x in min_h if x <= 45])
            goals_a['S_2T'] += len([x for x in min_h if x > 45])
            
            if min_a: times_a.append(min(min_a))

    # --- CALCOLI MEDIE ---
    def safe_div(n, d): return n / d if d > 0 else 0

    # Casa
    avg_h_ft = safe_div(goals_h['FT'], match_h)
    avg_h_ht = safe_div(goals_h['HT'], match_h)
    avg_h_2t = safe_div(goals_h['2T'], match_h)
    avg_h_s_ft = safe_div(goals_h['S_FT'], match_h)
    avg_h_s_ht = safe_div(goals_h['S_HT'], match_h)
    avg_h_s_2t = safe_div(goals_h['S_2T'], match_h)

    # Ospite
    avg_a_ft = safe_div(goals_a['FT'], match_a)
    avg_a_ht = safe_div(goals_a['HT'], match_a)
    avg_a_2t = safe_div(goals_a['2T'], match_a)
    avg_a_s_ft = safe_div(goals_a['S_FT'], match_a)
    avg_a_s_ht = safe_div(goals_a['S_HT'], match_a)
    avg_a_s_2t = safe_div(goals_a['S_2T'], match_a)

    # --- VISUALIZZAZIONE MEDIE ---
    st.write("### 📊 Statistiche Medie")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**🏠 {sel_home}** ({match_h} match)\n\n"
                f"**1°T:** F {avg_h_ht:.2f} | S {avg_h_s_ht:.2f}\n\n"
                f"**FIN:** F {avg_h_ft:.2f} | S {avg_h_s_ft:.2f}")
    with col2:
        st.warning(f"**✈️ {sel_away}** ({match_a} match)\n\n"
                 f"**1°T:** F {avg_a_ht:.2f} | S {avg_a_s_ht:.2f}\n\n"
                 f"**FIN:** F {avg_a_ft:.2f} | S {avg_a_s_ft:.2f}")

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

    # --- GRAFICI AVANZATI ---
    tab1, tab2, tab3 = st.tabs(["📉 Ritmo Gol (Kaplan-Meier)", "⚽ Heatmap Fatti", "🛡️ Heatmap Subiti"])

    with tab1:
        if times_h and times_a:
            fig, ax = plt.subplots(figsize=(10, 5))
            kmf_h = KaplanMeierFitter()
            kmf_a = KaplanMeierFitter()
            kmf_l = KaplanMeierFitter()
            
            kmf_h.fit(times_h, label=f'{sel_home} (1° Gol)')
            kmf_a.fit(times_a, label=f'{sel_away} (1° Gol)')
            
            # Plot Media Lega
            if times_league:
                kmf_l.fit(times_league, label='Media Lega')
                kmf_l.plot_survival_function(ax=ax, ci_show=False, linewidth=2, color='gray', linestyle='--')
            
            # Plot Squadre
            kmf_h.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue')
            kmf_a.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red')
            
            # Mediana e Log Rank
            med_h = kmf_h.median_survival_time_
            med_a = kmf_a.median_survival_time_
            
            plt.axhline(y=0.5, color='green', linestyle=':', label='Mediana (50%)')
            plt.title(f'Tempo al 1° Gol: {sel_home} (~{med_h:.0f}\') vs {sel_away} (~{med_a:.0f}\')')
            plt.grid(True, alpha=0.3)
            plt.xlabel("Minuti")
            plt.ylabel("Probabilità 0-0")
            plt.legend()
            st.pyplot(fig)
            
            # Log Rank Info
            try:
                results = logrank_test(times_h, times_a)
                st.info(f"🧪 **Log-Rank Test:** P-Value = {results.p_value:.4f} " + 
                        ("✅ (Differenza Significativa)" if results.p_value < 0.05 else "❌ (Ritmi Simili)"))
            except: pass
        else:
            st.warning("Dati insufficienti per il grafico del Ritmo.")

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
        plt.title(f'Quando Segnano ({sel_home} vs {sel_away})')
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(10, 3))
        sns.heatmap(df_s[intervals], annot=True, cmap="Reds", fmt="d", cbar=False, ax=ax)
        plt.title(f'Quando Subiscono ({sel_home} vs {sel_away})')
        st.pyplot(fig)
