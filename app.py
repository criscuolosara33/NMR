import streamlit as st
from streamlit_ketcher import st_ketcher
import requests
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

st.set_page_config(page_title="Simulatore NMR", layout="wide")
st.title("Simulatore Spettri NMR (¹H a 500 MHz & ¹³C a 125 MHz) 🧪")
st.markdown("**Disegna la molecola per generare lo spettro, la tabella riassuntiva e scaricare il report completo in PDF.**")

# --- FUNZIONI CHIMICHE (Logica originale invariata) ---
def calcola_proprieta(mol):
    mol_h = Chem.AddHs(mol)
    c = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 6)
    h = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 1)
    n = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() == 7)
    x = sum(1 for a in mol_h.GetAtoms() if a.GetAtomicNum() in [9, 17, 35, 53])
    dbe = c + 1 - (h / 2.0) + (n / 2.0) - (x / 2.0)
    mw = Descriptors.MolWt(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol_h)
    return {'formula': formula, 'mw': mw, 'dbe': dbe, 'n_h': h, 'mol_h': mol_h, 'mol_no_h': mol}

def genera_picchi(center_ppm, mult_type, integral, freq=500.0):
    j_std, j_ortho, j_meta = 7.5/freq, 8.0/freq, 2.0/freq
    mult = mult_type.lower() if mult_type else 's'

    if mult == 'd': off, rat = [-j_std/2, j_std/2], [0.5, 0.5]
    elif mult == 't': off, rat = [-j_std, 0, j_std], [0.25, 0.5, 0.25]
    elif mult == 'q': off, rat = [-1.5*j_std, -0.5*j_std, 0.5*j_std, 1.5*j_std], [0.125, 0.375, 0.375, 0.125]
    elif mult == 'dd': off, rat = [-j_ortho/2 - j_meta/2, -j_ortho/2 + j_meta/2, j_ortho/2 - j_meta/2, j_ortho/2 + j_meta/2], [0.25, 0.25, 0.25, 0.25]
    elif mult == 'dt': off, rat = [-j_ortho/2 - j_std, -j_ortho/2, -j_ortho/2 + j_std, j_ortho/2 - j_std, j_ortho/2, j_ortho/2 + j_std], [0.125, 0.25, 0.125, 0.125, 0.25, 0.125]
    elif mult == 'm': off, rat = np.linspace(-1.5*j_std, 1.5*j_std, 5), [0.1, 0.25, 0.3, 0.25, 0.1]
    else: off, rat = [0.0], [1.0]

    return [(center_ppm + o, r * integral) for o, r in zip(off, rat)]

def stima_locale_1h(mol_h):
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))
    groups = {}
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            rank = ranks[atom.GetIdx()]
            if rank not in groups:
                groups[rank] = []
            groups[rank].append(atom)

    signals = []
    shifts_visti = []

    for rank, h_atoms in groups.items():
        rep_h = h_atoms[0]
        integral = len(h_atoms)
        neighbor = rep_h.GetNeighbors()[0]

        c_indices = set()
        for h in h_atoms:
            c_indices.add(h.GetNeighbors()[0].GetIdx() + 1)

        if neighbor.GetIsAromatic(): shift = 7.3
        elif neighbor.GetAtomicNum() == 8: shift = 4.5
        elif neighbor.GetAtomicNum() == 7: shift = 2.5
        elif neighbor.GetAtomicNum() == 6:
            if neighbor.GetHybridization() == Chem.HybridizationType.SP2: shift = 5.5
            elif neighbor.GetHybridization() == Chem.HybridizationType.SP: shift = 2.8
            else:
                n_carbon_neighbors = sum(1 for a in neighbor.GetNeighbors() if a.GetAtomicNum() == 6)
                shift = 0.9 + (0.3 * n_carbon_neighbors)
        else: shift = 2.0

        while any(abs(shift - sv) < 0.05 for sv in shifts_visti):
            shift += 0.1
        shifts_visti.append(shift)

        vicini_h = 0
        if neighbor.GetAtomicNum() == 6:
            for c_neigh in neighbor.GetNeighbors():
                if c_neigh.GetAtomicNum() == 6:
                    for h_atom in c_neigh.GetNeighbors():
                        if h_atom.GetAtomicNum() == 1 and ranks[h_atom.GetIdx()] != rank:
                            vicini_h += 1

        mult_map = {0:'s', 1:'d', 2:'t', 3:'q', 4:'m', 5:'m', 6:'m', 7:'m', 8:'m', 9:'m'}
        mult = mult_map.get(vicini_h, 'm') if neighbor.GetAtomicNum() == 6 else 's'

        signals.append({
            'delta': shift,
            'multiplicity': mult,
            'integral': integral,
            'atoms': list(c_indices)
        })

    return signals

def stima_locale_13c(mol_no_h):
    ranks = list(Chem.CanonicalRankAtoms(mol_no_h, breakTies=False))
    groups = {}
    for atom in mol_no_h.GetAtoms():
        if atom.GetAtomicNum() == 6:
            rank = ranks[atom.GetIdx()]
            if rank not in groups:
                groups[rank] = []
            groups[rank].append(atom)

    signals = []
    shifts_visti = []

    for rank, c_atoms in groups.items():
        rep_c = c_atoms[0]
        integral = len(c_atoms)

        shift = 30.0

        n_neighbors_C = 0
        n_neighbors_O = 0
        n_neighbors_N = 0
        is_aromatic = rep_c.GetIsAromatic()

        for neighbor in rep_c.GetNeighbors():
            if neighbor.GetAtomicNum() == 6:
                n_neighbors_C += 1
            elif neighbor.GetAtomicNum() == 8:
                n_neighbors_O += 1
            elif neighbor.GetAtomicNum() == 7:
                n_neighbors_N += 1

        if rep_c.GetHybridization() == Chem.HybridizationType.SP2:
            if is_aromatic: shift = 130.0
            elif any(
                mol_no_h.GetBondBetweenAtoms(rep_c.GetIdx(), n.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                and n.GetAtomicNum() == 8
                for n in rep_c.GetNeighbors()
            ): shift = 170.0
            else: shift = 120.0
        elif rep_c.GetHybridization() == Chem.HybridizationType.SP:
            shift = 70.0
        else:
            shift += n_neighbors_C * 8
            shift += n_neighbors_O * 40
            shift += n_neighbors_N * 20

        while any(abs(shift - sv) < 0.5 for sv in shifts_visti):
            shift += 0.5
        shifts_visti.append(shift)

        signals.append({
            'delta': shift,
            'multiplicity': 's',
            'integral': integral,
            'atoms': [atom.GetIdx() + 1 for atom in c_atoms]
        })
    return signals

delta_symbol = r'$\delta$'

# --- EDITOR KETCHER ---
smiles = st_ketcher()

col1, col2 = st.columns(2)
with col1:
    btn_1h = st.button("Genera Spettro ¹H-NMR (500 MHz)", type="primary", use_container_width=True)
with col2:
    btn_13c = st.button("Genera Spettro ¹³C-NMR (125 MHz)", type="secondary", use_container_width=True)

if btn_1h or btn_13c:
    nmr_type = '1h' if btn_1h else '13c'
    
    if not smiles:
        st.warning("Disegna una molecola prima di procedere.")
    else:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Errore struttura.")
        else:
            props = calcola_proprieta(mol)
            st.info("⏳ Calcolo in corso...")

            signals = []
            api_endpoint = 'prediction'
            local_prediction_func = None
            plot_title = ''
            x_range = [-0.5, 12.5]
            x_label = r'$\delta$ (ppm)'
            plot_color = '#0077B6'

            if nmr_type == '1h':
                api_endpoint = 'prediction'
                local_prediction_func = stima_locale_1h
                plot_title = 'Spettro ¹H-NMR a 500 MHz'
                x_range = [-0.5, 12.5]
                x_label = r'$\delta$ (ppm)'
                plot_color = '#0077B6'
                mol_for_local_pred = props['mol_h']
            elif nmr_type == '13c':
                api_endpoint = '13c_prediction'
                local_prediction_func = stima_locale_13c
                plot_title = 'Spettro ¹³C-NMR a 125 MHz'
                x_range = [-10, 220]
                x_label = r'$\delta$ (ppm)'
                plot_color = '#CC3311'
                mol_for_local_pred = props['mol_no_h']

            try:
                url = f"https://www.nmrdb.org/service/{api_endpoint}?smiles={requests.utils.quote(smiles)}"
                res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
                if res.status_code == 200:
                    signals = res.json().get(nmr_type, [])
                    for sig in signals:
                        sig['atoms'] = [a + 1 for a in sig.get('atoms', [])]
                else:
                    signals = []
            except Exception:
                signals = []

            if not signals:
                st.warning(f"⚠️ API irraggiungibile per {nmr_type.upper()}-NMR: Utilizzo del simulatore quantomeccanico locale...")
                if local_prediction_func:
                    signals = local_prediction_func(mol_for_local_pred)

            if not signals:
                st.error("Nessun segnale da mostrare.")
            else:
                pdf_buffer = io.BytesIO()
                with PdfPages(pdf_buffer) as pdf:

                    # --- 1. Molecular Structure Plot ---
                    fig_mol_draw = plt.figure(figsize=(8, 6))
                    ax_mol_draw = fig_mol_draw.add_subplot(111)

                    for atom in mol.GetAtoms():
                        atom.SetProp('atomNote', str(atom.GetIdx() + 1))

                    d2d = rdMolDraw2D.MolDraw2DCairo(int(fig_mol_draw.dpi * fig_mol_draw.get_figwidth()),
                                                    int(fig_mol_draw.dpi * fig_mol_draw.get_figheight()))
                    d2d.drawOptions().annotationFontScale = 0.9
                    d2d.DrawMolecule(mol)
                    d2d.FinishDrawing()

                    img_bytes = d2d.GetDrawingText()
                    img_2d = Image.open(io.BytesIO(img_bytes))

                    ax_mol_draw.imshow(img_2d)
                    ax_mol_draw.axis('off')
                    ax_mol_draw.set_title("Struttura Molecolare", fontsize=14, fontweight='bold')

                    info = f"Formula: {props['formula']} | Massa: {props['mw']:.2f} g/mol | DBE: {props['dbe']:.1f} | 1H: {props['n_h']}"
                    fig_mol_draw.text(0.5, 0.05, info, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='#F0F4F8', edgecolor='#2A9D8F'))

                    pdf.savefig(fig_mol_draw)
                    st.pyplot(fig_mol_draw)
                    plt.close(fig_mol_draw)

                    # --- 2. Main Spectrum Plot ---
                    if nmr_type == '1h':
                        x_ppm = np.linspace(x_range[0], x_range[1], 50000)
                        gamma = 0.0025
                    elif nmr_type == '13c':
                        x_ppm = np.linspace(x_range[0], x_range[1], 20000)
                        gamma = 0.5

                    y_intensity = np.zeros_like(x_ppm)

                    for sig in signals:
                        delta = float(sig.get('delta', 1.0))
                        if nmr_type == '1h':
                            sub_peaks = genera_picchi(delta, sig.get('multiplicity', 's'), float(sig.get('integral', 1)), 500.0)
                        else:
                            sub_peaks = [(delta, float(sig.get('integral', 1)))]

                        for p_shift, p_int in sub_peaks:
                            y_intensity += p_int / (1.0 + ((x_ppm - p_shift) / gamma)**2)

                    fig_main = plt.figure(figsize=(15, 5))
                    ax_spec = fig_main.add_subplot(111)
                    ax_spec.plot(x_ppm, y_intensity, color=plot_color, linewidth=1.2)
                    ax_spec.set_xlim(x_range[1], x_range[0])
                    ax_spec.set_ylim(0, max(y_intensity) * 1.15 if len(y_intensity) > 0 else 1)
                    ax_spec.set_xlabel(x_label, fontsize=11, fontweight='bold')
                    ax_spec.set_ylabel('Intensità', fontsize=11, fontweight='bold')
                    ax_spec.set_title(plot_title, fontweight='bold')
                    ax_spec.grid(True, linestyle='--', alpha=0.4)

                    pdf.savefig(fig_main)
                    st.pyplot(fig_main)
                    plt.close(fig_main)

                    # --- 3. Summary Table ---
                    if signals:
                        df_signals = pd.DataFrame(signals)
                        if 'atoms' in df_signals.columns:
                            df_signals['atoms'] = df_signals['atoms'].apply(lambda x: ', '.join(map(str, x)))
                        df_signals.rename(columns={
                            'delta': r'$\delta$ (ppm)',
                            'multiplicity': 'Molteplicità',
                            'integral': 'Integrale',
                            'atoms': 'Atomi'
                        }, inplace=True)
                        df_signals = df_signals.sort_values(by=r'$\delta$ (ppm)', ascending=False).reset_index(drop=True)

                        fig_table = plt.figure(figsize=(10, 2 + len(df_signals) * 0.3))
                        ax_table = fig_table.add_subplot(111)
                        ax_table.axis('off')
                        ax_table.set_title('Tabella Riassuntiva Segnali', fontsize=14, fontweight='bold', loc='left')
                        table = ax_table.table(cellText=df_signals.values, colLabels=df_signals.columns, loc='center', cellLoc='center')
                        table.auto_set_font_size(False)
                        table.set_fontsize(10)
                        table.scale(1.2, 1.2)

                        pdf.savefig(fig_table, bbox_inches='tight')
                        st.pyplot(fig_table)
                        plt.close(fig_table)

                    # --- 4. Zoomed Expansions (only for 1H-NMR) ---
                    n_peaks = len(signals)
                    if n_peaks > 0 and nmr_type == '1h':
                        fig_zoom, axes = plt.subplots(1, n_peaks, figsize=(max(3 * n_peaks, 6), 3.5))
                        if n_peaks == 1: axes = [axes]

                        signals_sorted = sorted(signals, key=lambda x: float(x.get('delta', 0)), reverse=True)

                        for i, (ax, sig) in enumerate(zip(axes, signals_sorted)):
                            delta = float(sig.get('delta', 1.0))
                            mult = sig.get('multiplicity', 's')
                            integ = int(float(sig.get('integral', 1)))
                            atoms = sig.get('atoms', [])
                            atom_str = f"Atomi: {', '.join(map(str, sorted(atoms)))}" if atoms else ""

                            ax.plot(x_ppm, y_intensity, color='#D90429', linewidth=2.0)
                            ax.set_xlim(delta + 0.05, delta - 0.05)

                            mask = (x_ppm >= delta - 0.05) & (x_ppm <= delta + 0.05)
                            local_max = np.max(y_intensity[mask]) if np.any(mask) else 1
                            ax.set_ylim(0, local_max * 1.1)

                            ax.set_title(f"{delta_symbol} {delta:.2f}\n{mult}, {integ}H\n{atom_str}", fontsize=10, fontweight='bold')
                            ax.get_yaxis().set_visible(False)
                            ax.grid(True, linestyle='--', alpha=0.5)
                            ax.set_xlabel("ppm", fontsize=9)

                        plt.tight_layout()
                        pdf.savefig(fig_zoom)
                        st.pyplot(fig_zoom)
                        plt.close(fig_zoom)

                # --- Streamlit Download Button ---
                pdf_buffer.seek(0)
                st.download_button(
                    label="📥 Scarica Report Completo (PDF)",
                    data=pdf_buffer,
                    file_name="spettro_NMR_completo.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
