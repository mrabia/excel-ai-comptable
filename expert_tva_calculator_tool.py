#!/usr/bin/env python3
"""
Copie du module expert_tva_calculator.py pour intégration comme outil dans l'agent
Version dédiée pour l'agent AccountantAgent
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import re

# Configuration colonnes possibles (même que l'original)
COL_MAP = {
    "account": [
        "code_compte", "compte", "account_code", "account", "code compte",
        "n° compte", "numero_compte", "numerocompte", "codecompte", "code"
    ],
    "date": [
        "date_ecriture", "date", "date_operation", "date_mouvement",
        "periode", "date ecriture", "date operation", "date_pièce",
        "datepiece", "dateecriture"
    ],
    "debit": [
        "debit", "débit", "montant_debit", "montant debit", "deb", 
        "solde_debit", "solde debit"
    ],
    "credit": [
        "credit", "crédit", "montant_credit", "montant credit", "cred", 
        "solde_credit", "solde credit"
    ]
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise les noms de colonnes (minuscules, sans espaces/accents)"""
    df = df.copy()
    new_cols = {}
    for col in df.columns:
        normalized = str(col).lower().strip()
        normalized = normalized.replace(" ", "_")
        normalized = normalized.replace("é", "e").replace("è", "e")
        normalized = normalized.replace("à", "a").replace("ç", "c")
        new_cols[col] = normalized
    df = df.rename(columns=new_cols)
    return df

def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Détecte une colonne selon une liste de candidats possibles"""
    for col in df.columns:
        col_clean = str(col).lower().strip()
        for candidate in candidates:
            if candidate.lower() in col_clean:
                return col
    return None

def is_tva_account_series(account_series: pd.Series) -> pd.Series:
    """Retourne un masque booléen pour les comptes 445 (TVA)"""
    return account_series.astype(str).str.startswith('445', na=False)

def coerce_types(df: pd.DataFrame, date_col: str, debit_col: str, credit_col: str) -> pd.DataFrame:
    """Force les types de données appropriés"""
    df = df.copy()
    
    # Dates
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # Montants numériques
    for col in [debit_col, credit_col]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    return df

def filter_period(df: pd.DataFrame, date_col: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Filtre les données selon la période"""
    if date_col is None or date_col not in df.columns:
        return df
    
    mask = (df[date_col] >= start_dt) & (df[date_col] <= end_dt)
    return df[mask]

def calculate_tva_collectee_expert_tool(sheets: Dict[str, pd.DataFrame], start_date: str, end_date: str) -> str:
    """
    EXPERT DROP-IN: Calculate TVA collectée using 100% robust methodology
    Version adaptée pour intégration comme outil dans l'agent
    """
    
    # 1) Parser les dates
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
    except Exception:
        return "Periode invalide. Format attendu: YYYY-MM-DD."
    
    # 2) Agrégation TVA nette sur toutes les feuilles qui contiennent des écritures
    results = []
    total_entries_processed = 0
    total_sheets_scanned = len(sheets)
    
    for sheet_name, raw_df in sheets.items():
        if raw_df is None or len(raw_df) == 0:
            continue
        
        df = normalize_columns(raw_df)
        
        # Détection colonnes minimales
        account_col = detect_col(df, COL_MAP["account"])
        date_col    = detect_col(df, COL_MAP["date"])
        debit_col   = detect_col(df, COL_MAP["debit"])
        credit_col  = detect_col(df, COL_MAP["credit"])
        
        # Si une colonne clé manque, on skippe la feuille MAIS on logge explicitement
        missing = [n for n, c in [("compte", account_col), ("date", date_col), ("debit", debit_col), ("credit", credit_col)] if c is None]
        if missing:
            results.append({
                "sheet": sheet_name, 
                "status": "colonnes_manquantes", 
                "missing": missing,
                "total_rows": len(df)
            })
            continue
        
        df = coerce_types(df, date_col, debit_col, credit_col)
        if date_col:
            df = df[df[date_col].notna()]
        
        # 3) Filtrer TVA (comptes 445…)
        if account_col is None:
            continue
        df_tva = df[is_tva_account_series(df[account_col])]
        if df_tva.empty:
            results.append({
                "sheet": sheet_name, 
                "status": "pas_de_comptes_445",
                "total_rows": len(df)
            })
            continue
        
        # 4) Filtrer période
        df_may = filter_period(df_tva, date_col, start_dt, end_dt)
        if df_may.empty:
            results.append({
                "sheet": sheet_name, 
                "status": "ok_pas_de_mouvement_periode",
                "comptes_445_hors_periode": len(df_tva)
            })
            continue
        
        # 5) Calcul TVA nette (Crédit – Débit)
        credits_sum = float(df_may[credit_col].sum()) if credit_col else 0.0
        debits_sum = float(df_may[debit_col].sum()) if debit_col else 0.0
        tva_net = credits_sum - debits_sum
        total_entries_processed += len(df_may)
        
        results.append({
            "sheet": sheet_name, 
            "status": "ok", 
            "tva_net": tva_net, 
            "entries": len(df_may), 
            "credits": credits_sum, 
            "debits": debits_sum,
            "total_rows_sheet": len(df),
            "comptes_445_found": len(df_tva)
        })
    
    # 6) Synthèse: somme des feuilles "ok"
    tva_total = sum(r["tva_net"] for r in results if r.get("status") == "ok")
    total_credits = sum(r["credits"] for r in results if r.get("status") == "ok")
    total_debits = sum(r["debits"] for r in results if r.get("status") == "ok")
    
    missing_any = [r for r in results if r.get("status") == "colonnes_manquantes"]
    ok_sheets = [r for r in results if r.get("status") == "ok"]
    no_445_sheets = [r for r in results if r.get("status") == "pas_de_comptes_445"]
    
    # 7) Formatage de la réponse avec montants concrets
    out_lines = []
    
    # RÉSULTAT PRINCIPAL AVEC MONTANTS CONCRETS
    out_lines.append("DECLARATION TVA COLLECTEE - RESULTATS OFFICIELS")
    out_lines.append("=" * 60)
    out_lines.append(f"TVA COLLECTEE NETTE: {tva_total:,.2f} MAD")
    out_lines.append(f"Periode: {start_date} -> {end_date}")
    out_lines.append(f"Formule: Sum(Credits 445) - Sum(Debits 445)")
    out_lines.append("")
    
    # DÉTAIL DU CALCUL
    out_lines.append("DETAIL DU CALCUL:")
    out_lines.append(f"- Total Credits (445): {total_credits:,.2f} MAD")
    out_lines.append(f"- Total Debits (445): {total_debits:,.2f} MAD")
    out_lines.append(f"- TVA Nette: {tva_total:,.2f} MAD")
    out_lines.append("")
    
    # MÉTRIQUES DE TRAITEMENT
    out_lines.append("METRIQUES DE TRAITEMENT:")
    out_lines.append(f"- Ecritures TVA analysees: {total_entries_processed:,}")
    out_lines.append(f"- Feuilles exploitees: {len(ok_sheets)}/{total_sheets_scanned}")
    out_lines.append("")
    
    if ok_sheets:
        out_lines.append("DETAIL PAR FEUILLE EXPLOITEE:")
        for r in ok_sheets:
            out_lines.append(f"- '{r['sheet']}': {r['entries']} ecritures, "
                           f"Credits: {r['credits']:,.2f} MAD, "
                           f"Debits: {r['debits']:,.2f} MAD, "
                           f"Net: {r['tva_net']:,.2f} MAD")
        out_lines.append("")
    
    # AVERTISSEMENTS ET INFORMATIONS
    if missing_any or no_445_sheets:
        out_lines.append("INFORMATIONS COMPLEMENTAIRES:")
        
        for r in missing_any:
            out_lines.append(f"- Feuille '{r['sheet']}' non exploitee: colonnes manquantes {r['missing']}")
        
        for r in no_445_sheets:
            out_lines.append(f"- Feuille '{r['sheet']}' sans comptes 445 (normal pour certains types de documents)")
        
        out_lines.append("")
    
    # RECOMMANDATIONS
    out_lines.append("RECOMMANDATIONS:")
    out_lines.append("- Verifiez que tous les comptes TVA sont bien codifies en 445xxxx")
    out_lines.append("- Controlez la coherence avec le grand livre comptable")
    out_lines.append("- Archivez ce rapport pour audit fiscal")
    
    return "\n".join(out_lines).replace(",", " ")

def create_expert_tva_tool_for_agent():
    """
    Crée une fonction wrapper pour l'intégration dans l'agent
    Cette fonction sera utilisée comme outil dans l'AgentExecutor
    """
    def expert_tva_wrapper(file_path: str, start_date: str, end_date: str) -> str:
        """
        Outil expert TVA pour l'agent - calcule la TVA collectée avec méthodologie robuste
        
        Args:
            file_path: Chemin vers le fichier Excel
            start_date: Date de début (YYYY-MM-DD)
            end_date: Date de fin (YYYY-MM-DD)
            
        Returns:
            Rapport détaillé avec montants concrets en MAD
        """
        try:
            # Lecture des feuilles Excel avec pandas
            sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
            
            # Appel du calculateur expert
            return calculate_tva_collectee_expert_tool(sheets, start_date, end_date)
            
        except Exception as e:
            return f"Erreur lors de l'analyse du fichier Excel: {str(e)}"
    
    return expert_tva_wrapper

# Test standalone si exécuté directement
if __name__ == "__main__":
    print("Module expert_tva_calculator_tool.py - Version agent")
    print("Fonction create_expert_tva_tool_for_agent() disponible pour intégration")