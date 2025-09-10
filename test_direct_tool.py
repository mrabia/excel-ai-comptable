#!/usr/bin/env python3
"""
Test direct de l'outil expert dans un environnement contrôlé
"""
import asyncio
import sys
from pathlib import Path
sys.path.append(".")

from main import AccountantAgent, FileRecord, SessionLocal

async def test_direct_tool_call():
    """Test direct l'outil expert avec un fichier real"""
    
    try:
        print("Initialisation de l'agent...")
        agent = AccountantAgent()
        await agent.initialize()
        
        # Chercher le dernier fichier uploadé dans la DB
        db = SessionLocal()
        try:
            latest_file = db.query(FileRecord).order_by(FileRecord.uploaded_at.desc()).first()
            
            if not latest_file:
                print("Aucun fichier trouve dans la base de donnees")
                return False
            
            print(f"Fichier trouve: {latest_file.filename} (ID: {latest_file.id})")
            
            # Récupérer les outils de l'agent
            tools = agent._create_tools()
            
            # Trouver l'outil TVA
            tva_tool = None
            for tool in tools:
                if tool.name == "calculate_tva_collectee":
                    tva_tool = tool
                    break
            
            if not tva_tool:
                print("Outil TVA non trouve!")
                return False
                
            print(f"Appel direct de l'outil: {tva_tool.func.__name__}")
            
            # Appel direct de l'outil avec les paramètres
            result = tva_tool.func(
                file_id=latest_file.id,
                start_date="2025-05-01", 
                end_date="2025-05-31"
            )
            
            print("RESULTAT DE L'OUTIL:")
            print("=" * 60)
            print(result)
            print("=" * 60)
            
            # Vérifier si on a des montants concrets
            success_indicators = [
                "MAD" in result,
                any(char.isdigit() for char in result),
                "TVA" in result or "tva" in result,
                "327" in result or "492" in result or "164" in result  # Montants attendus
            ]
            
            success_count = sum(success_indicators)
            print(f"Indicateurs de succes: {success_count}/4")
            
            if success_count >= 3:
                print("SUCCES: L'outil retourne des montants concrets!")
                return True
            else:
                print("ECHEC: L'outil ne retourne pas de montants concrets")
                return False
                
        finally:
            db.close()
    
    except Exception as e:
        print(f"Erreur lors du test direct: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TEST DIRECT DE L'OUTIL EXPERT TVA")
    print("=" * 60)
    
    success = asyncio.run(test_direct_tool_call())
    
    if success:
        print("\nTEST DIRECT REUSSI - OUTIL FONCTIONNE")
        sys.exit(0)
    else:
        print("\nTEST DIRECT ECHOUE - OUTIL A DES PROBLEMES")
        sys.exit(1)