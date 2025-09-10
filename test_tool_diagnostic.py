#!/usr/bin/env python3
"""
Test de diagnostic - Vérifier si l'outil TVA expert est disponible dans l'agent
"""
import asyncio
import sys
from pathlib import Path
sys.path.append(".")

from main import AccountantAgent

async def test_tool_availability():
    """Test si l'outil calculate_tva_collectee_expert est disponible"""
    
    try:
        print("Initialisation de l'agent...")
        agent = AccountantAgent()
        await agent.initialize()
        
        # Lister tous les outils disponibles
        tools = agent._create_tools()
        print(f"Nombre d'outils trouves: {len(tools)}")
        
        print("\nOUTILS DISPONIBLES:")
        for i, tool in enumerate(tools):
            print(f"{i+1}. {tool.name} - {tool.description[:80]}...")
        
        # Chercher spécifiquement l'outil TVA
        tva_tool = None
        for tool in tools:
            if tool.name == "calculate_tva_collectee":
                tva_tool = tool
                break
        
        if tva_tool:
            print(f"\nOUTIL TVA TROUVE!")
            print(f"   Nom: {tva_tool.name}")
            print(f"   Description: {tva_tool.description}")
            print(f"   Fonction: {tva_tool.func.__name__}")
            return True
        else:
            print(f"\nOUTIL TVA NON TROUVE!")
            return False
    
    except Exception as e:
        print(f"Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("TEST DIAGNOSTIC - DISPONIBILITÉ OUTIL TVA EXPERT")
    print("=" * 60)
    
    success = asyncio.run(test_tool_availability())
    
    if success:
        print("\nTEST DIAGNOSTIC REUSSI - OUTIL DISPONIBLE")
        sys.exit(0)
    else:
        print("\nTEST DIAGNOSTIC ECHOUE - OUTIL NON DISPONIBLE")
        sys.exit(1)