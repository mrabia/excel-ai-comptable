"""
Simple Demo of the Enhanced Agentic Excel Analysis System
"""

import pandas as pd
from pathlib import Path

# Create demo financial data
def create_demo_data():
    """Create demonstration financial data"""
    
    # Financial Statement Data
    income_statement = pd.DataFrame({
        'Account': ['Revenue', 'Cost of Goods Sold', 'Gross Profit', 'Operating Expenses', 'Net Income'],
        'Q1_2024': [500000, 300000, 200000, 120000, 80000],
        'Q2_2024': [550000, 320000, 230000, 125000, 105000],
        'Q3_2024': [580000, 340000, 240000, 130000, 110000],
        'Q4_2024': [620000, 360000, 260000, 135000, 125000]
    })
    
    # Balance Sheet Data
    balance_sheet = pd.DataFrame({
        'Account': ['Cash', 'Accounts Receivable', 'Inventory', 'Fixed Assets', 'Total Assets',
                   'Accounts Payable', 'Long-term Debt', 'Equity', 'Total Liabilities & Equity'],
        'Dec_2023': [100000, 150000, 200000, 800000, 1250000, 80000, 400000, 770000, 1250000],
        'Mar_2024': [120000, 160000, 180000, 820000, 1280000, 85000, 390000, 805000, 1280000],
        'Jun_2024': [140000, 180000, 190000, 840000, 1350000, 90000, 380000, 880000, 1350000],
        'Sep_2024': [160000, 200000, 200000, 860000, 1420000, 95000, 370000, 955000, 1420000]
    })
    
    # Budget vs Actual Data
    budget_actual = pd.DataFrame({
        'Category': ['Revenue', 'Marketing', 'Operations', 'R&D', 'Administration'],
        'Budget_Q3': [550000, 50000, 80000, 30000, 25000],
        'Actual_Q3': [580000, 52000, 78000, 35000, 28000],
        'Variance': [30000, -2000, 2000, -5000, -3000],
        'Variance_Pct': [5.45, -4.00, 2.50, -16.67, -12.00]
    })
    
    # Create Excel files
    Path("demo_data").mkdir(exist_ok=True)
    
    with pd.ExcelWriter("demo_data/financial_statements.xlsx") as writer:
        income_statement.to_excel(writer, sheet_name='Income_Statement', index=False)
        balance_sheet.to_excel(writer, sheet_name='Balance_Sheet', index=False)
        budget_actual.to_excel(writer, sheet_name='Budget_vs_Actual', index=False)
    
    print("Demo financial data created: demo_data/financial_statements.xlsx")
    return "demo_data/financial_statements.xlsx"

if __name__ == "__main__":
    print("=" * 60)
    print("ENHANCED AGENTIC EXCEL ANALYSIS SYSTEM")
    print("=" * 60)
    print()
    
    # Create demo data
    demo_file = create_demo_data()
    
    print()
    print("SYSTEM FEATURES:")
    print("1. Multi-Agent Architecture with 4 specialized agents")
    print("2. LangGraph workflow orchestration")
    print("3. Autonomous analysis type detection")
    print("4. Natural language interaction")
    print("5. Real-time web interface")
    print()
    
    print("HOW TO USE:")
    print("1. Open browser: http://localhost:8000")
    print("2. Upload: demo_data/financial_statements.xlsx")
    print("3. Chat: 'analyze this financial data'")
    print()
    
    print("The application is running with full agentic capabilities!")
    print("Multiple specialized agents will coordinate to analyze your Excel files.")