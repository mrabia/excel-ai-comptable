"""
Enhanced Agentic Excel Analysis System
Using LangChain and LangGraph for multi-agent coordination
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# LangChain imports
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# LangGraph imports  
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExcelAnalysisState(TypedDict):
    """State object for the Excel analysis workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    file_path: str
    file_data: Dict[str, Any]
    analysis_type: str
    analysis_results: Dict[str, Any]
    recommendations: List[str]
    next_action: str

class ExcelAnalysisTools:
    """Collection of Excel analysis tools for agents"""
    
    @tool
    def load_excel_file(file_path: str) -> Dict[str, Any]:
        """Load and analyze Excel file structure"""
        try:
            # Load Excel file
            excel_file = pd.ExcelFile(file_path)
            sheets_data = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets_data[sheet_name] = {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "null_counts": df.isnull().sum().to_dict(),
                    "sample_data": df.head(3).to_dict(),
                    "summary_stats": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
                }
            
            return {
                "sheet_names": excel_file.sheet_names,
                "sheets_data": sheets_data,
                "file_size": os.path.getsize(file_path),
                "analysis_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            return {"error": str(e)}

    @tool
    def calculate_financial_ratios(sheet_data: Dict, config: Dict) -> Dict[str, Any]:
        """Calculate financial ratios from Excel data"""
        try:
            ratios = {}
            
            # Extract column mappings from config
            revenue_col = config.get("revenue_column")
            cost_col = config.get("cost_column") 
            assets_col = config.get("assets_column")
            liabilities_col = config.get("liabilities_column")
            
            if revenue_col and cost_col:
                revenue = sum([row.get(revenue_col, 0) for row in sheet_data.get("sample_data", {}).values()])
                cost = sum([row.get(cost_col, 0) for row in sheet_data.get("sample_data", {}).values()])
                if revenue > 0:
                    ratios["gross_margin"] = ((revenue - cost) / revenue) * 100
            
            if assets_col and liabilities_col:
                assets = sum([row.get(assets_col, 0) for row in sheet_data.get("sample_data", {}).values()])
                liabilities = sum([row.get(liabilities_col, 0) for row in sheet_data.get("sample_data", {}).values()])
                if liabilities > 0:
                    ratios["debt_to_asset"] = (liabilities / assets) * 100
            
            return {
                "ratios": ratios,
                "analysis_date": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
            return {"error": str(e)}

    @tool
    def perform_variance_analysis(budget_data: Dict, actual_data: Dict) -> Dict[str, Any]:
        """Perform variance analysis between budget and actual data"""
        try:
            variances = {}
            
            # Compare numeric columns
            for col in budget_data.get("columns", []):
                if col in actual_data.get("columns", []):
                    budget_val = budget_data.get("summary_stats", {}).get(col, {}).get("mean", 0)
                    actual_val = actual_data.get("summary_stats", {}).get(col, {}).get("mean", 0)
                    
                    if budget_val != 0:
                        variance_pct = ((actual_val - budget_val) / budget_val) * 100
                        variances[col] = {
                            "budget": budget_val,
                            "actual": actual_val,
                            "variance": actual_val - budget_val,
                            "variance_pct": variance_pct
                        }
            
            return {
                "variances": variances,
                "analysis_date": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in variance analysis: {e}")
            return {"error": str(e)}

    @tool
    def generate_audit_trail(data: Dict, threshold: float = 10000) -> Dict[str, Any]:
        """Generate audit trail for high-value transactions"""
        try:
            audit_records = []
            
            for sheet_name, sheet_data in data.get("sheets_data", {}).items():
                sample_data = sheet_data.get("sample_data", {})
                
                # Check each numeric column for values above threshold
                for col, values in sample_data.items():
                    if isinstance(values, dict):
                        for idx, value in values.items():
                            if isinstance(value, (int, float)) and abs(value) > threshold:
                                audit_records.append({
                                    "sheet": sheet_name,
                                    "column": col,
                                    "row": idx,
                                    "value": value,
                                    "flag_reason": f"Value exceeds threshold of {threshold}"
                                })
            
            return {
                "audit_records": audit_records,
                "total_flagged": len(audit_records),
                "threshold_used": threshold,
                "analysis_date": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error generating audit trail: {e}")
            return {"error": str(e)}

class AgenticExcelAnalyzer:
    """Multi-agent Excel analysis system using LangGraph"""
    
    def __init__(self):
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize tools
        self.tools = [
            ExcelAnalysisTools.load_excel_file,
            ExcelAnalysisTools.calculate_financial_ratios, 
            ExcelAnalysisTools.perform_variance_analysis,
            ExcelAnalysisTools.generate_audit_trail
        ]
        
        # Initialize memory
        self.memory = MemorySaver()
        
        # Create specialized agents
        self.data_analyst_agent = self._create_data_analyst_agent()
        self.financial_analyst_agent = self._create_financial_analyst_agent()
        self.audit_agent = self._create_audit_agent()
        self.coordinator_agent = self._create_coordinator_agent()
        
        # Initialize workflow
        self.workflow = self._create_workflow()
        
    def _create_data_analyst_agent(self):
        """Create specialized data analysis agent"""
        system_prompt = """You are a Data Analysis Agent specialized in Excel file processing.
        
        Your responsibilities:
        1. Load and examine Excel file structure
        2. Analyze data quality and completeness
        3. Identify patterns and anomalies in data
        4. Provide data summary and insights
        
        Always use the available tools to analyze data thoroughly.
        Focus on data structure, quality, and initial insights."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}")
        ])
        
        return create_react_agent(self.llm, [self.tools[0]])
    
    def _create_financial_analyst_agent(self):
        """Create specialized financial analysis agent"""
        system_prompt = """You are a Financial Analysis Agent specialized in accounting and financial analysis.
        
        Your responsibilities:
        1. Calculate financial ratios and metrics
        2. Perform variance analysis between budget and actual
        3. Analyze profitability, liquidity, and efficiency
        4. Provide financial insights and recommendations
        
        Use the financial analysis tools to provide comprehensive financial insights.
        Focus on financial health, performance trends, and actionable recommendations."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt), 
            ("placeholder", "{messages}")
        ])
        
        return create_react_agent(self.llm, [self.tools[1], self.tools[2]])
    
    def _create_audit_agent(self):
        """Create specialized audit and compliance agent"""
        system_prompt = """You are an Audit Agent specialized in compliance and risk analysis.
        
        Your responsibilities:
        1. Generate audit trails for high-value transactions
        2. Identify compliance issues and risks
        3. Flag unusual patterns or outliers
        4. Provide audit recommendations
        
        Use the audit tools to ensure compliance and identify risks.
        Focus on transaction validation, compliance checking, and risk assessment."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}")
        ])
        
        return create_react_agent(self.llm, [self.tools[3]])
    
    def _create_coordinator_agent(self):
        """Create coordinator agent for orchestrating the workflow"""
        system_prompt = """You are the Coordinator Agent responsible for orchestrating Excel analysis workflows.
        
        Your responsibilities:
        1. Determine the analysis strategy based on user requests
        2. Coordinate between specialized agents
        3. Synthesize results from different agents
        4. Provide final recommendations and next steps
        
        Analyze the user request and file data to determine which agents should be involved
        and in what order. Provide comprehensive final analysis."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{messages}")
        ])
        
        return prompt | self.llm | StrOutputParser()
    
    def _create_workflow(self):
        """Create the LangGraph workflow for multi-agent coordination"""
        
        def route_analysis(state: ExcelAnalysisState):
            """Determine which agents to route to based on analysis type"""
            analysis_type = state.get("analysis_type", "comprehensive")
            
            if "financial" in analysis_type.lower():
                return "financial_analysis"
            elif "audit" in analysis_type.lower():
                return "audit_analysis" 
            elif "data" in analysis_type.lower():
                return "data_analysis"
            else:
                return "data_analysis"  # Default to data analysis first
        
        async def data_analysis_node(state: ExcelAnalysisState):
            """Data analysis agent node"""
            messages = state.get("messages", [])
            file_path = state.get("file_path", "")
            
            # Add context about the file
            analysis_message = HumanMessage(
                content=f"Please analyze the Excel file at: {file_path}. "
                       f"Provide detailed insights about the data structure, quality, and any patterns you observe."
            )
            messages.append(analysis_message)
            
            # Process with data analyst agent
            result = await self.data_analyst_agent.ainvoke({"messages": messages})
            
            return {
                "messages": result["messages"],
                "file_data": state.get("file_data", {}),
                "analysis_results": {"data_analysis": "completed"},
                "next_action": "financial_analysis"
            }
        
        async def financial_analysis_node(state: ExcelAnalysisState):
            """Financial analysis agent node"""
            messages = state.get("messages", [])
            
            financial_message = HumanMessage(
                content="Based on the data analysis results, please perform comprehensive financial analysis. "
                       "Calculate relevant financial ratios and provide insights about financial performance."
            )
            messages.append(financial_message)
            
            result = await self.financial_analyst_agent.ainvoke({"messages": messages})
            
            analysis_results = state.get("analysis_results", {})
            analysis_results["financial_analysis"] = "completed"
            
            return {
                "messages": result["messages"],
                "analysis_results": analysis_results,
                "next_action": "audit_analysis"
            }
        
        async def audit_analysis_node(state: ExcelAnalysisState):
            """Audit analysis agent node"""
            messages = state.get("messages", [])
            
            audit_message = HumanMessage(
                content="Please perform audit analysis on the data. Generate audit trails, "
                       "identify any compliance issues, and flag high-risk transactions."
            )
            messages.append(audit_message)
            
            result = await self.audit_agent.ainvoke({"messages": messages})
            
            analysis_results = state.get("analysis_results", {})
            analysis_results["audit_analysis"] = "completed"
            
            return {
                "messages": result["messages"],
                "analysis_results": analysis_results,
                "next_action": "coordination"
            }
        
        async def coordination_node(state: ExcelAnalysisState):
            """Coordinator agent node for final synthesis"""
            messages = state.get("messages", [])
            analysis_results = state.get("analysis_results", {})
            
            coordination_message = HumanMessage(
                content=f"Please synthesize all the analysis results: {analysis_results}. "
                       f"Provide comprehensive recommendations and next steps for the user."
            )
            messages.append(coordination_message)
            
            final_response = await self.coordinator_agent.ainvoke({"messages": messages})
            
            return {
                "messages": messages + [AIMessage(content=final_response)],
                "analysis_results": analysis_results,
                "recommendations": [final_response],
                "next_action": "complete"
            }
        
        # Build the workflow graph
        workflow = StateGraph(ExcelAnalysisState)
        
        # Add nodes
        workflow.add_node("data_analysis", data_analysis_node)
        workflow.add_node("financial_analysis", financial_analysis_node)
        workflow.add_node("audit_analysis", audit_analysis_node)
        workflow.add_node("coordination", coordination_node)
        
        # Add edges
        workflow.add_edge(START, "data_analysis")
        workflow.add_edge("data_analysis", "financial_analysis")
        workflow.add_edge("financial_analysis", "audit_analysis")
        workflow.add_edge("audit_analysis", "coordination")
        workflow.add_edge("coordination", END)
        
        return workflow.compile(checkpointer=self.memory)
    
    async def analyze_excel_file(self, file_path: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Main method to analyze Excel file using multi-agent workflow"""
        try:
            # Initialize state
            initial_state = ExcelAnalysisState(
                messages=[HumanMessage(content=f"Analyze Excel file: {file_path}")],
                file_path=file_path,
                file_data={},
                analysis_type=analysis_type,
                analysis_results={},
                recommendations=[],
                next_action="start"
            )
            
            # Run the workflow
            config = {"configurable": {"thread_id": f"excel_analysis_{datetime.now().timestamp()}"}}
            
            result = await self.workflow.ainvoke(initial_state, config)
            
            return {
                "status": "success",
                "file_path": file_path,
                "analysis_type": analysis_type,
                "analysis_results": result.get("analysis_results", {}),
                "recommendations": result.get("recommendations", []),
                "messages": [msg.content for msg in result.get("messages", []) if hasattr(msg, 'content')],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in Excel analysis workflow: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# Example usage and testing
async def main():
    """Test the agentic Excel analyzer"""
    analyzer = AgenticExcelAnalyzer()
    
    # Create a sample Excel file for testing
    sample_data = {
        "Revenue": [100000, 150000, 120000, 180000],
        "Costs": [60000, 90000, 70000, 110000],
        "Assets": [500000, 520000, 540000, 580000],
        "Liabilities": [200000, 210000, 220000, 250000]
    }
    
    df = pd.DataFrame(sample_data)
    test_file_path = "test_financial_data.xlsx"
    df.to_excel(test_file_path, index=False)
    
    print("Starting multi-agent Excel analysis...")
    
    # Analyze the Excel file
    result = await analyzer.analyze_excel_file(test_file_path, "comprehensive")
    
    print("Analysis Results:")
    print(json.dumps(result, indent=2))
    
    # Clean up test file
    os.remove(test_file_path)

if __name__ == "__main__":
    asyncio.run(main())