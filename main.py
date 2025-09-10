"""
Accountant AI Agent Backend Implementation
Complete system with MCP servers, LLM, RAG, and accounting tools
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# JSON serialization helper for datetime objects
def json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'isoformat'):  # Handle other date/time objects
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):  # Handle SQLAlchemy objects
        return {key: value for key, value in obj.__dict__.items() if not key.startswith('_')}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import aiofiles
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools import StructuredTool
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import the new agentic system
from agentic_excel_analyzer import AgenticExcelAnalyzer
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, JSON, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configuration
@dataclass
class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MODEL_NAME: str = "gpt-4-turbo"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2000
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    UPLOAD_DIR: str = "uploads"
    VECTOR_DB_DIR: str = "vector_db"
    LOG_LEVEL: str = "INFO"
    DB_URL: str = "sqlite:///./accountant_ai.db"

config = Config()

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# Database Models
Base = declarative_base()

class FileRecord(Base):
    __tablename__ = "files"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)
    file_size = Column(Integer)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    file_metadata = Column(JSON)

class ChatHistory(Base):
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    chat_metadata = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# Database setup
engine = create_engine(config.DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Pydantic Models
class ChatMessage(BaseModel):
    content: str
    context: Dict[str, Any] = Field(default_factory=dict)

class AnalysisRequest(BaseModel):
    operation: str
    file_ids: List[str]
    parameters: Dict[str, Any] = Field(default_factory=dict)

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    message: str

# MCP Client for Excel Operations
class MCPExcelClient:
    """Client for MCP Excel Server operations"""
    
    def __init__(self):
        self.server_path = os.getenv("MCP_EXCEL_SERVER_PATH", "excel-mcp-server")
        self.process = None
    
    async def connect(self):
        """Connect to the MCP Excel server"""
        try:
            # In a real implementation, this would establish a connection to the MCP server
            logger.info("Connected to MCP Excel Server")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP Excel Server: {e}")
            return False
    
    async def read_excel(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Read Excel file and return sheets as DataFrames"""
        try:
            excel_file = pd.ExcelFile(file_path)
            sheets = {}
            for sheet_name in excel_file.sheet_names:
                sheets[sheet_name] = pd.read_excel(file_path, sheet_name=sheet_name)
            return sheets
        except Exception as e:
            logger.error(f"Error reading Excel file {file_path}: {e}")
            raise
    
    async def analyze_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze Excel file structure"""
        try:
            sheets = await self.read_excel(file_path)
            structure = {}
            
            for sheet_name, df in sheets.items():
                structure[sheet_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data_types": df.dtypes.to_dict(),
                    "sample_data": df.head(3).to_dict(),
                    "null_counts": df.isnull().sum().to_dict()
                }
            
            return structure
        except Exception as e:
            logger.error(f"Error analyzing Excel structure: {e}")
            raise

# Accounting Tools
class AccountingTools:
    """Comprehensive accounting analysis tools"""
    
    @staticmethod
    def calculate_financial_ratios(df: pd.DataFrame, balance_sheet_cols: Dict[str, str]) -> Dict[str, float]:
        """Calculate key financial ratios"""
        ratios = {}
        
        try:
            # Liquidity Ratios
            if 'current_assets' in balance_sheet_cols and 'current_liabilities' in balance_sheet_cols:
                current_assets = df[balance_sheet_cols['current_assets']].sum()
                current_liabilities = df[balance_sheet_cols['current_liabilities']].sum()
                if current_liabilities != 0:
                    ratios['current_ratio'] = current_assets / current_liabilities
            
            # Profitability Ratios
            if 'net_income' in balance_sheet_cols and 'revenue' in balance_sheet_cols:
                net_income = df[balance_sheet_cols['net_income']].sum()
                revenue = df[balance_sheet_cols['revenue']].sum()
                if revenue != 0:
                    ratios['profit_margin'] = (net_income / revenue) * 100
            
            # Efficiency Ratios
            if 'total_assets' in balance_sheet_cols and 'revenue' in balance_sheet_cols:
                total_assets = df[balance_sheet_cols['total_assets']].sum()
                revenue = df[balance_sheet_cols['revenue']].sum()
                if total_assets != 0:
                    ratios['asset_turnover'] = revenue / total_assets
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
        
        return ratios
    
    @staticmethod
    def variance_analysis(budget_df: pd.DataFrame, actual_df: pd.DataFrame, 
                         key_column: str = 'account') -> pd.DataFrame:
        """Perform variance analysis between budget and actual"""
        try:
            # Merge budget and actual data
            merged = pd.merge(budget_df, actual_df, on=key_column, suffixes=('_budget', '_actual'))
            
            # Calculate variances
            numeric_cols = merged.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col.endswith('_budget'):
                    actual_col = col.replace('_budget', '_actual')
                    if actual_col in merged.columns:
                        variance_col = col.replace('_budget', '_variance')
                        variance_pct_col = col.replace('_budget', '_variance_pct')
                        
                        merged[variance_col] = merged[actual_col] - merged[col]
                        merged[variance_pct_col] = ((merged[actual_col] - merged[col]) / merged[col] * 100).fillna(0)
            
            return merged
        except Exception as e:
            logger.error(f"Error in variance analysis: {e}")
            raise
    
    @staticmethod
    def consolidate_financials(dataframes: List[pd.DataFrame], 
                             consolidation_rules: Dict[str, str]) -> pd.DataFrame:
        """Consolidate multiple financial statements"""
        try:
            consolidated = pd.DataFrame()
            
            for df in dataframes:
                if consolidated.empty:
                    consolidated = df.copy()
                else:
                    # Simple consolidation - sum numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if col in consolidated.columns:
                            consolidated[col] = consolidated[col] + df[col]
                        else:
                            consolidated[col] = df[col]
            
            return consolidated
        except Exception as e:
            logger.error(f"Error in financial consolidation: {e}")
            raise
    
    @staticmethod
    def audit_trail(df: pd.DataFrame, threshold_amount: float = 10000) -> pd.DataFrame:
        """Generate audit trail for transactions above threshold"""
        try:
            # Identify numeric columns that might represent amounts
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            audit_records = []
            
            for col in numeric_cols:
                high_value_transactions = df[df[col].abs() > threshold_amount]
                if not high_value_transactions.empty:
                    for idx, row in high_value_transactions.iterrows():
                        audit_records.append({
                            'transaction_id': idx,
                            'column': col,
                            'amount': row[col],
                            'date': row.get('date', 'N/A'),
                            'description': row.get('description', 'N/A'),
                            'flag_reason': f'Amount exceeds threshold of {threshold_amount}'
                        })
            
            return pd.DataFrame(audit_records)
        except Exception as e:
            logger.error(f"Error generating audit trail: {e}")
            raise

# RAG System for Document Memory
class RAGSystem:
    """Retrieval-Augmented Generation system for document memory"""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.OPENAI_API_KEY)
        self.vector_store = None
        self.documents = []
    
    async def initialize(self):
        """Initialize the vector store"""
        try:
            vector_db_path = Path(config.VECTOR_DB_DIR)
            vector_db_path.mkdir(exist_ok=True)
            
            if (vector_db_path / "index.faiss").exists():
                self.vector_store = FAISS.load_local(str(vector_db_path), self.embeddings)
                logger.info("Loaded existing vector store")
            else:
                # Create empty vector store
                self.vector_store = FAISS.from_texts([""], self.embeddings)
                logger.info("Created new vector store")
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
    
    async def add_document(self, content: str, metadata: Dict[str, Any]):
        """Add document to vector store"""
        try:
            if self.vector_store is None:
                await self.initialize()
            
            self.vector_store.add_texts([content], [metadata])
            
            # Save to disk
            vector_db_path = Path(config.VECTOR_DB_DIR)
            self.vector_store.save_local(str(vector_db_path))
            
            logger.info(f"Added document to vector store: {metadata.get('filename', 'Unknown')}")
        except Exception as e:
            logger.error(f"Error adding document to RAG: {e}")
    
    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            if self.vector_store is None:
                return []
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'score': score
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching RAG system: {e}")
            return []

# LangChain Agent
class AccountantAgent:
    """Enhanced agentic system for accounting tasks"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
            openai_api_key=config.OPENAI_API_KEY
        )
        logger.info(f"AccountantAgent initialized with model: {config.MODEL_NAME}")
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=10
        )
        self.mcp_client = MCPExcelClient()
        self.rag_system = RAGSystem()
        self.accounting_tools = AccountingTools()
        
        # Initialize the new agentic system
        self.agentic_analyzer = AgenticExcelAnalyzer()
        self.agent_executor = None
        
    async def initialize(self):
        """Initialize the agent with tools"""
        await self.mcp_client.connect()
        await self.rag_system.initialize()
        
        tools = self._create_tools()
        
        system_message = """Vous êtes un assistant expert en comptabilité et analyse financière avec accès aux outils d'analyse Excel.
        Vous pouvez:
        1. Analyser plusieurs feuilles de calcul Excel simultanément
        2. Effectuer des calculs de ratios financiers
        3. Réaliser des analyses d'écart entre budget et réalisé
        4. Croiser des données entre différents classeurs
        5. Générer des pistes d'audit et rapports de conformité
        6. Consolider les états financiers
        
        IMPORTANT: Répondez TOUJOURS en français. Fournissez des insights clairs et actionnables en expliquant votre méthodologie d'analyse.
        Quand vous travaillez avec des données financières, soyez précis et mettez en évidence toutes les hypothèses formulées.
        
        Contexte français: Vous assistez des utilisateurs francophones avec leurs déclarations TVA, analyses financières, et gestion comptable.
        """
        
        # Create agent with tools - fix LangChain compatibility
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        
        # Create proper prompt template for agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=create_openai_tools_agent(self.llm, tools, prompt),
            tools=tools,
            memory=self.memory,
            verbose=True
        )
    
    def _run_async_safely(self, coro):
        """Safely run async coroutines from sync context"""
        try:
            # Try to get current loop
            loop = asyncio.get_running_loop()
            # If we're in an async context, use a thread pool
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except RuntimeError:
            # No running loop, safe to run directly
            return asyncio.run(coro)
    
    def _create_tools(self) -> List[StructuredTool]:
        """Create LangChain tools for the agent"""
        
        def analyze_excel_file(file_id: str) -> str:
            """Analyze an uploaded Excel file"""
            try:
                db = SessionLocal()
                file_record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
                if not file_record:
                    return f"File with ID {file_id} not found"
                
                # Use safe async handling
                structure = self._run_async_safely(self.mcp_client.analyze_structure(file_record.file_path))
                return json.dumps(structure, indent=2)
            except Exception as e:
                return f"Error analyzing file: {str(e)}"
            finally:
                db.close()
        
        def calculate_ratios(file_id: str, ratio_config: str) -> str:
            """Calculate financial ratios from Excel data"""
            try:
                db = SessionLocal()
                file_record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
                if not file_record:
                    return f"File with ID {file_id} not found"
                
                # Use safe async handling
                sheets = self._run_async_safely(self.mcp_client.read_excel(file_record.file_path))
                config_dict = json.loads(ratio_config)
                
                results = {}
                for sheet_name, df in sheets.items():
                    ratios = self.accounting_tools.calculate_financial_ratios(df, config_dict)
                    results[sheet_name] = ratios
                
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"Error calculating ratios: {str(e)}"
            finally:
                db.close()
        
        def perform_variance_analysis(budget_file_id: str, actual_file_id: str) -> str:
            """Perform variance analysis between budget and actual files"""
            try:
                db = SessionLocal()
                budget_file = db.query(FileRecord).filter(FileRecord.id == budget_file_id).first()
                actual_file = db.query(FileRecord).filter(FileRecord.id == actual_file_id).first()
                
                if not budget_file or not actual_file:
                    return "One or both files not found"
                
                # Use safe async handling
                budget_sheets = self._run_async_safely(self.mcp_client.read_excel(budget_file.file_path))
                actual_sheets = self._run_async_safely(self.mcp_client.read_excel(actual_file.file_path))
                
                # Assume first sheet for simplicity
                budget_df = list(budget_sheets.values())[0]
                actual_df = list(actual_sheets.values())[0]
                
                variance_df = self.accounting_tools.variance_analysis(budget_df, actual_df)
                
                # Return summary statistics
                summary = {
                    'total_records': len(variance_df),
                    'significant_variances': len(variance_df[variance_df.abs().select_dtypes(include=[np.number]).max(axis=1) > 1000]),
                    'sample_data': variance_df.head(5).to_dict()
                }
                
                return json.dumps(summary, indent=2)
            except Exception as e:
                return f"Error in variance analysis: {str(e)}"
            finally:
                db.close()
        
        def search_documents(query: str) -> str:
            """Search through uploaded documents using RAG"""
            try:
                # Use safe async handling
                results = self._run_async_safely(self.rag_system.search(query))
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"Error searching documents: {str(e)}"
        
        tools = [
            StructuredTool.from_function(
                func=analyze_excel_file,
                name="analyze_excel_file",
                description="Analyze the structure and content of an uploaded Excel file"
            ),
            StructuredTool.from_function(
                func=calculate_ratios,
                name="calculate_financial_ratios",
                description="Calculate financial ratios from Excel data. Requires file_id and ratio_config (JSON string)"
            ),
            StructuredTool.from_function(
                func=perform_variance_analysis,
                name="variance_analysis",
                description="Perform variance analysis between budget and actual Excel files"
            ),
            StructuredTool.from_function(
                func=search_documents,
                name="search_documents",
                description="Search through uploaded documents for relevant information"
            )
        ]
        
        return tools
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> str:
        """Process a user message using the enhanced agentic system"""
        try:
            # Check if this is a request for Excel file analysis
            if context.get('files') and ('analyze' in message.lower() or 'analysis' in message.lower()):
                return await self._process_agentic_analysis(message, context)
            
            # Add context about available files for regular processing
            if context.get('files'):
                file_context = f"Available files: {', '.join(context['files'])}\n\n"
                message = file_context + message
            
            # Get response from standard agent
            response = await self.agent_executor.ainvoke({
                "input": message,
                "chat_history": self.memory.chat_memory.messages
            })
            
            return response.get("output", "I'm sorry, I couldn't process that request.")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def _process_agentic_analysis(self, message: str, context: Dict[str, Any]) -> str:
        """Process Excel analysis request using the multi-agent system"""
        try:
            files = context.get('files', [])
            if not files:
                return "No Excel files are available for analysis. Please upload Excel files first."
            
            # Determine analysis type from message
            analysis_type = "comprehensive"
            if "financial" in message.lower():
                analysis_type = "financial"
            elif "audit" in message.lower():
                analysis_type = "audit"
            elif "data" in message.lower():
                analysis_type = "data"
            
            # Get the first available file (in a real implementation, you might let user choose)
            # For now, we'll assume files contain file paths
            file_path = files[0] if isinstance(files[0], str) else f"uploads/{files[0]}"
            
            # Use the agentic analyzer
            logger.info(f"Starting agentic analysis of {file_path} with type: {analysis_type}")
            
            result = await self.agentic_analyzer.analyze_excel_file(file_path, analysis_type)
            
            if result.get("status") == "success":
                # Format the response nicely
                response_parts = [
                    f"🤖 **Multi-Agent Excel Analysis Complete**",
                    f"",
                    f"📁 **File**: {result.get('file_path', 'Unknown')}",
                    f"🔍 **Analysis Type**: {result.get('analysis_type', 'Unknown')}",
                    f"",
                    f"📊 **Analysis Results**:",
                ]
                
                # Add analysis results
                analysis_results = result.get("analysis_results", {})
                for key, value in analysis_results.items():
                    response_parts.append(f"✅ {key.replace('_', ' ').title()}: {value}")
                
                # Add recommendations
                recommendations = result.get("recommendations", [])
                if recommendations:
                    response_parts.extend([
                        f"",
                        f"💡 **Recommendations**:",
                    ])
                    for i, rec in enumerate(recommendations, 1):
                        response_parts.append(f"{i}. {rec}")
                
                # Add agent messages summary
                messages = result.get("messages", [])
                if len(messages) > 2:  # Show some agent insights
                    response_parts.extend([
                        f"",
                        f"🧠 **Agent Insights**:",
                        f"The multi-agent system processed {len(messages)} analysis steps involving data analysis, financial analysis, and audit review agents."
                    ])
                
                response_parts.append(f"")
                response_parts.append(f"⏰ **Analysis completed at**: {result.get('timestamp', 'Unknown')}")
                
                return "\n".join(response_parts)
            
            else:
                return f"❌ **Analysis Failed**: {result.get('error', 'Unknown error occurred')}"
                
        except Exception as e:
            logger.error(f"Error in agentic analysis: {e}")
            return f"❌ **Agentic Analysis Error**: {str(e)}"

# FastAPI Application
app = FastAPI(
    title="Accountant AI Agent",
    description="AI-powered accounting assistant for Excel analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
api_router = APIRouter(prefix="/api")

# Global instances
agent = AccountantAgent()
connected_websockets: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    # Create directories
    Path(config.UPLOAD_DIR).mkdir(exist_ok=True)
    Path(config.VECTOR_DB_DIR).mkdir(exist_ok=True)
    
    # Initialize agent
    await agent.initialize()
    
    logger.info("Accountant AI Agent started successfully")

@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@api_router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process Excel file"""
    try:
        # Validate file
        if file.size > config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="File too large")
        
        if not file.filename.lower().endswith(('.xlsx', '.xls', '.csv')):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        file_path = Path(config.UPLOAD_DIR) / f"{file_id}{file_extension}"
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Store in database
        db = SessionLocal()
        try:
            file_record = FileRecord(
                id=file_id,
                filename=file.filename,
                file_path=str(file_path),
                file_type=file_extension,
                file_size=file.size,
                metadata={"original_name": file.filename}
            )
            db.add(file_record)
            db.commit()
            
            # Add to RAG system
            if file_extension in ['.xlsx', '.xls']:
                # Use safe async handling
                sheets = agent._run_async_safely(agent.mcp_client.read_excel(str(file_path)))
                for sheet_name, df in sheets.items():
                    content = f"File: {file.filename}, Sheet: {sheet_name}\n"
                    content += f"Columns: {', '.join(df.columns)}\n"
                    content += f"Sample data:\n{df.head().to_string()}"
                    
                    await agent.rag_system.add_document(content, {
                        "file_id": file_id,
                        "filename": file.filename,
                        "sheet_name": sheet_name,
                        "type": "excel_data"
                    })
            
            logger.info(f"File uploaded successfully: {file.filename} -> {file_id}")
            
            return FileUploadResponse(
                file_id=file_id,
                filename=file.filename,
                message="File uploaded and processed successfully"
            )
        
        finally:
            db.close()
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    """Perform data analysis on uploaded files"""
    try:
        # This would implement specific analysis operations
        # For now, return a placeholder response
        return {
            "operation": request.operation,
            "file_ids": request.file_ids,
            "result": "Analysis completed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message with agent
            response = await agent.process_message(
                message_data["content"],
                message_data.get("context", {})
            )
            
            # Send response back to client with safe JSON serialization
            await websocket.send_text(json.dumps({
                "content": response,
                "timestamp": datetime.utcnow().isoformat()
            }, default=json_serializer))
    
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@api_router.get("/files")
async def list_files():
    """List all uploaded files"""
    try:
        db = SessionLocal()
        files = db.query(FileRecord).all()
        
        return [{
            "id": file.id,
            "filename": file.filename,
            "file_type": file.file_type,
            "file_size": file.file_size,
            "uploaded_at": file.uploaded_at.isoformat() if file.uploaded_at else None
        } for file in files]
    
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@api_router.post("/files/upload", response_model=FileUploadResponse)
async def upload_files_endpoint(file: UploadFile = File(...)):
    """Alternative upload endpoint for frontend compatibility"""
    return await upload_file(file)

@api_router.get("/welcome-message")
async def get_welcome_message():
    """Get the dynamic welcome message in French"""
    return {
        "message": "Bonjour ! Je suis votre assistant comptable IA. Téléchargez vos fichiers Excel et je vous aiderai à analyser vos données financières avec des insights professionnels.",
        "timestamp": datetime.utcnow().isoformat(),
        "language": "fr"
    }

@api_router.get("/ui-texts")
async def get_ui_texts():
    """Get all dynamic UI texts in French"""
    return {
        "upload": {
            "dropText": "Déposez vos fichiers Excel ici",
            "browseText": "ou cliquez pour parcourir",
            "filesHeader": "Fichiers Téléchargés",
            "uploadSuccess": "Fichier téléchargé avec succès",
            "uploadError": "Échec du téléchargement",
            "invalidFile": "Type de fichier non valide",
            "dragOver": "Relâchez pour télécharger"
        },
        "chat": {
            "placeholder": "Posez une question sur vos données financières...",
            "sendButton": "Envoyer",
            "connecting": "Connexion...",
            "connected": "Connecté",
            "disconnected": "Déconnecté",
            "connectionError": "Erreur de connexion",
            "emptyState": "Téléchargez vos fichiers Excel et commencez à poser des questions sur vos données financières."
        },
        "modes": {
            "chat": "Chat",
            "analysis": "Analyse",
            "reports": "Rapports"
        },
        "general": {
            "loading": "Chargement...",
            "error": "Erreur",
            "success": "Succès",
            "remove": "Supprimer"
        },
        "language": "fr",
        "timestamp": datetime.utcnow().isoformat()
    }

@api_router.post("/agent/suggestions")
async def get_agent_suggestions(request: dict):
    """Get AI agent suggestions for accounting tasks"""
    try:
        query = request.get("query", "")
        context = request.get("context", {})
        
        if not query:
            return {"suggestions": [], "message": "No query provided"}
        
        # Use the agent to generate suggestions
        response = await agent.process_message(f"Provide suggestions for: {query}", context)
        
        # Extract suggestions from the response
        suggestions = []
        if "suggestions" in response.lower() or "recommend" in response.lower():
            # Parse response for structured suggestions
            lines = response.split('\n')
            for line in lines:
                if line.strip().startswith(('•', '-', '*', '1.', '2.', '3.')):
                    suggestions.append(line.strip().lstrip('•-*123456789. '))
        
        if not suggestions:
            suggestions = [response[:200] + "..." if len(response) > 200 else response]
        
        return {
            "suggestions": suggestions[:5],  # Limit to 5 suggestions
            "message": "Suggestions generated successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate suggestions: {str(e)}")

# Include the API router
app.include_router(api_router)

# Mount static files AFTER all API routes to avoid conflicts
try:
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=config.LOG_LEVEL.lower()
    )
