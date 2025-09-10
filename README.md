# 📊 Accountant AI Agent - Excel Analysis System

A powerful AI-powered accounting assistant that can analyze multiple Excel spreadsheets simultaneously, perform cross-referencing, generate financial reports, and provide intelligent insights using MCP servers, LangChain, and RAG technology.

## 🚀 Features

### Core Capabilities
- **Multi-Spreadsheet Analysis**: Process multiple Excel files simultaneously
- **Cross-Reference Data**: Automatically link and compare data across different workbooks
- **Financial Analysis**: Calculate ratios, perform variance analysis, consolidate statements
- **Intelligent Chat Interface**: Natural language queries about your financial data
- **Audit Trail Generation**: Create comprehensive audit documentation
- **RAG-Powered Memory**: Long-term knowledge retention with vector database

### MCP Server Integration
- **Excel MCP Server**: Direct Excel manipulation and formula processing
- **FileSystem MCP**: Document management and storage
- **Memory MCP**: Persistent context and learning

## 📋 Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose (for containerized deployment)
- OpenAI API Key
- 8GB RAM minimum (16GB recommended for large datasets)

## 🛠️ Installation

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/mrabia/accountant-ai-agent.git
cd accountant-ai-agent
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your OpenAI API key and other configurations
```

3. **Build and run with Docker**
```bash
make build
make run
```

The application will be available at:
- Frontend: http://localhost
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Local Development Setup

1. **Install Python dependencies**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install MCP Excel Server**
```bash
npm install -g @modelcontextprotocol/server-excel
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env file
```

4. **Initialize the database**
```bash
# If using PostgreSQL locally
psql -U postgres -f init.sql
```

5. **Run the application**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 💻 Usage

### Web Interface

#### Upload Excel Files
- Drag and drop or click to upload Excel files
- Supports .xlsx, .xls, and .csv formats
- Multiple files can be uploaded for cross-analysis

#### Chat with AI Assistant
Ask natural language questions about your data:
- "What's the profit margin for Q3?"
- "Compare revenue across all uploaded sheets"
- "Generate a variance analysis between budget and actual"
- "Show me all transactions over $10,000"

#### Analysis Modes
- **Chat Mode**: Interactive Q&A about your data
- **Analysis Mode**: Deep dive into individual spreadsheets
- **Compare Mode**: Side-by-side comparison of multiple files
- **Consolidate Mode**: Merge financial statements
- **Audit Mode**: Generate audit trails and compliance reports

### API Usage

#### Upload a file
```python
import requests

with open('financial_data.xlsx', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/upload',
        files={'file': f}
    )
```

#### Analyze spreadsheets
```python
response = requests.post(
    'http://localhost:8000/api/analyze',
    json={
        'operation': 'analyze',
        'file_ids': ['file1.xlsx', 'file2.xlsx'],
        'parameters': {'key_column': 'account_id'}
    }
)
```

#### WebSocket Chat
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat');

ws.onopen = () => {
    ws.send(JSON.stringify({
        content: "What's the total revenue across all sheets?",
        context: {}
    }));
};

ws.onmessage = (event) => {
    const response = JSON.parse(event.data);
    console.log('AI Response:', response.content);
};
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Frontend (React)                   │
├─────────────────────────────────────────────────────────┤
│                    FastAPI Backend                        │
├──────────────┬──────────────┬──────────────┬────────────┤
│  LangChain   │   MCP        │    RAG       │ Accounting │
│   Agent      │  Servers     │   System     │   Tools    │
├──────────────┼──────────────┼──────────────┼────────────┤
│   OpenAI     │ Excel MCP    │   FAISS      │  Pandas    │
│   GPT-4      │ FileSystem   │  Vector DB   │  NumPy     │
└──────────────┴──────────────┴──────────────┴────────────┘
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 | Required |
| `DB_PASSWORD` | PostgreSQL password | changeme |
| `REDIS_HOST` | Redis server host | redis |
| `MCP_EXCEL_SERVER_PATH` | Path to Excel MCP server | /usr/local/lib/node_modules/@modelcontextprotocol/server-excel |
| `LOG_LEVEL` | Logging level | INFO |

### Advanced Configuration

Edit Config class in main.py:
```python
class Config:
    MODEL_NAME = "gpt-4-turbo-preview"  # or "gpt-3.5-turbo"
    TEMPERATURE = 0.1  # 0.0-1.0, lower = more focused
    MAX_TOKENS = 2000  # Maximum response length
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
```

## 📊 Example Use Cases

### 1. Monthly Financial Report
```bash
# Upload P&L statements for each month
# Ask: "Generate a monthly trend analysis for revenue and expenses"
```

### 2. Budget vs Actual Analysis
```bash
# Upload budget.xlsx and actual.xlsx
# Ask: "Show me all variances greater than 10% between budget and actual"
```

### 3. Multi-Entity Consolidation
```bash
# Upload statements from multiple subsidiaries
# Ask: "Consolidate all subsidiary financials and show group totals"
```

### 4. Audit Preparation
```bash
# Upload transaction logs
# Ask: "Create an audit trail for all transactions over $50,000 in Q4"
```

## 🧪 Testing

Run the test suite:
```bash
# Unit tests
pytest tests/unit

# Integration tests
pytest tests/integration

# Full test suite with coverage
pytest --cov=./ --cov-report=html
```

## 📈 Performance Optimization

### For Large Datasets

1. **Increase memory allocation**
```yaml
# docker-compose.yml
services:
  backend:
    mem_limit: 8g
```

2. **Enable caching**
```python
# Use Redis for operation caching
CACHE_TTL = 3600  # 1 hour
```

3. **Batch processing**
```python
# Process large files in chunks
CHUNK_SIZE = 10000
```

## 🐛 Troubleshooting

### Common Issues

#### MCP Server Connection Failed
- Ensure Node.js is installed
- Check MCP server path in .env
- Verify permissions on server executable

#### Out of Memory with Large Files
- Increase Docker memory limits
- Use chunked processing
- Consider upgrading to 16GB RAM

#### Slow Response Times
- Check OpenAI API rate limits
- Enable Redis caching
- Optimize vector database queries

## 📝 Development

### Project Structure
```
accountant-ai-agent/
├── main.py                 # Main application
├── frontend/              
│   └── index.html         # Web interface
├── mcp_servers/           
│   ├── excel_server.js    # Excel MCP implementation
│   └── memory_server.js   # Memory MCP implementation
├── tools/                 
│   ├── accounting.py      # Accounting tools
│   └── analysis.py        # Analysis utilities
├── docker-compose.yml     # Container orchestration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### Adding New Features

#### New MCP Server
```python
class CustomMCPClient:
    async def connect(self):
        # Implementation
```

#### New Accounting Tool
```python
@staticmethod
def new_analysis_method(data: pd.DataFrame):
    # Implementation
```

#### New LangChain Tool
```python
StructuredTool.from_function(
    func=custom_function,
    name="tool_name",
    description="Tool description"
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MCP Excel Server by Haris Musa
- LangChain for LLM orchestration
- OpenAI for GPT-4 API
- FAISS for vector search

## 📞 Support

- **Documentation**: docs.example.com
- **Issues**: GitHub Issues
- **Discord**: Join our community

## 🚀 Quick Start Commands

```bash
# Clone and setup
git clone <repository>
cd accountant-ai-agent
cp .env.example .env

# Edit .env with your API keys
nano .env

# Run with Docker
docker-compose up

# Or run locally
pip install -r requirements.txt
python main.py

# Access the application
open http://localhost
```

---

**Built with ❤️ for accountants and financial analysts who want to leverage AI for better insights.**
