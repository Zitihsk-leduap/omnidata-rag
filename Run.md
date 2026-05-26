# Running the Nepali RAG Assistant

This guide covers all steps to run the project in development and production environments.

## Prerequisites

- Docker (v20.10+) and Docker Compose (v2.0+)
- Python 3.12 (for local development)
- 8GB+ RAM (for embedding models)
- Internet connection (for Google Gemini API)
- Google API Key: Get from aistudio.google.com (https://aistudio.google.com/app/apikey)

### Check Prerequisites
```bash
# Verify Docker
docker --version          # Should show v20.10+
docker compose version    # Should show v2.0+

# Verify Python (for local dev)
python3 --version         # Should show v3.12+
```

---

## Environment Setup

### Step 1: Create Environment File

```bash
# From project root
cp .env.example .env
```

### Step 2: Add Your Google Gemini API Key

Edit `.env` and replace the placeholder:
```bash
GEMINI_API_KEY=your_actual_api_key_here
```

To get your API key:
1. Visit https://aistudio.google.com/app/apikey
2. Click "Get API Key"
3. Create a new API key or copy existing one
4. Paste into `.env`

### Step 3: Verify .env Configuration

```bash
# Contents of .env should look like:
BACKEND_URL=http://backend:8080
GEMINI_API_KEY=AIza...xxxxx
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

**SECURITY NOTE**: Never commit `.env` to Git. It's already in `.gitignore`.

---

## Initialize Vector Database (First Time Only)

The system requires a populated Chroma vector database before starting.

### For Docker Deployment

```bash
# Build the backend image
docker compose -f Docker/docker-compose.yml build backend

# Run the database initialization
docker compose -f Docker/docker-compose.yml run --rm backend python -m AI.load

# This will:
# 1. Read AI/Data/company_act_ne.txt
# 2. Parse and chunk the document
# 3. Generate embeddings
# 4. Store in AI/chroma/ directory
# 5. Exit when complete
```

### For Local Development

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Load the database
python -m AI.load

You'll see output like:
# CHROMA PATH: /path/to/omnidata-rag/AI/chroma
# Loading multilingual reranker...
# Reranker loaded.
# Processed X documents, Y chunks created
```

NOTE: First load may take 5-10 minutes (downloading embedding models).

---

## Running with Docker (Recommended)

### Quick Start (One Command)

```bash
# From project root
docker compose -f Docker/docker-compose.yml up
```

### Access the Application

After containers start:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8080
- API Documentation: http://localhost:8080/docs
- Health Check: http://localhost:8080/health

### Running in Background

```bash
# Start services in detached mode
docker compose -f Docker/docker-compose.yml up -d

# View logs
docker compose -f Docker/docker-compose.yml logs -f frontend    # Frontend logs
docker compose -f Docker/docker-compose.yml logs -f backend     # Backend logs

# Stop services
docker compose -f Docker/docker-compose.yml down
```

### View Container Status

```bash
docker compose -f Docker/docker-compose.yml ps

# Output should show:
# NAME        STATUS
# backend     Up X minutes
# frontend    Up X minutes
```

---

## Local Development (Without Docker)

### Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Load database
python -m AI.load

# Verify environment variables are set
echo $GEMINI_API_KEY  # Should not be empty
```

### Run Backend

```bash
# Terminal 1
cd /path/to/omnidata-rag
source venv/bin/activate

uvicorn backend.main:app --reload --port 8080

# Output:
# INFO:     Uvicorn running on http://127.0.0.1:8080
# INFO:     Application startup complete
```

### Run Frontend

```bash
# Terminal 2
cd /path/to/omnidata-rag
source venv/bin/activate

streamlit run frontend/app_new.py

# Output:
# You can now view your Streamlit app in your browser.
# URL: http://localhost:8501
```

---

## Testing the System

### Test Backend Health

```bash
# Check if backend is running
curl http://localhost:8080/health

# Expected response:
# {"status":"ok"}
```

### Test Chat Endpoint

```bash
# Try a query
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is the Nepal Company Act?"}'

# Expected response:
# {"reply":"The Nepal Company Act 2063 is...","status":"success"}
```

### Test Frontend Connection

1. Open http://localhost:8501
2. Type a question in the text area
3. Click "🔍 Ask"
4. Should see response in 10-30 seconds

---

## Advanced Configuration

### Modify Backend Ports

Edit `Docker/docker-compose.yml`:
```yaml
backend:
  ports:
    - "9000:8080"  # Change to http://localhost:9000
```

Then update `BACKEND_URL` in `.env`:
```bash
BACKEND_URL=http://backend:9000
```

### Modify Frontend Port

Edit `Docker/docker-compose.yml`:
```yaml
frontend:
  ports:
    - "9501:8501"  # Change to http://localhost:9501
```

### Enable Persistent Chroma Storage

The current setup stores data in containers. For persistence:

Edit `Docker/docker-compose.yml` to add:
```yaml
volumes:
  chroma_data:
    driver: local

services:
  backend:
    volumes:
      - ../AI:/app/AI
      - ../backend:/app/backend
      - chroma_data:/app/AI/chroma  # Add this line
```

---

## Troubleshooting

### Issue: "Connection refused on localhost:8080"

**Cause**: Backend not running or not ready

**Solution**:
```bash
# Check if container is running
docker compose -f Docker/docker-compose.yml ps

# View backend logs
docker compose -f Docker/docker-compose.yml logs backend

# Restart backend
docker compose -f Docker/docker-compose.yml restart backend
```

### Issue: "GEMINI_API_KEY not set"

**Cause**: Environment variable missing

**Solution**:
```bash
# Verify .env exists
cat .env | grep GEMINI_API_KEY

# If empty, update:
nano .env  # Edit and add your key
```

### Issue: "Chroma database not found"

**Cause**: Database initialization skipped

**Solution**:
```bash
# Run initialization
docker compose -f Docker/docker-compose.yml run --rm backend python -m AI.load

# Verify it exists
ls -la AI/chroma/
```

### Issue: "Frontend can't reach backend"

**Cause**: Wrong BACKEND_URL in .env

**Solution**:
```bash
# For Docker, this MUST be:
BACKEND_URL=http://backend:8080

# NOT http://localhost:8080 (that's for local dev)

# Restart containers
docker compose -f Docker/docker-compose.yml down
docker compose -f Docker/docker-compose.yml up
```

### Issue: "Out of memory" or slow responses

**Cause**: Insufficient RAM for embedding models

**Solution**:
- Close other applications
- Allocate more RAM to Docker: Settings → Resources → Memory
- Use CPU-only mode (slower but uses less RAM)

### Issue: "Models downloading very slowly"

**Cause**: First run downloads 2GB+ of models

**Solution**:
```bash
This is normal, wait 5-10 minutes. Monitor with:
watch -n 5 'du -sh AI/chroma/'

# Progress will appear in logs
docker compose -f Docker/docker-compose.yml logs -f backend
```

---

## Monitoring

### View System Health

```bash
# Docker resource usage
docker stats

# Specific container resource usage
docker compose -f Docker/docker-compose.yml stats
```

### Check Logs

```bash
# All services
docker compose -f Docker/docker-compose.yml logs

# Last 50 lines
docker compose -f Docker/docker-compose.yml logs --tail=50

# Follow in real-time
docker compose -f Docker/docker-compose.yml logs -f

# Specific service
docker compose -f Docker/docker-compose.yml logs frontend
```

---

## Stopping Services

```bash
# Graceful shutdown
docker compose -f Docker/docker-compose.yml down

# Remove volumes (clears data)
docker compose -f Docker/docker-compose.yml down -v

# Force stop
docker compose -f Docker/docker-compose.yml kill
```

---

## Additional Resources

- [API Documentation](http://localhost:8080/docs) - Interactive API explorer
- [README.md](README.md) - Project overview
- [Requirements](requirements.txt) - Python dependencies
- Google Gemini Docs: https://ai.google.dev

---

## Startup Checklist

Before reporting issues, verify:

- Docker & Docker Compose installed
- GEMINI_API_KEY in `.env`
- Vector database initialized (`AI/chroma/` exists)
- No port conflicts (8080, 8501 free)
- Sufficient RAM (8GB+)
- Internet connection (for API)
- Firewall allows localhost connections

---

## Need Help?

Check logs first:
```bash
docker compose -f Docker/docker-compose.yml logs --tail=100
```

Then refer to the troubleshooting section above.

