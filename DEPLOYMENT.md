# Deployment Guide

## Prerequisites

- **Azure resources:**
  - Azure OpenAI: `text-embedding-3-large` + `gpt-4.1` deployments
  - Azure AI Search (Basic tier or above for semantic ranking)
  - Azure App Service (Linux, Python 3.12)
- **Local:** Python 3.12+, Azure CLI (`az`)

## Environment Variables

Set in `.env` locally and as App Settings on the Web App:

```env
ADO_ORG=your-org
ADO_PROJECT=your-project
ADO_WIKI=your-wiki
ADO_PAT=your-ado-pat

AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_EMBED_MODEL=text-embedding-3-large
AZURE_OPENAI_CHAT_MODEL=gpt-4.1

SEARCH_SERVICE_NAME=your-search-service
SEARCH_SERVICE_KEY=your-search-key
INDEX_NAME=ado_wiki
```

**Important App Settings for Azure:**
- `ENVIRONMENT=local` — reads config from App Settings (not Key Vault)
- `WEBSITES_CONTAINER_START_TIME_LIMIT=600` — allows time for pip install on cold start
- `SCM_DO_BUILD_DURING_DEPLOYMENT=false` — startup.sh handles pip install

## 1. Ingest Locally

```powershell
pip install -r requirements.txt
python ingest.py
```

- Fetches all wiki pages, chunks at 800 tokens with 200 overlap
- Embeds with `text-embedding-3-large` (3072 dimensions)
- Upserts to Azure AI Search with delta detection
- For fresh start: delete `ingestion_state.json` and the Azure Search index

## 2. Configure Web App

```powershell
# Set App Settings (run once)
az webapp config appsettings set -g <rg> -n <app> --settings `
  ADO_ORG=... ADO_PROJECT=... ADO_WIKI=... ADO_PAT=... `
  AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_KEY=... `
  AZURE_OPENAI_EMBED_MODEL=text-embedding-3-large `
  AZURE_OPENAI_CHAT_MODEL=gpt-4.1 `
  SEARCH_SERVICE_NAME=... SEARCH_SERVICE_KEY=... `
  INDEX_NAME=ado_wiki ENVIRONMENT=local `
  WEBSITES_CONTAINER_START_TIME_LIMIT=600

# Set startup command
az webapp config set -g <rg> -n <app> --startup-file "bash startup.sh"
```

## 3. Deploy

**Recommended (one command):**
```powershell
cd wikiagent
.\deploy.ps1
```

**Manual:**
```powershell
Compress-Archive -Path streamlit_app.py,query.py,azure_search_client.py,utils.py,ingest.py,startup.sh,requirements.txt,runtime.txt -DestinationPath app.zip -Force
az webapp deploy -g <rg> -n <app> --src-path app.zip --type zip
```

## 4. Validate

- Open `https://<app>.azurewebsites.net` (allow 3-5 min for first cold start)
- Ask a question in the chat UI
- Check logs: `az webapp log tail -g <rg> -n <app>`

## Redeploy After Code Changes

```powershell
cd wikiagent
.\deploy.ps1
```

The deploy script: fixes startup.sh line endings → builds zip → deploys → monitors until Running.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Application Error page | Wait 3-5 min (cold start); check `WEBSITES_CONTAINER_START_TIME_LIMIT=600` |
| `$'\r'` in logs | `deploy.ps1` auto-fixes; or manually convert startup.sh to LF |
| Vector dimension mismatch | Delete index in Azure Portal → re-run `python ingest.py` |
| Wrong embedding model in logs | Set `ENVIRONMENT=local` to bypass Key Vault; verify App Settings |
| `query_language` error | Already fixed in code; ensure latest deploy |
| No search results | Verify index has documents: check ingestion logs |
