# PCB Inspection App

## AI Provider Configuration

This project supports two LLM providers via environment variables:

- `AI_PROVIDER=OPENAI` (default)
- `AI_PROVIDER=CLAUDE`

### Required environment variables

For OpenAI:

- `AI_PROVIDER=OPENAI`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4o` (optional, defaults to `gpt-4o`)

For Claude:

- `AI_PROVIDER=CLAUDE`
- `ANTHROPIC_API_KEY=...`
- `CLAUDE_MODEL=claude-3-5-sonnet-latest` (optional default)

Roboflow and Google Drive settings are still required as before.

## Additional Required Variables

- `ROBOFLOW_API_KEY=...`
- `ROBOFLOW_MODEL_ID=...`

Google Drive upload supports two modes:

1. Local development (interactive OAuth in browser):
- `client_secrets.json` file in project root

2. Cloud deployment (recommended):
- `GDRIVE_SERVICE_ACCOUNT_JSON=...` (full JSON string of service-account key)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run pcb_app.py
```

## Deploy (Render/Railway/Any Procfile Host)

This repo includes:
- `Procfile` with web start command
- `.streamlit/config.toml` for headless server mode

### Steps

1. Create a new Web Service from this GitHub repo.
2. Use build command:
```bash
pip install -r requirements.txt
```
3. Start command:
```bash
streamlit run pcb_app.py --server.address=0.0.0.0 --server.port=$PORT
```
4. Add environment variables:
- `AI_PROVIDER`
- `OPENAI_API_KEY` and optional `OPENAI_MODEL` (or Claude vars)
- `ROBOFLOW_API_KEY`
- `ROBOFLOW_MODEL_ID`
- `GDRIVE_SERVICE_ACCOUNT_JSON`
5. Redeploy.

## Deploy (Streamlit Community Cloud)

1. New app -> select this repo and `pcb_app.py`.
2. In app Settings -> Secrets, add the same environment variables as above.
3. Deploy.
