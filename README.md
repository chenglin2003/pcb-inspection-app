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
