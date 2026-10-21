# Hy Memory Plugins

Integration code for agent and editor platforms. Memory capabilities come from the Python SDK in the repository root (`hy_memory/`, PyPI: `hy-memory`); the plugin layer adapts it to the host framework (HTTP, hooks, MCP and so on).

## Layout

| Directory | Description | Published as |
|------|------|------|
| `openclaw/` | OpenClaw memory plugin (TypeScript, starts the Python `hy_memory.server`) | npm: `openclaw-hy-memory` |
| `hook/` | hook integration for Claude Code and similar (reserved) | -- |
| `mcp/` | MCP Server integration (reserved) | -- |

Difference from [Mem0](https://github.com/mem0ai/mem0): the Mem0 OpenClaw plugin depends on the **TypeScript SDK** (`mem0ai` npm); the plugins in this repository depend on the **Python SDK** and talk to it over a local HTTP service.

## Development

```bash
# OpenClaw plugin
cd plugins/openclaw
npm install
npm run build
```

Publish to npmjs:

```bash
bash scripts/publish_npmjs.sh <version>
```
