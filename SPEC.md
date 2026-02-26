# hindsight-relay — Build Spec

## What
A tiny Go Fiber HTTP server that streams text/file events into Qdrant vector embeddings in real-time.
Replaces the 30-min Python cron sync with sub-second embedding on every write/message.

## Binary name
`hindsight-relay`

## Port
`7358`

## Stack
- Go 1.22+
- `github.com/gofiber/fiber/v2`
- `github.com/qdrant/go-client` (Qdrant gRPC client)
- `github.com/fsnotify/fsnotify` (file watching)
- `github.com/sashabaranov/go-openai` (OpenAI embeddings)
- Standard library for everything else

## Endpoints

### POST /embed
Embed a raw text chunk immediately.
```json
{
  "agent_id": "main",
  "text": "Aksel decided to use Vaultwarden for secrets",
  "source": "message",
  "role": "assistant",
  "timestamp": 1234567890
}
```
Response: `{"ok": true, "vector_id": "uuid", "collection": "ezer_memories"}`

### POST /embed/file
Read a file from disk, chunk it, embed all chunks.
```json
{
  "agent_id": "main",
  "path": "/Users/botbot/.openclaw/workspace/memory/2026-02-26.md"
}
```
Response: `{"ok": true, "chunks": 12, "collection": "ezer_memories"}`

### POST /embed/batch
Bulk embed from a directory (replaces sync.py).
```json
{
  "agent_id": "main",
  "dir": "/Users/botbot/.openclaw/workspace/memory"
}
```
Response: `{"ok": true, "files": 42, "chunks": 318, "skipped": 5}`

### GET /health
Response: `{"ok": true, "service": "hindsight-relay", "version": "0.1.0", "queue_depth": 0}`

### GET /stats
Response: `{"embeddings_total": 1234, "embeddings_per_min": 12, "last_sync": "2026-02-26T...", "watched_dirs": 3, "queue_depth": 0}`

### POST /watch
Add a directory to the file watcher.
```json
{
  "agent_id": "main",
  "dir": "/Users/botbot/.openclaw/workspace/memory"
}
```
Response: `{"ok": true, "watching": ["/Users/botbot/.openclaw/workspace/memory"]}`

## Agent → Qdrant Collection Mapping
Read from env var `HINDSIGHT_AGENT_MAP` (JSON):
```json
{
  "main": {
    "memories": "ezer_memories",
    "facts": "ezer_facts",
    "episodes": "ezer_episodes"
  },
  "vault": {
    "memories": "vault_memories",
    "facts": "vault_facts",
    "episodes": "vault_episodes"
  }
}
```
Default collection if agent not in map: `{agent_id}_memories`

## File Watcher Behavior
- Use fsnotify to watch all dirs registered via POST /watch
- On any Write/Create event to a `.md` file: debounce 200ms, then POST /embed/file
- On rename/delete: log but don't panic
- Watched dirs survive restarts via a local `watched.json` state file

## Chunking Strategy
- Split on `\n\n` (double newline = paragraph)
- Max chunk size: 800 tokens (estimate: 4 chars/token → 3200 chars)
- Overlap: 100 tokens (400 chars) between chunks
- Skip chunks < 20 chars
- Include file path + heading context in each chunk's payload

## Embedding Model
- `text-embedding-3-small` (1536 dims) — matches existing Hindsight setup
- Batch up to 100 chunks per OpenAI API call

## Deduplication
- SHA256 hash of chunk text → stored as payload field `content_hash`
- Before inserting: check if hash exists in collection (Qdrant scroll + filter)
- If exists and content identical: skip (return existing vector_id)
- If exists but content changed: delete old, insert new

## Qdrant Payload Schema (per vector)
```json
{
  "content": "actual text chunk",
  "content_hash": "sha256hex",
  "source": "message|file|batch",
  "agent_id": "main",
  "file_path": "/path/to/file.md",
  "heading": "## Section title",
  "role": "user|assistant|system",
  "timestamp": 1234567890,
  "created_at": "2026-02-26T10:00:00Z"
}
```

## Environment Variables
```
OPENAI_API_KEY        required
QDRANT_URL            default: http://localhost:6333
QDRANT_API_KEY        optional (for remote Qdrant)
PORT                  default: 7358
HINDSIGHT_AGENT_MAP   JSON string (see above)
LOG_LEVEL             debug|info|warn|error (default: info)
```

## Startup behavior
1. Validate OPENAI_API_KEY present
2. Connect to Qdrant (fail fast if unreachable)
3. Load watched.json (resume watched dirs from last run)
4. Start fsnotify watcher for all saved dirs
5. Start Fiber HTTP server on PORT
6. Log: "hindsight-relay ready on :7358, watching N dirs"

## Error handling
- OpenAI rate limits: exponential backoff (1s, 2s, 4s, max 30s)
- Qdrant unavailable: queue events in memory (max 1000), retry every 5s
- Invalid agent_id: use default collection, log warning
- File not found: 404 response, don't crash

## Graceful shutdown
- On SIGTERM/SIGINT: drain queue, save watched.json, exit 0

## Project structure
```
hindsight-relay/
├── main.go           # Fiber app, routes, startup
├── embedder.go       # OpenAI embedding calls + batching
├── qdrant.go         # Qdrant client wrapper (upsert, dedup check)
├── chunker.go        # Text chunking logic
├── watcher.go        # fsnotify file watcher
├── config.go         # Env vars, agent map
├── queue.go          # In-memory retry queue
├── go.mod
└── go.sum
```

## Build
```bash
go build -o hindsight-relay .
```
Binary: ~8MB, no dependencies needed at runtime.

## Test
After building, run:
```bash
OPENAI_API_KEY=xxx QDRANT_URL=http://localhost:6333 ./hindsight-relay &

# Health
curl http://localhost:7358/health

# Embed text
curl -X POST http://localhost:7358/embed \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"main","text":"Test embedding for Ezer memory","source":"test"}'

# Watch a dir
curl -X POST http://localhost:7358/watch \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"main","dir":"/Users/botbot/.openclaw/workspace/memory"}'

# Stats
curl http://localhost:7358/stats
```

## Success criteria
1. `go build` produces a working binary
2. All 6 endpoints respond correctly
3. Embedding a text chunk returns a valid vector_id in Qdrant
4. File watcher detects a .md write within 200ms and embeds it
5. Duplicate text returns same vector_id (dedup works)
6. `GET /stats` shows correct counts
