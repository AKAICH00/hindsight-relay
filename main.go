package main

import (
	"context"
	"crypto/sha256"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

// ---------------------------------------------------------------------------
// App-level state
// ---------------------------------------------------------------------------

type App struct {
	cfg       Config
	qdrant    *QdrantClient
	embedder  *Embedder
	queue     *RetryQueue
	watcher   *Watcher

	mu              sync.Mutex
	embeddingsTotal int64
	lastSync        time.Time
	startTime       time.Time
}

// ---------------------------------------------------------------------------
// Embed + upsert helper
// ---------------------------------------------------------------------------

func (a *App) embedAndUpsert(
	ctx context.Context,
	agentID string,
	chunks []string,
	source, filePath, heading, role string,
	ts int64,
) ([]string, error) {
	if len(chunks) == 0 {
		return nil, nil
	}

	collection := a.cfg.CollectionForAgent(agentID, func(f string, args ...any) {
		log.Printf("[warn] "+f, args...)
	})

	// Ensure collection exists (1536 dims for text-embedding-3-small)
	if err := a.qdrant.EnsureCollection(ctx, collection, 1536); err != nil {
		return nil, fmt.Errorf("ensure collection: %w", err)
	}

	vecs, err := a.embedder.EmbedBatch(ctx, chunks)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	var ids []string
	for i, chunk := range chunks {
		hash := fmt.Sprintf("%x", sha256.Sum256([]byte(chunk)))

		// Dedup check
		existingID, found, err := a.qdrant.CheckHashExists(ctx, collection, hash)
		if err == nil && found {
			ids = append(ids, existingID)
			continue
		}

		vectorID := uuid.New().String()
		item := QueueItem{
			Collection: collection,
			VectorID:   vectorID,
			Vector:     vecs[i],
			Content:    chunk,
			Hash:       hash,
			Payload: map[string]any{
				"source":    source,
				"agent_id":  agentID,
				"file_path": filePath,
				"heading":   heading,
				"role":      role,
				"timestamp": ts,
			},
		}

		if err := a.qdrant.Upsert(ctx, collection, item); err != nil {
			log.Printf("[warn] upsert failed, queuing: %v", err)
			a.queue.Enqueue(item)
		} else {
			atomic.AddInt64(&a.embeddingsTotal, 1)
			a.mu.Lock()
			a.lastSync = time.Now()
			a.mu.Unlock()
		}

		ids = append(ids, vectorID)
	}

	return ids, nil
}

// ---------------------------------------------------------------------------
// Retry worker
// ---------------------------------------------------------------------------

func (a *App) retryWorker(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			// Drain on shutdown
			a.drainQueue(context.Background())
			return
		case <-ticker.C:
			a.drainQueue(ctx)
		}
	}
}

func (a *App) drainQueue(ctx context.Context) {
	items := a.queue.PopAll()
	if len(items) == 0 {
		return
	}

	var failed []QueueItem
	for _, item := range items {
		if err := a.qdrant.Upsert(ctx, item.Collection, item); err != nil {
			log.Printf("[retry] upsert failed again, re-queuing: %v", err)
			failed = append(failed, item)
		} else {
			atomic.AddInt64(&a.embeddingsTotal, 1)
		}
	}

	if len(failed) > 0 {
		a.queue.PushFront(failed)
	}
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

func (a *App) setupRoutes(fib *fiber.App) {
	// GET /health
	fib.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"ok":          true,
			"service":     a.cfg.ServiceName,
			"version":     a.cfg.ServiceVer,
			"queue_depth": a.queue.Depth(),
		})
	})

	// GET /stats
	fib.Get("/stats", func(c *fiber.Ctx) error {
		total := atomic.LoadInt64(&a.embeddingsTotal)
		a.mu.Lock()
		lastSync := a.lastSync
		a.mu.Unlock()

		elapsed := time.Since(a.startTime).Minutes()
		var perMin float64
		if elapsed > 0 {
			perMin = float64(total) / elapsed
		}

		lastSyncStr := ""
		if !lastSync.IsZero() {
			lastSyncStr = lastSync.UTC().Format(time.RFC3339)
		}

		return c.JSON(fiber.Map{
			"embeddings_total":   total,
			"embeddings_per_min": perMin,
			"last_sync":          lastSyncStr,
			"watched_dirs":       a.watcher.Count(),
			"queue_depth":        a.queue.Depth(),
		})
	})

	// POST /embed
	fib.Post("/embed", func(c *fiber.Ctx) error {
		var req struct {
			AgentID   string `json:"agent_id"`
			Text      string `json:"text"`
			Source    string `json:"source"`
			Role      string `json:"role"`
			Timestamp int64  `json:"timestamp"`
		}
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		if req.Text == "" {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": "text is required"})
		}

		chunks := ChunkText(req.Text)
		heading := ExtractHeading(req.Text)
		ids, err := a.embedAndUpsert(c.Context(), req.AgentID, chunks, req.Source, "", heading, req.Role, req.Timestamp)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}

		vectorID := ""
		if len(ids) > 0 {
			vectorID = ids[0]
		}
		collection := a.cfg.CollectionForAgent(req.AgentID, nil)
		return c.JSON(fiber.Map{"ok": true, "vector_id": vectorID, "collection": collection})
	})

	// POST /embed/file
	fib.Post("/embed/file", func(c *fiber.Ctx) error {
		var req struct {
			AgentID string `json:"agent_id"`
			Path    string `json:"path"`
		}
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		if req.Path == "" {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": "path is required"})
		}

		data, err := os.ReadFile(req.Path)
		if err != nil {
			return c.Status(404).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}

		text := string(data)
		chunks := ChunkText(text)
		heading := ExtractHeading(text)
		ids, err := a.embedAndUpsert(c.Context(), req.AgentID, chunks, "file", req.Path, heading, "", 0)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}

		collection := a.cfg.CollectionForAgent(req.AgentID, nil)
		return c.JSON(fiber.Map{"ok": true, "chunks": len(ids), "collection": collection})
	})

	// POST /embed/batch
	fib.Post("/embed/batch", func(c *fiber.Ctx) error {
		var req struct {
			AgentID string `json:"agent_id"`
			Dir     string `json:"dir"`
		}
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		if req.Dir == "" {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": "dir is required"})
		}

		files := 0
		totalChunks := 0
		skipped := 0

		err := filepath.Walk(req.Dir, func(path string, info os.FileInfo, err error) error {
			if err != nil || info.IsDir() || !isMarkdown(path) {
				return nil
			}
			data, err := os.ReadFile(path)
			if err != nil {
				skipped++
				return nil
			}
			text := string(data)
			chunks := ChunkText(text)
			heading := ExtractHeading(text)
			ids, err := a.embedAndUpsert(c.Context(), req.AgentID, chunks, "batch", path, heading, "", 0)
			if err != nil {
				log.Printf("[batch] skip %s: %v", path, err)
				skipped++
				return nil
			}
			files++
			totalChunks += len(ids)
			return nil
		})
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}

		return c.JSON(fiber.Map{"ok": true, "files": files, "chunks": totalChunks, "skipped": skipped})
	})

	// POST /watch
	fib.Post("/watch", func(c *fiber.Ctx) error {
		var req struct {
			AgentID string `json:"agent_id"`
			Dir     string `json:"dir"`
		}
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		if req.Dir == "" {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": "dir is required"})
		}

		if err := a.watcher.AddDir(req.Dir); err != nil {
			return c.Status(500).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		if err := a.watcher.SaveState(a.cfg.WatchState); err != nil {
			log.Printf("[watch] save state failed: %v", err)
		}

		return c.JSON(fiber.Map{"ok": true, "watching": a.watcher.Dirs()})
	})
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

func main() {
	cfg, err := LoadConfig()
	if err != nil {
		log.Fatalf("config error: %v", err)
	}

	qc, err := NewQdrantClient(cfg.QdrantURL, cfg.QdrantAPIKey)
	if err != nil {
		log.Fatalf("qdrant connect: %v", err)
	}

	app := &App{
		cfg:       cfg,
		qdrant:    qc,
		embedder:  NewEmbedder(cfg.OpenAIAPIKey),
		queue:     NewRetryQueue(cfg.QueueMaxDepth),
		startTime: time.Now(),
	}

	// Watcher
	watcher, err := NewWatcher(func(path string) {
		log.Printf("[watcher] file changed: %s", path)
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		data, err := os.ReadFile(path)
		if err != nil {
			log.Printf("[watcher] read error %s: %v", path, err)
			return
		}
		text := string(data)
		chunks := ChunkText(text)
		heading := ExtractHeading(text)
		// Use "main" as default agent for file watcher events
		ids, err := app.embedAndUpsert(ctx, "main", chunks, "file", path, heading, "", 0)
		if err != nil {
			log.Printf("[watcher] embed error %s: %v", path, err)
			return
		}
		log.Printf("[watcher] embedded %d chunks from %s", len(ids), path)
	})
	if err != nil {
		log.Fatalf("watcher init: %v", err)
	}
	app.watcher = watcher

	// Restore watched dirs from last run
	if err := watcher.LoadState(cfg.WatchState); err != nil {
		log.Printf("[startup] load watch state: %v", err)
	}

	// Context for background workers
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	// Start file watcher in background
	go watcher.Start(ctx)

	// Start retry worker
	go app.retryWorker(ctx)

	// Fiber app
	fib := fiber.New(fiber.Config{
		DisableStartupMessage: false,
	})
	app.setupRoutes(fib)

	// Shutdown on context cancel
	go func() {
		<-ctx.Done()
		log.Println("[shutdown] draining queue...")
		app.drainQueue(context.Background())
		if err := watcher.SaveState(cfg.WatchState); err != nil {
			log.Printf("[shutdown] save watch state: %v", err)
		}
		log.Println("[shutdown] stopping server...")
		_ = fib.Shutdown()
	}()

	log.Printf("[startup] %s v%s ready on :%s, watching %d dirs",
		cfg.ServiceName, cfg.ServiceVer, cfg.Port, watcher.Count())

	if err := fib.Listen(":" + cfg.Port); err != nil {
		log.Printf("[server] stopped: %v", err)
	}
}
