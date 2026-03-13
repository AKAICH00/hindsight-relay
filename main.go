package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
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
	embedder  *MultiEmbedder
	queue     *RetryQueue
	watcher   *Watcher
	contacts  *ContactStore
	diverge   *DivergenceEngine

	mu              sync.Mutex
	embeddingsTotal int64
	tokensTotal     int64
	costTotal       float64
	alertTriggered  bool
	lastSync        time.Time
	startTime       time.Time
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------

func (a *App) checkAlerts() {
	if a.cfg.AlertWebhookURL == "" || a.cfg.AlertThresholdUSD <= 0 {
		return
	}

	a.mu.Lock()
	cost := a.costTotal
	triggered := a.alertTriggered
	a.mu.Unlock()

	if cost >= a.cfg.AlertThresholdUSD && !triggered {
		log.Printf("[alert] Cost threshold exceeded: $%.6f (limit: $%.6f)", cost, a.cfg.AlertThresholdUSD)
		
		// Send async webhook
		go func() {
			payload := map[string]any{
				"text": fmt.Sprintf("⚠️ *Gemini Cost Alert*: Threshold of $%.2f reached. Current spend: $%.6f.", 
					a.cfg.AlertThresholdUSD, cost),
			}
			data, _ := json.Marshal(payload)
			http.Post(a.cfg.AlertWebhookURL, "application/json", strings.NewReader(string(data)))
		}()

		a.mu.Lock()
		a.alertTriggered = true
		a.mu.Unlock()
	}
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

	// Ensure collection exists with configured dimensions
	if err := a.qdrant.EnsureCollection(ctx, collection, a.cfg.EmbeddingDim); err != nil {
		return nil, fmt.Errorf("ensure collection: %w", err)
	}

	vecs, usage, err := a.embedder.EmbedBatch(ctx, chunks)
	if err != nil {
		return nil, fmt.Errorf("embed: %w", err)
	}

	a.mu.Lock()
	a.tokensTotal += usage.TotalTokens
	a.costTotal += usage.EstimatedCost
	a.mu.Unlock()

	a.checkAlerts()

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
const dashboardHTML = `
<!DOCTYPE html>
<html>
<head>
    <title>Gemini Builder Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: 'Inter', system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #1e293b; padding: 20px; border-radius: 12px; border: 1px solid #334155; }
        .stat { font-size: 2.5rem; font-weight: bold; color: #38bdf8; }
        .label { color: #94a3b8; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; margin-bottom: 10px; }
        .gauge-container { height: 20px; background: #334155; border-radius: 10px; overflow: hidden; margin-top: 20px; }
        .gauge-fill { height: 100%; background: linear-gradient(90deg, #38bdf8, #818cf8); transition: width 0.5s ease; }
        h1 { margin-bottom: 30px; font-weight: 300; }
    </style>
</head>
<body>
    <h1>Gemini 2026 : Builder Dashboard</h1>
    <div class="grid">
        <div class="card">
            <div class="label">Total Spend (USD)</div>
            <div id="cost" class="stat">$0.000000</div>
            <div class="gauge-container"><div id="costGauge" class="gauge-fill" style="width: 0%"></div></div>
            <div id="thresholdLabel" style="font-size: 0.7rem; color: #64748b; margin-top: 5px;"></div>
        </div>
        <div class="card">
            <div class="label">Total Tokens</div>
            <div id="tokens" class="stat">0</div>
        </div>
        <div class="card">
            <div class="label">Velocity (Embeds/Min)</div>
            <div id="velocity" class="stat">0.0</div>
        </div>
        <div class="card" style="grid-column: span 2;">
            <div class="label">Live Embedding Stream</div>
            <canvas id="chart" height="100"></canvas>
        </div>
    </div>

    <script>
        const ctx = document.getElementById('chart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Embeddings', data: [], borderColor: '#38bdf8', tension: 0.4, fill: true, backgroundColor: 'rgba(56, 189, 248, 0.1)' }] },
            options: { scales: { y: { beginAtZero: true, grid: { color: '#334155' } }, x: { grid: { display: false } } }, plugins: { legend: { display: false } } }
        });

        async function update() {
            const [statsResp, usageResp] = await Promise.all([fetch('/stats'), fetch('/usage')]);
            const stats = await statsResp.json();
            const usage = await usageResp.json();

            document.getElementById('cost').innerText = '$' + usage.estimated_cost_usd;
            document.getElementById('tokens').innerText = usage.total_tokens.toLocaleString();
            document.getElementById('velocity').innerText = stats.embeddings_per_min.toFixed(1);

            // Update Gauge
            const threshold = parseFloat(window.threshold || 0);
            if (threshold > 0) {
                const percent = Math.min((parseFloat(usage.estimated_cost_usd) / threshold) * 100, 100);
                document.getElementById('costGauge').style.width = percent + '%';
                document.getElementById('thresholdLabel').innerText = 'Limit: $' + threshold.toFixed(2);
                if (percent > 90) document.getElementById('costGauge').style.background = '#f43f5e';
            }

            // Update Chart
            const now = new Date().toLocaleTimeString();
            chart.data.labels.push(now);
            chart.data.datasets[0].data.push(stats.embeddings_total);
            if (chart.data.labels.length > 20) { chart.data.labels.shift(); chart.data.datasets[0].data.shift(); }
            chart.update();
        }

        window.threshold = {{.Threshold}};
        update();
        setInterval(update, 5000);
    </script>
</body>
</html>
`

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

func (a *App) setupRoutes(fib *fiber.App) {
	// GET /dashboard
	fib.Get("/dashboard", func(c *fiber.Ctx) error {
		c.Set("Content-Type", "text/html")
		html := strings.ReplaceAll(dashboardHTML, "{{.Threshold}}", fmt.Sprintf("%.6f", a.cfg.AlertThresholdUSD))
		return c.SendString(html)
	})

	// GET /health
	fib.Get("/health", func(c *fiber.Ctx) error {
		return c.JSON(fiber.Map{
			"ok":          true,
			"service":     a.cfg.ServiceName,
			"version":     a.cfg.ServiceVer,
			"queue_depth": a.queue.Depth(),
		})
	})

	// GET /usage
	fib.Get("/usage", func(c *fiber.Ctx) error {
		a.mu.Lock()
		tokens := a.tokensTotal
		cost := a.costTotal
		a.mu.Unlock()

		return c.JSON(fiber.Map{
			"ok":               true,
			"provider":         a.cfg.EmbeddingProvider,
			"model":            a.cfg.EmbeddingModel,
			"total_tokens":     tokens,
			"estimated_cost_usd": fmt.Sprintf("%.6f", cost),
		})
	})

	// POST /count-tokens
	fib.Post("/count-tokens", func(c *fiber.Ctx) error {
		var req struct {
			Texts []string `json:"texts"`
		}
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		
		tokens, err := a.embedder.CountTokens(c.Context(), req.Texts)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		
		// Estimate cost
		cost := float64(tokens) * 0.00000001 // Default Gemini rate
		if a.cfg.EmbeddingProvider == "openai" {
			cost = float64(tokens) * 0.00000013
		}

		return c.JSON(fiber.Map{
			"ok":               true,
			"token_count":      tokens,
			"estimated_cost_usd": fmt.Sprintf("%.6f", cost),
		})
	})

	// GET /stats
	fib.Get("/stats", func(c *fiber.Ctx) error {
		total := atomic.LoadInt64(&a.embeddingsTotal)
		a.mu.Lock()
		tokens := a.tokensTotal
		cost := a.costTotal
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
			"tokens_total":       tokens,
			"estimated_cost_usd": fmt.Sprintf("%.6f", cost),
			"embeddings_per_min": perMin,
			"last_sync":          lastSyncStr,
			"watched_dirs":       a.watcher.Count(),
			"queue_depth":        a.queue.Depth(),
		})
	})

	// POST /embed
	fib.Post("/embed", func(c *fiber.Ctx) error {
		var req struct {
			AgentID     string `json:"agent_id"`
			Text        string `json:"text"`
			Source      string `json:"source"`
			Role        string `json:"role"`
			Timestamp   int64  `json:"timestamp"`
			ContactUUID string `json:"contact_uuid"`
		}
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		if req.Text == "" {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": "text is required"})
		}

		chunks := ChunkText(req.Text)
		heading := ExtractHeading(req.Text)

		// If contact_uuid provided, resolve and verify identity
		var contact *Contact
		var divResult *DivergenceResult
		if req.ContactUUID != "" && a.contacts != nil {
			ct, err := a.contacts.GetByUUID(c.Context(), req.ContactUUID)
			if err == nil && ct != nil {
				ok, stored, computed := ct.VerifyHash()
				if !ok {
					return c.Status(400).JSON(fiber.Map{
						"ok":    false,
						"error": "identity hash mismatch — possible tampering",
						"stored_hash": stored, "computed_hash": computed,
					})
				}
				contact = ct
			}
		}

		ids, err := a.embedAndUpsert(c.Context(), req.AgentID, chunks, req.Source, "", heading, req.Role, req.Timestamp)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}

		// Divergence scoring
		if contact != nil && a.diverge != nil && len(ids) > 0 {
			// Re-embed the text to get its vector for divergence scoring
			vecs, usage, embedErr := a.embedder.EmbedBatch(c.Context(), []string{req.Text[:min(len(req.Text), 800)]})
			if embedErr == nil && len(vecs) > 0 {
				a.mu.Lock()
				a.tokensTotal += usage.TotalTokens
				a.costTotal += usage.EstimatedCost
				a.mu.Unlock()

				collection := a.cfg.CollectionForAgent(req.AgentID, nil)
				divResult, _ = a.diverge.Score(c.Context(), collection, req.ContactUUID, contact, vecs[0])
			}
			// Update contact stats
			_ = a.contacts.RecordMessage(c.Context(), contact)
			if divResult != nil {
				_ = a.contacts.RecordDivergence(c.Context(), contact, divResult.Score, divResult.Alert)
			}
		}

		vectorID := ""
		if len(ids) > 0 {
			vectorID = ids[0]
		}
		collection := a.cfg.CollectionForAgent(req.AgentID, nil)

		resp := fiber.Map{"ok": true, "vector_id": vectorID, "collection": collection}
		if req.ContactUUID != "" {
			resp["contact_uuid"] = req.ContactUUID
		}
		if divResult != nil {
			resp["divergence"] = divResult
		}
		return c.JSON(resp)
	})

	// ---------------------------------------------------------------------------
	// Contact routes
	// ---------------------------------------------------------------------------

	// POST /contacts/register
	fib.Post("/contacts/register", func(c *fiber.Ctx) error {
		if a.contacts == nil {
			return c.Status(503).JSON(fiber.Map{"ok": false, "error": "contact store not initialized"})
		}
		var req struct {
			Name          string            `json:"name"`
			PrimaryType   string            `json:"primary_type"`   // e.g. "whatsapp"
			PrimaryID     string            `json:"primary_id"`     // e.g. "+521234567890"
			ExtraChannels map[string]string `json:"extra_channels"` // optional
			Role          string            `json:"role"`
			Security      string            `json:"security"`
			Language      string            `json:"language"`
			Formality     string            `json:"formality"`
			Topics        []string          `json:"topics"`
			Notes         string            `json:"notes"`
		}
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		if req.Name == "" || req.PrimaryType == "" || req.PrimaryID == "" {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": "name, primary_type, and primary_id are required"})
		}
		contact, err := a.contacts.Register(c.Context(), req.Name, req.PrimaryType, req.PrimaryID,
			req.ExtraChannels, ContactOpts{
				Role: req.Role, Security: req.Security, Language: req.Language,
				Formality: req.Formality, Topics: req.Topics, Notes: req.Notes,
			})
		if err != nil {
			return c.Status(409).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		return c.JSON(fiber.Map{"ok": true, "contact": contact})
	})

	// GET /contacts/lookup?type=whatsapp&id=+521234567890
	fib.Get("/contacts/lookup", func(c *fiber.Ctx) error {
		if a.contacts == nil {
			return c.Status(503).JSON(fiber.Map{"ok": false, "error": "contact store not initialized"})
		}
		channelType := c.Query("type")
		id := c.Query("id")
		uuid := c.Query("uuid")

		var contact *Contact
		var err error
		if uuid != "" {
			contact, err = a.contacts.GetByUUID(c.Context(), uuid)
		} else if channelType != "" && id != "" {
			contact, err = a.contacts.Lookup(c.Context(), channelType, id)
		} else {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": "provide uuid or type+id"})
		}
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		if contact == nil {
			return c.Status(404).JSON(fiber.Map{"ok": false, "error": "contact not found"})
		}
		return c.JSON(fiber.Map{"ok": true, "contact": contact})
	})

	// POST /contacts/verify
	fib.Post("/contacts/verify", func(c *fiber.Ctx) error {
		if a.contacts == nil {
			return c.Status(503).JSON(fiber.Map{"ok": false, "error": "contact store not initialized"})
		}
		var req struct {
			UUID string `json:"uuid"`
		}
		if err := c.BodyParser(&req); err != nil {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		contact, err := a.contacts.GetByUUID(c.Context(), req.UUID)
		if err != nil || contact == nil {
			return c.Status(404).JSON(fiber.Map{"ok": false, "error": "contact not found"})
		}
		ok, stored, computed := contact.VerifyHash()
		return c.JSON(fiber.Map{
			"ok": true, "tampered": !ok,
			"stored_hash": stored, "computed_hash": computed,
			"contact_uuid": contact.UUID, "name": contact.Name,
		})
	})

	// PATCH /contacts/:uuid
	fib.Patch("/contacts/:uuid", func(c *fiber.Ctx) error {
		if a.contacts == nil {
			return c.Status(503).JSON(fiber.Map{"ok": false, "error": "contact store not initialized"})
		}
		var patch ContactOpts
		if err := c.BodyParser(&patch); err != nil {
			return c.Status(400).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		contact, err := a.contacts.UpdateMutable(c.Context(), c.Params("uuid"), patch)
		if err != nil {
			return c.Status(500).JSON(fiber.Map{"ok": false, "error": err.Error()})
		}
		return c.JSON(fiber.Map{"ok": true, "contact": contact})
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

	fmt.Printf("[startup] provider=%s model=%s dim=%d agent=%s\n", cfg.EmbeddingProvider, cfg.EmbeddingModel, cfg.EmbeddingDim, cfg.AgentID)

	qc, err := NewQdrantClient(cfg.QdrantURL, cfg.QdrantAPIKey)
	if err != nil {
		log.Fatalf("qdrant connect: %v", err)
	}

	var embedImpl EmbedderInterface
	if cfg.EmbeddingProvider == "gemini" {
		gi, err := NewGeminiEmbedder(context.Background(), cfg.GoogleAPIKey, cfg.EmbeddingModel)
		if err != nil {
			log.Fatalf("gemini init: %v", err)
		}
		embedImpl = gi
	} else {
		embedImpl = NewOpenAIEmbedder(cfg.OpenAIAPIKey, cfg.EmbeddingModel)
	}

	app := &App{
		cfg:       cfg,
		qdrant:    qc,
		embedder:  NewMultiEmbedder(embedImpl),
		queue:     NewRetryQueue(cfg.QueueMaxDepth),
		diverge:   NewDivergenceEngine(qc),
		startTime: time.Now(),
	}

	// Contact store — init in background with retries so startup is never blocked
	go func() {
		backoffs := []time.Duration{2, 5, 10, 30, 60}
		for i := 0; ; i++ {
			cs, err := NewContactStore(qc, app.embedder, cfg.AgentID, cfg.ContactsIndex, cfg.EmbeddingDim)
			if err != nil {
				delay := backoffs[min(i, len(backoffs)-1)] * time.Second
				log.Printf("[contacts] init failed (%v), retrying in %s", err, delay)
				time.Sleep(delay)
				continue
			}
			ctx0 := context.Background()
			if err := cs.EnsureCollection(ctx0); err != nil {
				delay := backoffs[min(i, len(backoffs)-1)] * time.Second
				log.Printf("[contacts] collection init failed (%v), retrying in %s", err, delay)
				time.Sleep(delay)
				continue
			}
			app.mu.Lock()
			app.contacts = cs
			app.mu.Unlock()
			log.Printf("[contacts] ready (agent: %s)", cfg.AgentID)
			return
		}
	}()

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
