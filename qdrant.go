package main

import (
	"context"
	"crypto/sha256"
	"fmt"
	"net/url"
	"strconv"
	"time"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
)

// QdrantClient wraps the official qdrant.Client.
type QdrantClient struct {
	client *qdrant.Client
}

// NewQdrantClient parses a URL like http://host:port and connects.
func NewQdrantClient(rawURL, apiKey string) (*QdrantClient, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, fmt.Errorf("invalid qdrant URL %q: %w", rawURL, err)
	}

	host := u.Hostname()
	portStr := u.Port()
	if portStr == "" {
		portStr = "6333"
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return nil, fmt.Errorf("invalid qdrant port %q: %w", portStr, err)
	}

	cfg := &qdrant.Config{
		Host:   host,
		Port:   port,
		APIKey: apiKey,
		UseTLS: u.Scheme == "https",
	}

	client, err := qdrant.NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("qdrant connect: %w", err)
	}

	return &QdrantClient{client: client}, nil
}

// EnsureCollection creates the collection if it doesn't already exist.
func (q *QdrantClient) EnsureCollection(ctx context.Context, name string, dim uint64) error {
	exists, err := q.client.CollectionExists(ctx, name)
	if err != nil {
		return fmt.Errorf("check collection %q: %w", name, err)
	}
	if exists {
		return nil
	}

	err = q.client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: name,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     dim,
			Distance: qdrant.Distance_Cosine,
		}),
	})
	if err != nil {
		return fmt.Errorf("create collection %q: %w", name, err)
	}
	return nil
}

// CheckHashExists checks if a chunk with this content_hash already exists.
// Returns (vectorID, found, error).
func (q *QdrantClient) CheckHashExists(ctx context.Context, collection, hash string) (string, bool, error) {
	limit := uint32(1)
	resp, err := q.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: collection,
		Filter: &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewMatch("content_hash", hash),
			},
		},
		Limit:       &limit,
		WithPayload: qdrant.NewWithPayload(false),
		WithVectors: &qdrant.WithVectorsSelector{
			SelectorOptions: &qdrant.WithVectorsSelector_Enable{Enable: false},
		},
	})
	if err != nil {
		return "", false, fmt.Errorf("scroll for hash: %w", err)
	}

	if len(resp) == 0 {
		return "", false, nil
	}

	pt := resp[0]
	switch id := pt.Id.PointIdOptions.(type) {
	case *qdrant.PointId_Uuid:
		return id.Uuid, true, nil
	case *qdrant.PointId_Num:
		return strconv.FormatUint(id.Num, 10), true, nil
	}
	return "", true, nil
}

// ContentHash returns SHA256 hex of a string.
func ContentHash(s string) string {
	h := sha256.Sum256([]byte(s))
	return fmt.Sprintf("%x", h)
}

// Upsert inserts or updates a single QueueItem as a Qdrant point.
func (q *QdrantClient) Upsert(ctx context.Context, collection string, item QueueItem) error {
	payload := map[string]any{
		"content":      item.Content,
		"content_hash": item.Hash,
		"created_at":   time.Now().UTC().Format(time.RFC3339),
	}
	for k, v := range item.Payload {
		payload[k] = v
	}

	pointID := item.VectorID
	if pointID == "" {
		pointID = uuid.New().String()
	}

	_, err := q.client.Upsert(ctx, &qdrant.UpsertPoints{
		CollectionName: collection,
		Points: []*qdrant.PointStruct{
			{
				Id:      qdrant.NewID(pointID),
				Vectors: qdrant.NewVectors(item.Vector...),
				Payload: qdrant.NewValueMap(payload),
			},
		},
		Wait: boolPtr(true),
	})
	if err != nil {
		return fmt.Errorf("upsert to %q: %w", collection, err)
	}
	return nil
}

func boolPtr(b bool) *bool { return &b }
