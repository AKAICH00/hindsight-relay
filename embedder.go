package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	openai "github.com/sashabaranov/go-openai"
)

const embeddingBatchSize = 100

// Embedder wraps the OpenAI client for embedding operations.
type Embedder struct {
	client *openai.Client
}

// NewEmbedder creates a new Embedder with the given API key.
func NewEmbedder(apiKey string) *Embedder {
	return &Embedder{client: openai.NewClient(apiKey)}
}

// EmbedBatch embeds a slice of texts using text-embedding-3-small.
// Batches up to 100 texts per API call with exponential backoff on rate limits.
func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	var all [][]float32

	for i := 0; i < len(texts); i += embeddingBatchSize {
		end := i + embeddingBatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		vecs, err := e.embedWithBackoff(ctx, batch)
		if err != nil {
			return nil, fmt.Errorf("embedding batch %d: %w", i/embeddingBatchSize, err)
		}
		all = append(all, vecs...)
	}

	return all, nil
}

func (e *Embedder) embedWithBackoff(ctx context.Context, texts []string) ([][]float32, error) {
	backoffs := []time.Duration{1 * time.Second, 2 * time.Second, 4 * time.Second, 30 * time.Second}

	var lastErr error
	for attempt := 0; attempt <= len(backoffs); attempt++ {
		resp, err := e.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
			Model: openai.SmallEmbedding3,
			Input: texts,
		})
		if err == nil {
			vecs := make([][]float32, len(resp.Data))
			for i, d := range resp.Data {
				vecs[i] = d.Embedding
			}
			return vecs, nil
		}

		lastErr = err

		// Check for rate limit (429)
		var apiErr *openai.APIError
		if errors.As(err, &apiErr) && apiErr.HTTPStatusCode == http.StatusTooManyRequests {
			if attempt < len(backoffs) {
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(backoffs[attempt]):
					continue
				}
			}
		} else {
			// Non-rate-limit error — don't retry
			return nil, err
		}
	}

	return nil, fmt.Errorf("rate limit exceeded after retries: %w", lastErr)
}
