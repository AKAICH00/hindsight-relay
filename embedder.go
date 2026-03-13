package main

import (
	"context"
	"fmt"
	"time"

	"github.com/google/generative-ai-go/genai"
	openai "github.com/sashabaranov/go-openai"
	"google.golang.org/api/option"
)

const embeddingBatchSize = 100

// EmbedderInterface defines the contract for embedding operations.
type EmbedderInterface interface {
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, error)
}

// OpenAIEmbedder wraps the OpenAI client for embedding operations.
type OpenAIEmbedder struct {
	client *openai.Client
	model  openai.EmbeddingModel
}

func NewOpenAIEmbedder(apiKey, model string) *OpenAIEmbedder {
	em := openai.LargeEmbedding3 // default
	switch model {
	case "text-embedding-3-large":
		em = openai.LargeEmbedding3
	case "text-embedding-3-small":
		em = openai.SmallEmbedding3
	case "text-embedding-ada-002":
		em = openai.AdaEmbeddingV2
	}
	return &OpenAIEmbedder{client: openai.NewClient(apiKey), model: em}
}

func (e *OpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	resp, err := e.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: e.model,
		Input: texts,
	})
	if err != nil {
		return nil, err
	}
	vecs := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		vecs[i] = d.Embedding
	}
	return vecs, nil
}

// GeminiEmbedder wraps the Google Generative AI client.
type GeminiEmbedder struct {
	client *genai.Client
	model  string
}

func NewGeminiEmbedder(ctx context.Context, apiKey, model string) (*GeminiEmbedder, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, err
	}
	return &GeminiEmbedder{client: client, model: model}, nil
}

func (e *GeminiEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	em := e.client.EmbeddingModel(e.model)
	batch := em.NewBatch()
	for _, t := range texts {
		batch.AddContent(genai.Text(t))
	}

	resp, err := em.BatchEmbedContents(ctx, batch)
	if err != nil {
		return nil, err
	}

	vecs := make([][]float32, len(resp.Embeddings))
	for i, emb := range resp.Embeddings {
		vecs[i] = emb.Values
	}
	return vecs, nil
}

// MultiEmbedder manages batching and retries across any implementation.
type MultiEmbedder struct {
	impl EmbedderInterface
}

func NewMultiEmbedder(impl EmbedderInterface) *MultiEmbedder {
	return &MultiEmbedder{impl: impl}
}

func (e *MultiEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
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

func (e *MultiEmbedder) embedWithBackoff(ctx context.Context, texts []string) ([][]float32, error) {
	backoffs := []time.Duration{1 * time.Second, 2 * time.Second, 4 * time.Second, 10 * time.Second}

	var lastErr error
	for attempt := 0; attempt <= len(backoffs); attempt++ {
		vecs, err := e.impl.EmbedBatch(ctx, texts)
		if err == nil {
			return vecs, nil
		}

		lastErr = err
		if attempt < len(backoffs) {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(backoffs[attempt]):
				continue
			}
		}
	}

	return nil, fmt.Errorf("embedding failed after retries: %w", lastErr)
}
