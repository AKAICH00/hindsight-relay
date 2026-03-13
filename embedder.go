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

// Usage tracks token consumption across different modalities.
type Usage struct {
	TotalTokens    int64   `json:"total_tokens"`
	TextTokens     int64   `json:"text_tokens"`
	ImageCount     int     `json:"image_count"`
	VideoSeconds   float64 `json:"video_seconds"`
	EstimatedCost  float64 `json:"estimated_cost"`
}

// EmbedderInterface defines the contract for embedding operations.
type EmbedderInterface interface {
	EmbedBatch(ctx context.Context, texts []string) ([][]float32, Usage, error)
	CountTokens(ctx context.Context, texts []string) (int64, error)
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

func (e *OpenAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, Usage, error) {
	var usage Usage
	if len(texts) == 0 {
		return nil, usage, nil
	}
	resp, err := e.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: e.model,
		Input: texts,
	})
	if err != nil {
		return nil, usage, err
	}
	vecs := make([][]float32, len(resp.Data))
	for i, d := range resp.Data {
		vecs[i] = d.Embedding
	}
	
	usage.TotalTokens = int64(resp.Usage.TotalTokens)
	usage.TextTokens = usage.TotalTokens
	// text-embedding-3-large is $0.13 / 1M tokens
	usage.EstimatedCost = float64(usage.TotalTokens) * 0.00000013
	
	return vecs, usage, nil
}

func (e *OpenAIEmbedder) CountTokens(ctx context.Context, texts []string) (int64, error) {
	// OpenAI doesn't have a direct count_tokens API for embeddings in the SDK,
	// but we can estimate 1 token per 4 chars for monitoring purposes.
	totalChars := 0
	for _, t := range texts {
		totalChars += len(t)
	}
	return int64(totalChars / 4), nil
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

func (e *GeminiEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, Usage, error) {
	var usage Usage
	if len(texts) == 0 {
		return nil, usage, nil
	}

	em := e.client.EmbeddingModel(e.model)
	batch := em.NewBatch()
	for _, t := range texts {
		batch.AddContent(genai.Text(t))
	}

	resp, err := em.BatchEmbedContents(ctx, batch)
	if err != nil {
		return nil, usage, err
	}

	vecs := make([][]float32, len(resp.Embeddings))
	for i, emb := range resp.Embeddings {
		vecs[i] = emb.Values
	}
	
	// Gemini Embedding 2 metadata (if available in the specific response)
	// For now we count tokens via the API
	tokens, _ := e.CountTokens(ctx, texts)
	usage.TotalTokens = tokens
	usage.TextTokens = tokens
	// Gemini Embedding 2 text is $0.01 / 1M tokens ($0.10 for text-embedding-004)
	usage.EstimatedCost = float64(tokens) * 0.00000001
	
	return vecs, usage, nil
}

func (e *GeminiEmbedder) CountTokens(ctx context.Context, texts []string) (int64, error) {
	em := e.client.GenerativeModel("gemini-1.5-flash") // Use flash for token counting
	total := int64(0)
	for _, t := range texts {
		resp, err := em.CountTokens(ctx, genai.Text(t))
		if err == nil {
			total += int64(resp.TotalTokens)
		}
	}
	return total, nil
}

// MultiEmbedder manages batching and retries across any implementation.
type MultiEmbedder struct {
	impl EmbedderInterface
}

func NewMultiEmbedder(impl EmbedderInterface) *MultiEmbedder {
	return &MultiEmbedder{impl: impl}
}

func (e *MultiEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, Usage, error) {
	var totalUsage Usage
	if len(texts) == 0 {
		return nil, totalUsage, nil
	}

	var all [][]float32
	for i := 0; i < len(texts); i += embeddingBatchSize {
		end := i + embeddingBatchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[i:end]

		vecs, usage, err := e.embedWithBackoff(ctx, batch)
		if err != nil {
			return nil, totalUsage, fmt.Errorf("embedding batch %d: %w", i/embeddingBatchSize, err)
		}
		all = append(all, vecs...)
		totalUsage.TotalTokens += usage.TotalTokens
		totalUsage.TextTokens += usage.TextTokens
		totalUsage.EstimatedCost += usage.EstimatedCost
	}

	return all, totalUsage, nil
}

func (e *MultiEmbedder) embedWithBackoff(ctx context.Context, texts []string) ([][]float32, Usage, error) {
	backoffs := []time.Duration{1 * time.Second, 2 * time.Second, 4 * time.Second, 10 * time.Second}

	var lastErr error
	for attempt := 0; attempt <= len(backoffs); attempt++ {
		vecs, usage, err := e.impl.EmbedBatch(ctx, texts)
		if err == nil {
			return vecs, usage, nil
		}

		lastErr = err
		if attempt < len(backoffs) {
			select {
			case <-ctx.Done():
				return nil, Usage{}, ctx.Err()
			case <-time.After(backoffs[attempt]):
				continue
			}
		}
	}

	return nil, Usage{}, fmt.Errorf("embedding failed after retries: %w", lastErr)
}

func (e *MultiEmbedder) CountTokens(ctx context.Context, texts []string) (int64, error) {
	return e.impl.CountTokens(ctx, texts)
}
