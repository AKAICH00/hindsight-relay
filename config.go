package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

const (
	defaultPort      = "7358"
	defaultQdrantURL = "http://localhost:6334"
	serviceVersion   = "0.1.0"
)

type AgentCollections struct {
	Memories string `json:"memories"`
	Facts    string `json:"facts"`
	Episodes string `json:"episodes"`
}

type Config struct {
	OpenAIAPIKey   string
	GoogleAPIKey   string
	EmbeddingProvider string // "openai" or "gemini" (default: "openai")
	QdrantURL      string
	QdrantAPIKey   string
	Port           string
	LogLevel       string
	AgentMap       map[string]AgentCollections
	WatchState     string
	ServiceName    string
	ServiceVer     string
	QueueMaxDepth  int
	AgentID        string // which agent this relay instance serves
	ContactsIndex  string // path to contacts.json fast-lookup index
	EmbeddingModel string // model name (default: text-embedding-3-large for openai, text-embedding-004 for gemini)
	EmbeddingDim   uint64 // vector dimensions (derived from model)
}

func LoadConfig() (Config, error) {
	cfg := Config{
		OpenAIAPIKey:      strings.TrimSpace(os.Getenv("OPENAI_API_KEY")),
		GoogleAPIKey:      strings.TrimSpace(os.Getenv("GOOGLE_API_KEY")),
		EmbeddingProvider: strings.ToLower(getEnv("EMBEDDING_PROVIDER", "openai")),
		QdrantURL:         getEnv("QDRANT_URL", defaultQdrantURL),
		QdrantAPIKey:      strings.TrimSpace(os.Getenv("QDRANT_API_KEY")),
		Port:              getEnv("PORT", defaultPort),
		LogLevel:          strings.ToLower(getEnv("LOG_LEVEL", "info")),
		WatchState:        getEnv("WATCH_STATE", "watched.json"),
		ServiceName:       "hindsight-relay",
		ServiceVer:        serviceVersion,
		QueueMaxDepth:     1000,
		AgentID:           getEnv("AGENT_ID", "main"),
		ContactsIndex:     getEnv("CONTACTS_INDEX", "contacts.json"),
	}

	// Default models based on provider
	if cfg.EmbeddingProvider == "gemini" {
		cfg.EmbeddingModel = getEnv("EMBEDDING_MODEL", "text-embedding-004")
	} else {
		cfg.EmbeddingModel = getEnv("EMBEDDING_MODEL", "text-embedding-3-large")
	}

	// Derive dimension from model
	switch cfg.EmbeddingModel {
	case "text-embedding-3-large":
		cfg.EmbeddingDim = 3072
	case "text-embedding-3-small", "text-embedding-ada-002":
		cfg.EmbeddingDim = 1536
	case "text-embedding-004", "gemini-embedding-2-preview":
		cfg.EmbeddingDim = 768 // Gemini 004 is 768, Gemini 2 is scalable, defaulting to 768 for efficiency
	default:
		cfg.EmbeddingDim = 768
	}

	if cfg.EmbeddingProvider == "openai" && cfg.OpenAIAPIKey == "" {
		return Config{}, fmt.Errorf("OPENAI_API_KEY is required for openai provider")
	}
	if cfg.EmbeddingProvider == "gemini" && cfg.GoogleAPIKey == "" {
		return Config{}, fmt.Errorf("GOOGLE_API_KEY is required for gemini provider")
	}

	if _, err := strconv.Atoi(cfg.Port); err != nil {
		return Config{}, fmt.Errorf("PORT must be numeric: %w", err)
	}

	agentMapRaw := strings.TrimSpace(os.Getenv("HINDSIGHT_AGENT_MAP"))
	if agentMapRaw != "" {
		if err := json.Unmarshal([]byte(agentMapRaw), &cfg.AgentMap); err != nil {
			return Config{}, fmt.Errorf("invalid HINDSIGHT_AGENT_MAP JSON: %w", err)
		}
	} else {
		cfg.AgentMap = map[string]AgentCollections{}
	}

	return cfg, nil
}

func getEnv(key, fallback string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return fallback
	}
	return v
}

func (c Config) CollectionForAgent(agentID string, warnf func(format string, args ...any)) string {
	agentID = strings.TrimSpace(agentID)
	if agentID == "" {
		if warnf != nil {
			warnf("invalid agent_id: empty, using default collection")
		}
		return "default_memories"
	}
	if mapped, ok := c.AgentMap[agentID]; ok && strings.TrimSpace(mapped.Memories) != "" {
		return mapped.Memories
	}
	if warnf != nil {
		warnf("agent_id %q not found in map, using default collection", agentID)
	}
	return fmt.Sprintf("%s_memories", agentID)
}
