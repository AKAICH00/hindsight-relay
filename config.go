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
	defaultQdrantURL = "http://localhost:6333"
	serviceVersion   = "0.1.0"
)

type AgentCollections struct {
	Memories string `json:"memories"`
	Facts    string `json:"facts"`
	Episodes string `json:"episodes"`
}

type Config struct {
	OpenAIAPIKey  string
	QdrantURL     string
	QdrantAPIKey  string
	Port          string
	LogLevel      string
	AgentMap      map[string]AgentCollections
	WatchState    string
	ServiceName   string
	ServiceVer    string
	QueueMaxDepth int
}

func LoadConfig() (Config, error) {
	cfg := Config{
		OpenAIAPIKey:  strings.TrimSpace(os.Getenv("OPENAI_API_KEY")),
		QdrantURL:     getEnv("QDRANT_URL", defaultQdrantURL),
		QdrantAPIKey:  strings.TrimSpace(os.Getenv("QDRANT_API_KEY")),
		Port:          getEnv("PORT", defaultPort),
		LogLevel:      strings.ToLower(getEnv("LOG_LEVEL", "info")),
		WatchState:    "watched.json",
		ServiceName:   "hindsight-relay",
		ServiceVer:    serviceVersion,
		QueueMaxDepth: 1000,
	}

	if cfg.OpenAIAPIKey == "" {
		return Config{}, fmt.Errorf("OPENAI_API_KEY is required")
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

