package main

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
)

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

// IdentityCore fields are immutable after creation. Changing them = new contact.
type IdentityCore struct {
	UUID           string `json:"uuid"`
	Name           string `json:"name"`
	PrimaryChannel string `json:"primary_channel"` // e.g. "whatsapp:+521234567890"
	CreatedAt      string `json:"created_at"`
	IdentityHash   string `json:"identity_hash"` // SHA256(uuid|name|primary_channel)
}

// Contact is the full contact record.
type Contact struct {
	IdentityCore

	// Channels — lookup index. primary_channel is locked; others are addable.
	Channels map[string]string `json:"channels"` // "whatsapp" → "+521234567890"

	// Mutable context
	Role      string   `json:"role"`
	Security  string   `json:"security"` // owner|admin|team|vendor|customer|unknown
	Language  string   `json:"language"`
	Formality string   `json:"formality"` // casual|formal|usted
	Topics    []string `json:"topics"`
	Notes     string   `json:"notes"`
	Agent     string   `json:"agent"` // which agent owns this contact

	// Behavioral stats (auto-updated)
	MessageCount      int     `json:"message_count"`
	BaselineReady     bool    `json:"baseline_ready"` // true when >= 10 messages
	LastSeen          string  `json:"last_seen,omitempty"`
	AvgDivergence     float64 `json:"avg_divergence"`
	DivergenceAlerts  int     `json:"divergence_alerts"`
	ConsecutiveAlerts int     `json:"consecutive_alerts"` // resets on non-alert
}

// ---------------------------------------------------------------------------
// Identity hash
// ---------------------------------------------------------------------------

func computeIdentityHash(id, name, primaryChannel string) string {
	raw := id + "|" + name + "|" + primaryChannel
	h := sha256.Sum256([]byte(raw))
	return fmt.Sprintf("%x", h)
}

func (c *Contact) VerifyHash() (ok bool, stored, computed string) {
	computed = computeIdentityHash(c.UUID, c.Name, c.PrimaryChannel)
	return computed == c.IdentityHash, c.IdentityHash, computed
}

// ---------------------------------------------------------------------------
// Index — channel → UUID fast lookup (contacts.json)
// ---------------------------------------------------------------------------

type ContactIndex struct {
	mu      sync.RWMutex
	index   map[string]string // "whatsapp:+52..." → uuid
	path    string
}

func NewContactIndex(path string) *ContactIndex {
	return &ContactIndex{
		index: make(map[string]string),
		path:  path,
	}
}

func (ci *ContactIndex) Load() error {
	ci.mu.Lock()
	defer ci.mu.Unlock()

	data, err := os.ReadFile(ci.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	return json.Unmarshal(data, &ci.index)
}

func (ci *ContactIndex) Save() error {
	ci.mu.RLock()
	defer ci.mu.RUnlock()

	data, err := json.MarshalIndent(ci.index, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(ci.path, data, 0o644)
}

func (ci *ContactIndex) Set(channelKey, contactUUID string) {
	ci.mu.Lock()
	defer ci.mu.Unlock()
	ci.index[channelKey] = contactUUID
	ci.index["uuid:"+contactUUID] = contactUUID
}

func (ci *ContactIndex) Lookup(channelKey string) (string, bool) {
	ci.mu.RLock()
	defer ci.mu.RUnlock()
	v, ok := ci.index[channelKey]
	return v, ok
}

// channelKey builds a lookup key from type + id, e.g. "whatsapp:+521234567890"
func channelKey(channelType, id string) string {
	return strings.ToLower(channelType) + ":" + strings.TrimSpace(id)
}

// ---------------------------------------------------------------------------
// ContactStore — Qdrant-backed registry
// ---------------------------------------------------------------------------

const (
	contactsCollection = "lattice_contacts"
	contactsDim        = uint64(1536)
	baselineMinMsgs    = 10
)

type ContactStore struct {
	qc      *QdrantClient
	emb     *Embedder
	index   *ContactIndex
	agentID string
}

func NewContactStore(qc *QdrantClient, emb *Embedder, agentID, indexPath string) (*ContactStore, error) {
	idx := NewContactIndex(indexPath)
	if err := idx.Load(); err != nil {
		return nil, fmt.Errorf("load contact index: %w", err)
	}
	return &ContactStore{qc: qc, emb: emb, index: idx, agentID: agentID}, nil
}

func (cs *ContactStore) EnsureCollection(ctx context.Context) error {
	return cs.qc.EnsureCollection(ctx, contactsCollection, contactsDim)
}

// Register creates a new contact. Returns error if primary channel already registered.
func (cs *ContactStore) Register(ctx context.Context, name, primaryType, primaryID string, extra map[string]string, opts ContactOpts) (*Contact, error) {
	pkey := channelKey(primaryType, primaryID)

	// Dedup check
	if existing, ok := cs.index.Lookup(pkey); ok {
		return nil, fmt.Errorf("contact already registered with this channel (uuid: %s)", existing)
	}

	id := uuid.New().String()
	primary := pkey
	hash := computeIdentityHash(id, name, primary)
	now := time.Now().UTC().Format(time.RFC3339)

	channels := map[string]string{primaryType: primaryID}
	for k, v := range extra {
		channels[k] = v
	}

	c := &Contact{
		IdentityCore: IdentityCore{
			UUID:           id,
			Name:           name,
			PrimaryChannel: primary,
			CreatedAt:      now,
			IdentityHash:   hash,
		},
		Channels:  channels,
		Role:      opts.Role,
		Security:  opts.Security,
		Language:  opts.Language,
		Formality: opts.Formality,
		Topics:    opts.Topics,
		Notes:     opts.Notes,
		Agent:     cs.agentID,
	}
	if c.Security == "" {
		c.Security = "unknown"
	}

	if err := cs.upsertToQdrant(ctx, c); err != nil {
		return nil, fmt.Errorf("qdrant upsert: %w", err)
	}

	// Update index for all channels
	cs.index.Set(pkey, id)
	for t, pid := range channels {
		cs.index.Set(channelKey(t, pid), id)
	}
	if err := cs.index.Save(); err != nil {
		return nil, fmt.Errorf("save index: %w", err)
	}

	return c, nil
}

// Lookup finds a contact by any channel key. Returns nil if not found.
func (cs *ContactStore) Lookup(ctx context.Context, channelType, id string) (*Contact, error) {
	uuid, ok := cs.index.Lookup(channelKey(channelType, id))
	if !ok {
		return nil, nil
	}
	return cs.GetByUUID(ctx, uuid)
}

// GetByUUID fetches a contact directly by UUID from Qdrant.
func (cs *ContactStore) GetByUUID(ctx context.Context, contactUUID string) (*Contact, error) {
	limit := uint32(1)
	results, err := cs.qc.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: contactsCollection,
		Filter: &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewMatch("uuid", contactUUID),
			},
		},
		Limit:       &limit,
		WithPayload: qdrant.NewWithPayload(true),
		WithVectors: &qdrant.WithVectorsSelector{
			SelectorOptions: &qdrant.WithVectorsSelector_Enable{Enable: false},
		},
	})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, nil
	}
	return contactFromPayload(results[0].Payload)
}

// UpdateMutable updates only mutable fields. Rejects name/primary_channel changes.
func (cs *ContactStore) UpdateMutable(ctx context.Context, contactUUID string, patch ContactOpts) (*Contact, error) {
	c, err := cs.GetByUUID(ctx, contactUUID)
	if err != nil || c == nil {
		return nil, fmt.Errorf("contact not found: %s", contactUUID)
	}

	if patch.Role != "" {
		c.Role = patch.Role
	}
	if patch.Security != "" {
		c.Security = patch.Security
	}
	if patch.Language != "" {
		c.Language = patch.Language
	}
	if patch.Formality != "" {
		c.Formality = patch.Formality
	}
	if len(patch.Topics) > 0 {
		c.Topics = patch.Topics
	}
	if patch.Notes != "" {
		c.Notes = patch.Notes
	}

	if err := cs.upsertToQdrant(ctx, c); err != nil {
		return nil, err
	}
	return c, nil
}

// RecordMessage increments message_count, updates last_seen and baseline_ready.
func (cs *ContactStore) RecordMessage(ctx context.Context, c *Contact) error {
	c.MessageCount++
	c.LastSeen = time.Now().UTC().Format(time.RFC3339)
	if c.MessageCount >= baselineMinMsgs {
		c.BaselineReady = true
	}
	return cs.upsertToQdrant(ctx, c)
}

// RecordDivergence updates divergence stats on the contact record.
func (cs *ContactStore) RecordDivergence(ctx context.Context, c *Contact, score float64, alerted bool) error {
	// Rolling average (simple)
	if c.MessageCount > 1 {
		c.AvgDivergence = (c.AvgDivergence*float64(c.MessageCount-1) + score) / float64(c.MessageCount)
	} else {
		c.AvgDivergence = score
	}
	if alerted {
		c.DivergenceAlerts++
		c.ConsecutiveAlerts++
	} else {
		c.ConsecutiveAlerts = 0
	}
	return cs.upsertToQdrant(ctx, c)
}

// ---------------------------------------------------------------------------
// Qdrant serialization
// ---------------------------------------------------------------------------

func (cs *ContactStore) upsertToQdrant(ctx context.Context, c *Contact) error {
	// Embed the contact's searchable description
	description := fmt.Sprintf("%s %s %s %s %s",
		c.Name, c.Role, c.Notes, c.Language, strings.Join(c.Topics, " "))
	vecs, err := cs.emb.EmbedBatch(ctx, []string{description})
	if err != nil {
		return fmt.Errorf("embed contact: %w", err)
	}

	topicsJSON, _ := json.Marshal(c.Topics)
	channelsJSON, _ := json.Marshal(c.Channels)

	payload := map[string]any{
		"uuid":               c.UUID,
		"name":               c.Name,
		"primary_channel":    c.PrimaryChannel,
		"created_at":         c.CreatedAt,
		"identity_hash":      c.IdentityHash,
		"channels":           string(channelsJSON),
		"role":               c.Role,
		"security":           c.Security,
		"language":           c.Language,
		"formality":          c.Formality,
		"topics":             string(topicsJSON),
		"notes":              c.Notes,
		"agent":              c.Agent,
		"message_count":      int64(c.MessageCount),
		"baseline_ready":     c.BaselineReady,
		"last_seen":          c.LastSeen,
		"avg_divergence":     c.AvgDivergence,
		"divergence_alerts":  int64(c.DivergenceAlerts),
		"consecutive_alerts": int64(c.ConsecutiveAlerts),
	}

	item := QueueItem{
		VectorID:   c.UUID,
		Vector:     vecs[0],
		Content:    description,
		Hash:       ContentHash(c.UUID),
		Collection: contactsCollection,
		Payload:    payload,
	}
	return cs.qc.Upsert(ctx, contactsCollection, item)
}

func contactFromPayload(p map[string]*qdrant.Value) (*Contact, error) {
	// Convert Qdrant payload to JSON then unmarshal
	raw := make(map[string]any)
	for k, v := range p {
		switch x := v.Kind.(type) {
		case *qdrant.Value_StringValue:
			raw[k] = x.StringValue
		case *qdrant.Value_IntegerValue:
			raw[k] = x.IntegerValue
		case *qdrant.Value_DoubleValue:
			raw[k] = x.DoubleValue
		case *qdrant.Value_BoolValue:
			raw[k] = x.BoolValue
		}
	}

	data, err := json.Marshal(raw)
	if err != nil {
		return nil, err
	}

	// Intermediate struct for JSON fields stored as strings
	var intermediate struct {
		UUID               string  `json:"uuid"`
		Name               string  `json:"name"`
		PrimaryChannel     string  `json:"primary_channel"`
		CreatedAt          string  `json:"created_at"`
		IdentityHash       string  `json:"identity_hash"`
		Channels           string  `json:"channels"`
		Role               string  `json:"role"`
		Security           string  `json:"security"`
		Language           string  `json:"language"`
		Formality          string  `json:"formality"`
		Topics             string  `json:"topics"`
		Notes              string  `json:"notes"`
		Agent              string  `json:"agent"`
		MessageCount       int64   `json:"message_count"`
		BaselineReady      bool    `json:"baseline_ready"`
		LastSeen           string  `json:"last_seen"`
		AvgDivergence      float64 `json:"avg_divergence"`
		DivergenceAlerts   int64   `json:"divergence_alerts"`
		ConsecutiveAlerts  int64   `json:"consecutive_alerts"`
	}
	if err := json.Unmarshal(data, &intermediate); err != nil {
		return nil, err
	}

	var channels map[string]string
	_ = json.Unmarshal([]byte(intermediate.Channels), &channels)
	var topics []string
	_ = json.Unmarshal([]byte(intermediate.Topics), &topics)

	return &Contact{
		IdentityCore: IdentityCore{
			UUID:           intermediate.UUID,
			Name:           intermediate.Name,
			PrimaryChannel: intermediate.PrimaryChannel,
			CreatedAt:      intermediate.CreatedAt,
			IdentityHash:   intermediate.IdentityHash,
		},
		Channels:          channels,
		Role:              intermediate.Role,
		Security:          intermediate.Security,
		Language:          intermediate.Language,
		Formality:         intermediate.Formality,
		Topics:            topics,
		Notes:             intermediate.Notes,
		Agent:             intermediate.Agent,
		MessageCount:      int(intermediate.MessageCount),
		BaselineReady:     intermediate.BaselineReady,
		LastSeen:          intermediate.LastSeen,
		AvgDivergence:     intermediate.AvgDivergence,
		DivergenceAlerts:  int(intermediate.DivergenceAlerts),
		ConsecutiveAlerts: int(intermediate.ConsecutiveAlerts),
	}, nil
}

// ---------------------------------------------------------------------------
// ContactOpts — mutable fields for Register/Update
// ---------------------------------------------------------------------------

type ContactOpts struct {
	Role      string
	Security  string
	Language  string
	Formality string
	Topics    []string
	Notes     string
}
