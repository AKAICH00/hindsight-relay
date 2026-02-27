package main

import (
	"context"
	"fmt"
	"math"

	"github.com/qdrant/go-client/qdrant"
)

// ---------------------------------------------------------------------------
// Divergence tiers
// ---------------------------------------------------------------------------

type DivergenceTier string

const (
	TierNormal  DivergenceTier = "normal"   // 0.00 – 0.35
	TierDrift   DivergenceTier = "drift"    // 0.35 – 0.55
	TierUnusual DivergenceTier = "unusual"  // 0.55 – 0.70
	TierAnomaly DivergenceTier = "anomaly"  // 0.70 – 1.00
)

type DivergenceResult struct {
	Score         float64        `json:"score"`
	Tier          DivergenceTier `json:"tier"`
	Alert         bool           `json:"alert"`
	TrendAlert    bool           `json:"trend_alert"`   // 3+ consecutive alerts
	BaselineMsgs  int            `json:"baseline_msgs"`
	Reason        string         `json:"reason"`
}

func scoreTier(score float64) DivergenceTier {
	switch {
	case score < 0.35:
		return TierNormal
	case score < 0.55:
		return TierDrift
	case score < 0.70:
		return TierUnusual
	default:
		return TierAnomaly
	}
}

// ---------------------------------------------------------------------------
// Centroid + cosine
// ---------------------------------------------------------------------------

// cosineSimilarity computes similarity between two equal-length vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		ai, bi := float64(a[i]), float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}

// centroid computes the mean vector of a slice.
func centroid(vecs [][]float32) []float32 {
	if len(vecs) == 0 {
		return nil
	}
	dim := len(vecs[0])
	out := make([]float32, dim)
	for _, v := range vecs {
		for i, x := range v {
			out[i] += x
		}
	}
	n := float32(len(vecs))
	for i := range out {
		out[i] /= n
	}
	return out
}

// ---------------------------------------------------------------------------
// DivergenceEngine
// ---------------------------------------------------------------------------

type DivergenceEngine struct {
	qc          *QdrantClient
	baselineN   int // how many recent messages to pull for baseline
}

func NewDivergenceEngine(qc *QdrantClient) *DivergenceEngine {
	return &DivergenceEngine{qc: qc, baselineN: 20}
}

// Score computes the divergence of newVec against the contact's message history.
// Returns nil if the contact has fewer than baselineMinMsgs messages.
func (de *DivergenceEngine) Score(
	ctx context.Context,
	collection string,
	contactUUID string,
	contact *Contact,
	newVec []float32,
) (*DivergenceResult, error) {
	if !contact.BaselineReady {
		return nil, nil // not enough history yet
	}

	// Pull recent message vectors for this contact
	limit := uint32(de.baselineN)
	points, err := de.qc.client.Scroll(ctx, &qdrant.ScrollPoints{
		CollectionName: collection,
		Filter: &qdrant.Filter{
			Must: []*qdrant.Condition{
				qdrant.NewMatch("contact_uuid", contactUUID),
			},
		},
		Limit:       &limit,
		WithPayload: qdrant.NewWithPayload(false),
		WithVectors: &qdrant.WithVectorsSelector{
			SelectorOptions: &qdrant.WithVectorsSelector_Enable{Enable: true},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("fetch baseline vectors: %w", err)
	}
	if len(points) < baselineMinMsgs {
		return nil, nil // still not enough
	}

	// Extract vectors from scroll results (VectorsOutput type)
	var vecs [][]float32
	for _, pt := range points {
		if pt.Vectors == nil {
			continue
		}
		switch v := pt.Vectors.VectorsOptions.(type) {
		case *qdrant.VectorsOutput_Vector:
			if v.Vector != nil {
				vecs = append(vecs, v.Vector.Data)
			}
		}
	}
	if len(vecs) < baselineMinMsgs {
		return nil, nil
	}

	c := centroid(vecs)
	sim := cosineSimilarity(newVec, c)
	score := 1.0 - sim // divergence = 1 - similarity
	if score < 0 {
		score = 0
	}

	tier := scoreTier(score)
	alert := tier == TierUnusual || tier == TierAnomaly
	trendAlert := contact.ConsecutiveAlerts >= 2 && alert // 3rd consecutive

	reason := ""
	switch tier {
	case TierDrift:
		reason = "mild drift from contact baseline"
	case TierUnusual:
		reason = "unusual semantic distance from contact baseline"
	case TierAnomaly:
		reason = "high semantic distance — possible impersonation or account change"
	}
	if trendAlert {
		reason = "trend alert: " + reason
	}

	return &DivergenceResult{
		Score:        score,
		Tier:         tier,
		Alert:        alert,
		TrendAlert:   trendAlert,
		BaselineMsgs: len(vecs),
		Reason:       reason,
	}, nil
}
