package main

import "strings"

const (
	maxChunkChars  = 3200
	overlapChars   = 400
	minChunkChars  = 20
)

// ChunkText splits text into overlapping paragraph-based chunks.
func ChunkText(text string) []string {
	paragraphs := strings.Split(text, "\n\n")

	var chunks []string
	current := ""

	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}

		candidate := current
		if candidate != "" {
			candidate += "\n\n"
		}
		candidate += para

		if len(candidate) > maxChunkChars && current != "" {
			// Save current chunk
			if len(current) >= minChunkChars {
				chunks = append(chunks, current)
			}
			// Start next chunk with overlap from end of current
			overlap := current
			if len(overlap) > overlapChars {
				overlap = overlap[len(overlap)-overlapChars:]
				// Trim to word boundary
				if idx := strings.Index(overlap, " "); idx >= 0 {
					overlap = overlap[idx+1:]
				}
			}
			current = overlap + "\n\n" + para
		} else {
			current = candidate
		}
	}

	if len(current) >= minChunkChars {
		chunks = append(chunks, current)
	}

	return chunks
}

// ExtractHeading returns the first Markdown heading found in text, else "".
func ExtractHeading(text string) string {
	for _, line := range strings.Split(text, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "## ") || strings.HasPrefix(line, "# ") {
			return line
		}
	}
	return ""
}
