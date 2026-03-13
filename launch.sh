#!/bin/zsh
# hindsight-relay launcher — loads secrets from Doppler
set -e

export OPENAI_API_KEY=$(/opt/homebrew/bin/doppler secrets --project ezer-mirror --config prd get OPENAI_API_KEY --plain)
export QDRANT_URL="http://localhost:6334"
export PORT="7358"
export LOG_LEVEL="info"

# Agent → collection map (matches Hindsight provisioning)
export HINDSIGHT_AGENT_MAP='{"main":{"memories":"ezer_memories","facts":"ezer_facts","episodes":"ezer_episodes"},"coding":{"memories":"coding_memories","facts":"coding_facts","episodes":"coding_episodes"},"marketing":{"memories":"marketing_memories","facts":"marketing_facts","episodes":"marketing_episodes"},"mortgage":{"memories":"mortgage_memories","facts":"mortgage_facts","episodes":"mortgage_episodes"},"money":{"memories":"money_memories","facts":"money_facts","episodes":"money_episodes"},"sales":{"memories":"sales_memories","facts":"sales_facts","episodes":"sales_episodes"},"research":{"memories":"research_memories","facts":"research_facts","episodes":"research_episodes"},"pm":{"memories":"pm_memories","facts":"pm_facts","episodes":"pm_episodes"}}'

exec /Users/botbot/hindsight-relay/hindsight-relay
