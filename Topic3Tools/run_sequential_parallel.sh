#!/bin/bash
# Topic 3 Task 1: Time sequential vs parallel execution.

echo "=== Sequential ==="
time ( python task1_ollama_single_topic.py --topic astronomy ; python task1_ollama_single_topic.py --topic business_ethics )

echo ""
echo "=== Parallel ==="
time ( python task1_ollama_single_topic.py --topic astronomy & python task1_ollama_single_topic.py --topic business_ethics & wait )
