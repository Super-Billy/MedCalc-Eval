#!/bin/bash

# Run MedPrompt with k=3 for both direct and cot prompt styles

echo "Running MedPrompt with prompt_style=direct and k=3..."
python medprompt.py --prompt_style direct --k 3

echo "Running MedPrompt with prompt_style=cot and k=3..."
python medprompt.py --prompt_style cot --k 3

echo "✅ All runs completed."
