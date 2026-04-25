---
title: GPT Text Generator
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "5.12.0"
app_file: app.py
pinned: false
---

# GPT Text Generator

A decoder-only Transformer built entirely from scratch using PyTorch, trained on the WikiText-103 dataset.

## Architecture
- **Type:** Decoder-only Transformer (GPT-style)
- **Decoder Layers:** 6
- **Attention Heads:** 8
- **Model Dimension:** 512
- **Feed-Forward Dimension:** 2048
- **Parameters:** ~70M
- **Tokenizer:** GPT-2 BPE (tiktoken)
- **Dataset:** WikiText-103 (~118M tokens from Wikipedia)
- **Validation Perplexity:** 36.06

## How It Works
Every component — embeddings, positional encoding, multi-head causal attention, feed-forward layers — was implemented from scratch. No pre-trained weights or HuggingFace model classes were used. The model predicts the next token autoregressively using causal masking.

## Controls
- **Max New Tokens:** Controls how many tokens the model generates (10-200)
- **Temperature:** Controls randomness. Low (0.1) = repetitive but safe. High (2.0) = creative but chaotic. Default 0.8 works well.

## Limitations
- Trained on Wikipedia text only, so outputs have Wikipedia-style writing
- 70M parameters is small — factual accuracy is limited
- No instruction tuning — the model continues text rather than answering questions
