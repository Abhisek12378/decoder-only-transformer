import torch
import torch.nn as nn
import math
import json
import tiktoken
import gradio as gr

# ============================================================
# Model Architecture
# ============================================================

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        ff_output = self.dropout2(ff_output)
        x = self.norm2(x + ff_output)
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embeddings = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_embeddings(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)
        x = self.fc_out(x)
        return x

# ============================================================
# Masking
# ============================================================

def create_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    mask = mask.unsqueeze(0).unsqueeze(1)
    return mask

# ============================================================
# Load Model
# ============================================================

device = torch.device('cpu')

with open("config.json", "r") as f:
    config = json.load(f)

enc = tiktoken.get_encoding("gpt2")

model = GPTModel(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    n_heads=config["n_heads"],
    d_ff=config["d_ff"],
    n_layers=config["n_layers"],
    max_seq_len=config["max_seq_len"],
    dropout=config["dropout"]
)

model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# ============================================================
# Generation Function
# ============================================================

def generate(prompt, max_new_tokens=100, temperature=0.8):
    if not prompt.strip():
        return "Please enter a prompt."

    token_ids = enc.encode(prompt)
    max_seq_len = config["max_seq_len"]

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            input_ids = token_ids[-max_seq_len:]
            x = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
            mask = create_causal_mask(x.size(1), device)
            output = model(x, mask)
            logits = output[0, -1, :]
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            token_ids.append(next_token)

    generated_text = enc.decode(token_ids)

    # Clean up WikiText artifacts
    generated_text = generated_text.replace(" @-@ ", "-")
    generated_text = generated_text.replace(" @,@ ", ",")
    generated_text = generated_text.replace(" @.@ ", ".")

    return generated_text

# ============================================================
# Gradio Interface
# ============================================================

examples = [
    ["The capital of France is", 80, 0.8],
    ["In the year 1969, the first", 80, 0.8],
    ["The theory of relativity states that", 100, 0.7],
    ["Football is a sport that", 80, 0.9],
    ["The history of artificial intelligence", 100, 0.8],
]

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter a text prompt..."),
        gr.Slider(minimum=10, maximum=200, value=80, step=10, label="Max New Tokens"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.8, step=0.1, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="GPT Text Generator",
    description="Decoder-only Transformer built from scratch using PyTorch. Trained on WikiText-103 (~118M tokens from Wikipedia). Architecture: 6 decoder layers, 8 attention heads, d_model=512, ~70M parameters. The model generates text by predicting one token at a time.",
    examples=examples,
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()
