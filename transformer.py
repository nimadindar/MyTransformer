import numpy as np

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self, vocab_size, num_layers, num_heads, model_dim, ff_dim, max_len):
        super(Transformer, self).__init__()
        self.model_dim = model_dim
        self.embdding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(max_len, model_dim)

        self.encoder = Encoder(num_layers, num_heads, model_dim, ff_dim)
        self.decoder = Decoder(num_layers, num_heads, model_dim, ff_dim)

        self.output_layer = nn.Linear(model_dim, vocab_size)

    def forward(self, src, tgt):
        src = self.embdding(src)
        tgt = self.embdding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        encoder_output = self.encoder(src)
        decoder_output = self.decoder(tgt, encoder_output)

        output = self.output_layer(decoder_output)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, model_dim):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * -(np.log(10000.0) / model_dim))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('encoding', encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]


class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, model_dim, fF_dim):
        super(Decoder, self).__init__()

        self.layers = nn.ModuleList([
            DecoderLayer(num_heads, model_dim, fF_dim) for _ in range(num_layers)
        ])
    
    def forward(self, x, encoder_output, tgt_mask = None):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, num_heads, model_dim, ff_dim):
        super(DecoderLayer, self).__init__()
        
        self.masked_multi_head_attention = MultiHeadAttention(num_heads, model_dim)
        self.cross_attention = MultiHeadAttention(num_heads, model_dim)

        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, model_dim)
        )

        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)
        self.layer_norm3 = nn.LayerNorm(model_dim)

    def forward(self, x, encoder_output, tgt_mask):
        masked_attention_output = self.masked_multi_head_attention.multi_head_attention(x, x, x, mask = tgt_mask)
        x = self.layer_norm1(masked_attention_output + x)

        cross_attention_output = self.cross_attention.multi_head_attention(x, encoder_output, encoder_output)
        x = self.layer_norm2(cross_attention_output + x)

        ffn_output = self.ffn(x)
        x = self.layer_norm3(ffn_output + x)

        return x


class Encoder(nn.Module):
    def __init__(self, num_heads, model_dim, ff_dim, num_layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(num_heads, model_dim, ff_dim) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, model_dim, ff_dim):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(num_heads, model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, model_dim))
        self.layer_norm1 = nn.LayerNorm(model_dim)
        self.layer_norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        attention_output = self.multi_head_attention.multi_head_attention(x,x,x)
        x = self.layer_norm1(attention_output + x)

        ffn_output = self.ffn(x)
        x = self.layer_norm2(ffn_output + x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, model_dim):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.d_k = model_dim // num_heads 

        self.W_Q = nn.Linear(model_dim, model_dim)
        self.W_K = nn.Linear(model_dim, model_dim)
        self.W_V = nn.Linear(model_dim, model_dim)
        
        self.W_O = nn.Linear(model_dim, model_dim)

    def split_heads(self, x):
        """Split the input tensor into multiple heads."""
        batch_size, seq_len, model_dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        return x

    def multi_head_attention(self, Q, K, V, mask = None):
        """Perform multi-head attention."""
        Q = self.split_heads(self.W_Q(Q))
        K = self.split_heads(self.W_K(K))
        V = self.split_heads(self.W_V(V)) 

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.nn.functional.softmax(scores, dim = -1)
        attention_output = torch.matmul(attention_weights, V)

        attention_output = attention_output.transpose(1,2).contiguous()
        attention_output = attention_output.view(Q.size(0), -1, self.model_dim)

        return self.W_O(attention_output)


class AttentionHead:
    def __init__(self, Q, K, V):
        self.Q = Q
        self.K = K
        self.V = V

        self.d_k = Q.shape[-1]  

    def attention_head(self):
        """Compute the scaled dot-product attention."""
        scores = np.dot(self.Q, self.K.transpose(-2, -1)) / np.sqrt(self.d_k)

        attention_weights = self.softmax(scores, axis=-1)
        output = np.dot(attention_weights, self.V)

        return output, attention_weights

    def softmax(self, x, axis):
        """Compute softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
