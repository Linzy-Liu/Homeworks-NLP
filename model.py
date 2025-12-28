"""RNN-based Neural Machine Translation Model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
import random


class Encoder(nn.Module):
    """编码器：双层单向RNN"""
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.3, rnn_type: str = 'GRU', padding_idx: int = 0):
        super().__init__()
        self.hidden_dim, self.num_layers, self.rnn_type = hidden_dim, num_layers, rnn_type
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        rnn_cls = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers, batch_first=True, 
                          dropout=dropout if num_layers > 1 else 0)
    
    def load_pretrained_embeddings(self, pretrained_weights: torch.Tensor, freeze: bool = False):
        """加载预训练词向量"""
        self.embedding.weight.data.copy_(pretrained_weights)
        if freeze:
            self.embedding.weight.requires_grad = False
        print(f"Encoder: 已加载预训练词向量 {pretrained_weights.shape}, freeze={freeze}")
    
    def forward(self, src: torch.Tensor, src_lengths: Optional[torch.Tensor] = None):
        embedded = self.dropout(self.embedding(src))
        if src_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, hidden = self.rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class DotProductAttention(nn.Module):
    """点积注意力"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.scale = hidden_dim ** 0.5
    
    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = torch.bmm(enc_outputs, dec_hidden.unsqueeze(2)).squeeze(2) / self.scale
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1), weights


class MultiplicativeAttention(nn.Module):
    """乘性注意力 (Luong)"""
    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.W = nn.Linear(enc_dim, dec_dim, bias=False)
    
    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = torch.bmm(self.W(enc_outputs), dec_hidden.unsqueeze(2)).squeeze(2)
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1), weights


class AdditiveAttention(nn.Module):
    """加性注意力 (Bahdanau)"""
    def __init__(self, enc_dim: int, dec_dim: int, attn_dim: int = 256):
        super().__init__()
        self.W1 = nn.Linear(dec_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(enc_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)
    
    def forward(self, dec_hidden: torch.Tensor, enc_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        scores = self.v(torch.tanh(self.W1(dec_hidden).unsqueeze(1) + self.W2(enc_outputs))).squeeze(2)
        if mask is not None: 
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1), weights


def get_attention(attn_type: str, enc_dim: int, dec_dim: int) -> nn.Module:
    if attn_type == 'dot':
        assert enc_dim == dec_dim, "点积注意力要求encoder和decoder的hidden_dim相同"
        return DotProductAttention(enc_dim)
    elif attn_type == 'multiplicative':
        return MultiplicativeAttention(enc_dim, dec_dim)
    return AdditiveAttention(enc_dim, dec_dim)


class Decoder(nn.Module):
    """解码器：双层单向RNN + Attention"""
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512,
                 enc_hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.3,
                 rnn_type: str = 'GRU', attention_type: str = 'additive', padding_idx: int = 0):
        super().__init__()
        self.hidden_dim, self.enc_hidden_dim = hidden_dim, enc_hidden_dim
        self.rnn_type, self.num_layers = rnn_type, num_layers
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.attention = get_attention(attention_type, enc_hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
        rnn_cls = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        self.rnn = rnn_cls(embed_dim + enc_hidden_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_dim + enc_hidden_dim, vocab_size)
    
    def load_pretrained_embeddings(self, pretrained_weights: torch.Tensor, freeze: bool = False):
        """加载预训练词向量"""
        self.embedding.weight.data.copy_(pretrained_weights)
        if freeze:
            self.embedding.weight.requires_grad = False
        print(f"Decoder: 已加载预训练词向量 {pretrained_weights.shape}, freeze={freeze}")
    
    def forward_step(self, token: torch.Tensor, hidden, enc_outputs: torch.Tensor, mask=None):
        if token.dim() == 1: token = token.unsqueeze(1)
        embedded = self.dropout(self.embedding(token))
        h_attn = hidden[0][-1] if self.rnn_type == 'LSTM' else hidden[-1]
        context, weights = self.attention(h_attn, enc_outputs, mask)
        rnn_out, hidden = self.rnn(torch.cat([embedded, context.unsqueeze(1)], dim=2), hidden)
        output = self.fc_out(torch.cat([rnn_out.squeeze(1), context], dim=1))
        return output, hidden, weights
    
    def forward(self, tgt: torch.Tensor, hidden, enc_outputs: torch.Tensor, mask=None, 
                tf_ratio: float = 1.0, return_attention: bool = False):
        """
        Args:
            return_attention: 是否返回注意力权重（训练时设为False节省显存）
        """
        outputs = []
        attentions = [] if return_attention else None
        token = tgt[:, 0]
        for t in range(1, tgt.size(1)):
            output, hidden, weights = self.forward_step(token, hidden, enc_outputs, mask)
            outputs.append(output)
            if return_attention:
                attentions.append(weights)
            token = tgt[:, t] if random.random() < tf_ratio else output.argmax(dim=1)
        attn_out = torch.stack(attentions, dim=1) if return_attention else None
        return torch.stack(outputs, dim=1), attn_out


class Seq2Seq(nn.Module):
    """完整Seq2Seq模型"""
    def __init__(self, encoder: Encoder, decoder: Decoder, src_pad_idx: int = 0):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.src_pad_idx = src_pad_idx
        
        if encoder.hidden_dim != decoder.hidden_dim:
            self.h_proj = nn.Linear(encoder.hidden_dim, decoder.hidden_dim)
            self.c_proj = nn.Linear(encoder.hidden_dim, decoder.hidden_dim) if encoder.rnn_type == 'LSTM' else None
        else:
            self.h_proj = self.c_proj = None
    
    def _project_hidden(self, hidden):
        if self.encoder.rnn_type == 'LSTM':
            h, c = hidden
            if self.h_proj: h, c = self.h_proj(h), self.c_proj(c)
            return (h, c)
        return self.h_proj(hidden) if self.h_proj else hidden
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_lengths=None, 
                tf_ratio: float = 1.0, return_attention: bool = False):
        enc_outputs, enc_hidden = self.encoder(src, src_lengths)
        # 确保mask长度与enc_outputs匹配
        mask = (src[:, :enc_outputs.size(1)] != self.src_pad_idx).long()
        return self.decoder(tgt, self._project_hidden(enc_hidden), enc_outputs, mask, tf_ratio, return_attention)
    
    def greedy_decode(self, src: torch.Tensor, src_lengths=None, max_len: int = 100, 
                      bos_idx: int = 1, eos_idx: int = 2):
        self.eval()
        batch_size, device = src.size(0), src.device
        with torch.no_grad():
            enc_outputs, enc_hidden = self.encoder(src, src_lengths)
            # 确保mask长度与enc_outputs匹配
            mask = (src[:, :enc_outputs.size(1)] != self.src_pad_idx).long()
            hidden = self._project_hidden(enc_hidden)
            
            token = torch.full((batch_size,), bos_idx, dtype=torch.long, device=device)
            preds, finished = [token], torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for _ in range(max_len):
                output, hidden, _ = self.decoder.forward_step(token, hidden, enc_outputs, mask)
                token = output.argmax(dim=1)
                preds.append(token)
                finished = finished | (token == eos_idx)
                if finished.all(): break
        return torch.stack(preds, dim=1), None
    
    def beam_search_decode(self, src: torch.Tensor, src_lengths=None, max_len: int = 100,
                          beam_size: int = 5, 
                          bos_idx: int = 1, eos_idx: int = 2, 
                          length_penalty: float = 0.6):
        self.eval()
        device = src.device
        all_preds, all_scores = [], []
        
        with torch.no_grad():
            for b in range(src.size(0)):
                single_src = src[b:b+1]
                single_len = src_lengths[b:b+1] if src_lengths is not None else None
                enc_outputs, enc_hidden = self.encoder(single_src, single_len)
                # 确保mask长度与enc_outputs匹配（pad_packed_sequence可能改变长度）
                enc_len = enc_outputs.size(1)
                mask = (single_src[:, :enc_len] != self.src_pad_idx).long()
                hidden = self._project_hidden(enc_hidden)
                
                beams = [(torch.tensor([bos_idx], device=device), 0.0, hidden)]
                completed = []
                
                for _ in range(max_len):
                    candidates = []
                    for seq, score, h in beams:
                        if seq[-1].item() == eos_idx:
                            ln = ((5 + len(seq)) / 6) ** length_penalty
                            completed.append((seq, score / ln))
                            continue
                        out, new_h, _ = self.decoder.forward_step(seq[-1].unsqueeze(0), h, enc_outputs, mask)
                        log_probs = F.log_softmax(out, dim=1)
                        topk_lp, topk_idx = log_probs.topk(beam_size, dim=1)
                        for i in range(beam_size):
                            candidates.append((torch.cat([seq, topk_idx[0, i:i+1]]), score + topk_lp[0, i].item(), new_h))
                    if not candidates: break
                    beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
                
                for seq, score, _ in beams:
                    ln = ((5 + len(seq)) / 6) ** length_penalty
                    completed.append((seq, score / ln))
                
                best = max(completed, key=lambda x: x[1]) if completed else (torch.tensor([bos_idx, eos_idx], device=device), float('-inf'))
                all_preds.append(best[0])
                all_scores.append(best[1])
        
        max_len = max(len(p) for p in all_preds)
        padded = torch.full((src.size(0), max_len), eos_idx, dtype=torch.long, device=device)
        for i, p in enumerate(all_preds): padded[i, :len(p)] = p
        return padded, torch.tensor(all_scores, device=device)


def create_model(src_vocab_size: int, tgt_vocab_size: int, 
                 embed_dim: int = 256, hidden_dim: int = 512,
                 num_layers: int = 2, 
                 dropout: float = 0.3, 
                 rnn_type: str = 'GRU',
                 attention_type: str = 'additive', 
                 src_pad_idx: int = 0, tgt_pad_idx: int = 0) -> Seq2Seq:
    
    encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout, rnn_type, src_pad_idx)
    decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, hidden_dim, num_layers, dropout, rnn_type, attention_type, tgt_pad_idx)
    model = Seq2Seq(encoder, decoder, src_pad_idx)
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.1, 0.1)
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for n, p in m.named_parameters():
                if 'weight' in n: nn.init.orthogonal_(p)
                elif 'bias' in n: nn.init.zeros_(p)
    
    model.apply(init_weights)
    return model
