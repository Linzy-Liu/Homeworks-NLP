"""
Transformer-based NMT 训练脚本
- 基于 PyTorch nn.Transformer 组件构建
- 消融实验：绝对/相对位置编码，LayerNorm/RMSNorm
- 超参数研究：batch size, lr, model size
- 支持预训练词向量
"""
import os
import json
import time
import math
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from preprocess import (Config, TranslationTokenizer, create_dataloader, 
                        load_all, check_preprocessed_exists, preprocess_pipeline)


# ==================== 配置 ====================

@dataclass
class TransformerConfig:
    """Transformer 模型配置"""
    d_model: int = 256
    n_heads: int = 4
    n_enc_layers: int = 2
    n_dec_layers: int = 2
    d_ff: int = 1024  # 默认值为4*d_model
    dropout: float = 0.1
    max_len: int = 512
    emb_dim: int = 200
    
    # 消融实验选项
    pos_encoding: str = 'absolute'  # 'absolute' or 'relative'
    norm_type: str = 'layernorm'    # 'layernorm' or 'rmsnorm'
    
    # 训练配置
    batch_size: int = 64
    learning_rate: float = 1e-3
    warmup_steps: int = 2000
    scheduler_type: str = 'cosine'  # 'cosine', 'linear', 'inverse_sqrt', 'constant'
    epochs: int = 20
    patience: int = 5
    clip_grad: float = 1.0
    label_smoothing: float = 0.1
    
    # 预训练词向量
    src_embedding_path: Optional[str] = "embeddings/light_Tencent_AILab_ChineseEmbedding.bin"
    tgt_embedding_path: Optional[str] = None
    embedding_binary: bool = True
    freeze_embeddings: bool = False

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerConfig':
        """
        从字典加载配置，仅更新类中已定义的属性，忽略未知键。
        
        Args:
            config_dict (dict): 配置字典
            
        Returns:
            TransformerConfig: 配置实例
        """
        # 获取类中所有带默认值的字段名（包括类型注解中定义的）
        valid_keys = set()
        for key in dir(cls):
            if not key.startswith('_') and not callable(getattr(cls, key)):
                valid_keys.add(key)
        
        # 或更安全的方式：显式列出所有字段（推荐，避免继承/元类干扰）
        # 但为了灵活性，这里使用 __annotations__（需 Python 3.6+）
        # 注意：类属性若未用类型注解，则 __annotations__ 不包含它们
        
        # 更鲁棒的方法：结合 __annotations__ 和类属性默认值
        all_keys = set(cls.__annotations__.keys())
        # 补充可能未注解但存在的类属性（如你全部都注解了，可省略）
        for key in dir(cls):
            if not key.startswith('_') and not callable(getattr(cls, key)):
                all_keys.add(key)
        
        # 创建新实例
        config = cls()
        
        # 更新已知字段
        for key, value in config_dict.items():
            if key in all_keys:
                setattr(config, key, value)
            else:
                # 可选：警告未知字段
                # print(f"Warning: Ignoring unknown config key: {key}")
                pass
        
        return config

    def to_dict(self) -> Dict[str, Any]:
        """
        将当前配置实例导出为字典。
        """
        return {
            key: getattr(self, key)
            for key in self.__annotations__.keys()
            if hasattr(self, key)
        }


# ==================== 归一化层 ====================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA 风格)"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# ==================== 位置编码 ====================

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码 (Attention Is All You Need)"""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class LearnedPositionalEncoding(nn.Module):
    """可学习位置编码"""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pe(positions))


class RelativePositionalBias(nn.Module):
    """相对位置偏置 (T5/ALiBi 风格)"""
    def __init__(self, n_heads: int, max_len: int = 512):
        super().__init__()
        self.n_heads = n_heads
        self.max_len = max_len
        self.relative_bias = nn.Embedding(2 * max_len - 1, n_heads)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        relative = positions.unsqueeze(0) - positions.unsqueeze(1) + self.max_len - 1
        bias = self.relative_bias(relative)  # [seq_len, seq_len, n_heads]
        return bias.permute(2, 0, 1).unsqueeze(0)  # [1, n_heads, seq_len, seq_len]


# ==================== Transformer 模型 ====================

class TransformerNMT(nn.Module):
    """
    基于 PyTorch nn.Transformer 的 NMT 模型
    支持消融实验：位置编码类型和归一化类型
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, config,
                 src_pad_idx: int = 0, tgt_pad_idx: int = 0):
        super().__init__()
        self.config = config
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.d_model = config.d_model
        self.emb_dim = config.emb_dim
        
        # 词嵌入（初始维度可能与预训练词向量不同）
        self.src_embedding = nn.Embedding(src_vocab_size, config.emb_dim, padding_idx=src_pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, config.d_model, padding_idx=tgt_pad_idx)
        
        # 线性投影层（用于维度转换，如果需要的话）
        if config.emb_dim != config.d_model:
            self.src_proj = nn.Linear(config.emb_dim, config.d_model, bias=False)
            self.tgt_proj = None
            print(f"[*] Created projection layers: {config.emb_dim} -> {config.d_model}")
        else:
            self.src_proj = None
            self.tgt_proj = None
        
        self.scale = math.sqrt(config.d_model)
        
        # 位置编码
        if config.pos_encoding == 'relative':
            self.src_pos_encoding = LearnedPositionalEncoding(config.d_model, config.max_len, config.dropout)
            self.tgt_pos_encoding = LearnedPositionalEncoding(config.d_model, config.max_len, config.dropout)
            self.relative_bias = RelativePositionalBias(config.n_heads, config.max_len)
        else:
            self.src_pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_len, config.dropout)
            self.tgt_pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_len, config.dropout)
            self.relative_bias = None
        
        # 选择归一化层
        norm_layer = RMSNorm if config.norm_type == 'rmsnorm' else nn.LayerNorm
        
        # 使用 PyTorch 内置 Transformer 组件
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN (更稳定)
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_enc_layers,
            norm=norm_layer(config.d_model)
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_dec_layers,
            norm=norm_layer(config.d_model)
        )
        
        # 输出层
        self.fc_out = nn.Linear(config.d_model, tgt_vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def load_pretrained_embeddings(self, src_weights: torch.Tensor = None, 
                                   tgt_weights: torch.Tensor = None, freeze: bool = False):
        """加载预训练词向量，支持维度不匹配时的自动投影"""
        if src_weights is not None:
            pretrained_dim = src_weights.shape[1]
            if pretrained_dim != self.d_model:
                # 维度不匹配，创建投影层
                print(f"源语言词向量维度 {pretrained_dim} != d_model {self.d_model}，创建投影层")
                self.src_proj = nn.Linear(pretrained_dim, self.d_model, bias=False).to(src_weights.device)
                # 重新创建embedding层为预训练维度
                self.src_embedding = nn.Embedding(
                    src_weights.shape[0], pretrained_dim, padding_idx=self.src_pad_idx
                ).to(src_weights.device)
            
            self.src_embedding.weight.data.copy_(src_weights)
            if freeze:
                self.src_embedding.weight.requires_grad = False
            print(f"Encoder: 已加载预训练词向量 {src_weights.shape}, freeze={freeze}")
        
        if tgt_weights is not None:
            pretrained_dim = tgt_weights.shape[1]
            if pretrained_dim != self.d_model:
                # 维度不匹配，创建投影层
                print(f"目标语言词向量维度 {pretrained_dim} != d_model {self.d_model}，创建投影层")
                self.tgt_proj = nn.Linear(pretrained_dim, self.d_model, bias=False).to(tgt_weights.device)
                # 重新创建embedding层为预训练维度
                self.tgt_embedding = nn.Embedding(
                    tgt_weights.shape[0], pretrained_dim, padding_idx=self.tgt_pad_idx
                ).to(tgt_weights.device)
            
            self.tgt_embedding.weight.data.copy_(tgt_weights)
            if freeze:
                self.tgt_embedding.weight.requires_grad = False
            print(f"Decoder: 已加载预训练词向量 {tgt_weights.shape}, freeze={freeze}")
    
    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """源序列 padding mask [batch, src_len]"""
        return src == self.src_pad_idx
    
    def make_tgt_mask(self, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """目标序列 causal mask 和 padding mask"""
        tgt_len = tgt.size(1)
        # Causal mask (上三角为 True)
        causal_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
        # Padding mask
        padding_mask = tgt == self.tgt_pad_idx
        return causal_mask, padding_mask
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # 嵌入 + 投影（如果需要）+ 位置编码
        src_emb = self.src_embedding(src)
        if self.src_proj is not None:
            src_emb = self.src_proj(src_emb)
        src_emb = self.src_pos_encoding(src_emb * self.scale)
        
        tgt_emb = self.tgt_embedding(tgt)
        if self.tgt_proj is not None:
            tgt_emb = self.tgt_proj(tgt_emb)
        tgt_emb = self.tgt_pos_encoding(tgt_emb * self.scale)
        
        # 创建 mask
        src_key_padding_mask = self.make_src_mask(src)
        tgt_mask, tgt_key_padding_mask = self.make_tgt_mask(tgt)
        
        # 编码
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        # 解码
        output = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        
        return self.fc_out(output)
    
    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, max_len: int = 100,
                      bos_idx: int = 1, eos_idx: int = 2) -> torch.Tensor:
        self.eval()
        batch_size, device = src.size(0), src.device
        
        # 编码（带投影）
        src_emb = self.src_embedding(src)
        if self.src_proj is not None:
            src_emb = self.src_proj(src_emb)
        src_emb = self.src_pos_encoding(src_emb * self.scale)
        src_mask = self.make_src_mask(src)
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        
        # 初始化
        tgt = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len):
            tgt_emb = self.tgt_embedding(tgt)
            if self.tgt_proj is not None:
                tgt_emb = self.tgt_proj(tgt_emb)
            tgt_emb = self.tgt_pos_encoding(tgt_emb * self.scale)
            tgt_mask, _ = self.make_tgt_mask(tgt)
            
            output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                                  memory_key_padding_mask=src_mask)
            next_token = self.fc_out(output[:, -1:]).argmax(dim=-1)
            tgt = torch.cat([tgt, next_token], dim=1)
            
            finished = finished | (next_token.squeeze(-1) == eos_idx)
            if finished.all():
                break
        
        return tgt
    
    @torch.no_grad()
    def beam_search_decode(self, src: torch.Tensor, max_len: int = 100,
                          beam_size: int = 5, bos_idx: int = 1, eos_idx: int = 2,
                          length_penalty: float = 0.6) -> torch.Tensor:
        self.eval()
        device = src.device
        all_preds = []
        
        for b in range(src.size(0)):
            single_src = src[b:b+1]
            src_emb = self.src_embedding(single_src)
            if self.src_proj is not None:
                src_emb = self.src_proj(src_emb)
            src_emb = self.src_pos_encoding(src_emb * self.scale)
            src_mask = self.make_src_mask(single_src)
            memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
            
            beams = [(torch.tensor([[bos_idx]], device=device), 0.0)]
            completed = []
            
            for _ in range(max_len):
                candidates = []
                for seq, score in beams:
                    if seq[0, -1].item() == eos_idx:
                        ln = ((5 + seq.size(1)) / 6) ** length_penalty
                        completed.append((seq, score / ln))
                        continue
                    
                    tgt_emb = self.tgt_embedding(seq)
                    if self.tgt_proj is not None:
                        tgt_emb = self.tgt_proj(tgt_emb)
                    tgt_emb = self.tgt_pos_encoding(tgt_emb * self.scale)
                    tgt_mask, _ = self.make_tgt_mask(seq)
                    output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask,
                                          memory_key_padding_mask=src_mask)
                    log_probs = F.log_softmax(self.fc_out(output[:, -1:]), dim=-1).squeeze(1)
                    topk_lp, topk_idx = log_probs.topk(beam_size, dim=-1)
                    
                    for i in range(beam_size):
                        new_seq = torch.cat([seq, topk_idx[:, i:i+1]], dim=1)
                        candidates.append((new_seq, score + topk_lp[0, i].item()))
                
                if not candidates:
                    break
                beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            
            for seq, score in beams:
                ln = ((5 + seq.size(1)) / 6) ** length_penalty
                completed.append((seq, score / ln))
            
            best = max(completed, key=lambda x: x[1]) if completed else \
                   (torch.tensor([[bos_idx, eos_idx]], device=device), float('-inf'))
            all_preds.append(best[0].squeeze(0))
        
        max_pred_len = max(p.size(0) for p in all_preds)
        padded = torch.full((src.size(0), max_pred_len), eos_idx, dtype=torch.long, device=device)
        for i, p in enumerate(all_preds):
            padded[i, :p.size(0)] = p
        return padded


# ==================== 损失函数 ====================

class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(-1, self.vocab_size)
        target = target.reshape(-1)
        
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        
        mask = target != self.padding_idx
        loss = -torch.sum(true_dist * F.log_softmax(pred, dim=-1), dim=-1)
        return loss.masked_select(mask).mean()


# ==================== 学习率调度 ====================

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, 
                     scheduler_type: str = 'cosine', peak_lr: float = None):
    """
    学习率调度器（支持多种策略）
    
    Args:
        optimizer: 优化器
        warmup_steps: warmup 步数
        total_steps: 总训练步数
        scheduler_type: 调度器类型
            - 'cosine': warmup + cosine annealing (推荐，峰值可调)
            - 'linear': warmup + linear decay
            - 'inverse_sqrt': 原始 Transformer 调度器 (warmup + 平方根衰减)
            - 'constant': warmup + 恒定学习率
        peak_lr: 峰值学习率（仅 cosine/linear/constant 使用，inverse_sqrt 自动计算）
    
    Returns:
        学习率调度器
    """
    if scheduler_type == 'inverse_sqrt':
        # 原始 Transformer 调度器（峰值较低）
        def lr_lambda(step):
            step = max(step, 1)
            d_model = 256  # 使用固定值或从 optimizer 获取
            return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'cosine':
        # Warmup + Cosine Annealing（峰值可调，推荐）
        def lr_lambda(step):
            step = max(step, 1)
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'linear':
        # Warmup + Linear Decay
        def lr_lambda(step):
            step = max(step, 1)
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return max(0.0, (total_steps - step) / (total_steps - warmup_steps))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'constant':
        # Warmup + Constant（适合快速实验）
        def lr_lambda(step):
            step = max(step, 1)
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 1.0
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


# ==================== 训练和评估 ====================

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, count = 0.0, 0
    pbar = tqdm(loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in pbar:
            src, tgt = batch['src_ids'].to(device), batch['tgt_ids'].to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output, tgt[:, 1:])
            batch_loss = loss.item()
            total_loss += batch_loss * src.size(0)
            count += src.size(0)
            pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg': f'{total_loss/count:.4f}'})
    
    return total_loss / count


def calculate_bleu(model, loader, tokenizer, device, decode='greedy', beam_size=5,
                   max_samples=200, src_lang='zh'):
    from train import compute_bleu
    model.eval()
    refs, hyps, count = [], [], 0
    tgt_vocab = tokenizer.en_vocab if src_lang == 'zh' else tokenizer.zh_vocab
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"BLEU ({decode})"):
            if count >= max_samples:
                break
            src, tgt = batch['src_ids'].to(device), batch['tgt_ids']
            
            preds = model.greedy_decode(src, 100, tgt_vocab.bos_idx, tgt_vocab.eos_idx) if decode == 'greedy' \
                    else model.beam_search_decode(src, 100, beam_size, tgt_vocab.bos_idx, tgt_vocab.eos_idx)
            
            for i in range(src.size(0)):
                if count >= max_samples:
                    break
                ref = [t for t in tgt_vocab.ids_to_tokens(tgt[i].tolist()) if t not in Config.SPECIAL_TOKENS]
                hyp = [t for t in tgt_vocab.ids_to_tokens(preds[i].tolist()) if t not in Config.SPECIAL_TOKENS]
                refs.append([ref])
                hyps.append(hyp)
                count += 1
    
    return compute_bleu(refs, hyps)


def translate(model, tokenizer, text, device, decode='greedy', beam_size=5, src_lang='zh'):
    model.eval()
    if src_lang == 'zh':
        ids = tokenizer.encode_chinese(text)
        tgt_vocab, decode_fn = tokenizer.en_vocab, tokenizer.decode_english
    else:
        ids = tokenizer.encode_english(text)
        tgt_vocab, decode_fn = tokenizer.zh_vocab, tokenizer.decode_chinese
    
    src = torch.tensor([ids], device=device)
    with torch.no_grad():
        preds = model.greedy_decode(src, 100, tgt_vocab.bos_idx, tgt_vocab.eos_idx) if decode == 'greedy' \
                else model.beam_search_decode(src, 100, beam_size, tgt_vocab.bos_idx, tgt_vocab.eos_idx)
    return decode_fn(preds[0].tolist())


# ==================== 主训练函数 ====================

def train(config: TransformerConfig, vocab_dir: str = "vocab", save_dir: str = "checkpoints",
          checkpoint: str = None, device: str = None, src_lang: str = 'zh', use_hunyuan: bool = False,
          train_file: str = 'data/train_10k.jsonl', experiment_type: str = ""):
    """Transformer 训练主函数
    
    Args:
        use_hunyuan: 是否使用混元重译数据 (zh_hy 字段)
        train_file: 训练数据文件路径
        experiment_type: 实验类型标识 ('ablation', 'hyperparam', '' 为普通训练)
    """
    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    if check_preprocessed_exists(vocab_dir):
        tokenizer, train_ds, valid_ds, _ = load_all(vocab_dir, src_lang=src_lang, use_hunyuan=use_hunyuan)
    else:
        tokenizer, train_ds, valid_ds, _ = preprocess_pipeline(train_file=train_file, save_dir=vocab_dir, src_lang=src_lang, use_hunyuan=use_hunyuan)
    
    train_loader = create_dataloader(train_ds, config.batch_size, shuffle=True)
    valid_loader = create_dataloader(valid_ds, config.batch_size, shuffle=False) if valid_ds else None
    
    # 词表大小
    if src_lang == 'zh':
        src_vocab_size, tgt_vocab_size = len(tokenizer.zh_vocab), len(tokenizer.en_vocab)
        src_vocab, tgt_vocab = tokenizer.zh_vocab, tokenizer.en_vocab
    else:
        src_vocab_size, tgt_vocab_size = len(tokenizer.en_vocab), len(tokenizer.zh_vocab)
        src_vocab, tgt_vocab = tokenizer.en_vocab, tokenizer.zh_vocab
    
    # 创建模型
    model = TransformerNMT(src_vocab_size, tgt_vocab_size, config, Config.PAD_IDX, Config.PAD_IDX).to(device)
    
    # 加载预训练词向量
    if config.src_embedding_path or config.tgt_embedding_path:
        from embeddings import load_pretrained_vectors
        
        src_weights, tgt_weights = None, None
        if config.src_embedding_path and os.path.exists(config.src_embedding_path):
            src_weights, _, _ = load_pretrained_vectors(
                config.src_embedding_path, src_vocab.token2idx, config.d_model, config.embedding_binary)
            src_weights = src_weights.to(device)
        
        if config.tgt_embedding_path and os.path.exists(config.tgt_embedding_path):
            tgt_weights, _, _ = load_pretrained_vectors(
                config.tgt_embedding_path, tgt_vocab.token2idx, config.d_model, config.embedding_binary)
            tgt_weights = tgt_weights.to(device)
        
        model.load_pretrained_embeddings(src_weights, tgt_weights, config.freeze_embeddings)
    
    exp_name = f"transformer_{config.pos_encoding}_{config.norm_type}_d{config.d_model}"
    
    # 添加实验类型和关键超参数信息
    if experiment_type:
        exp_name = f"{experiment_type}_{exp_name}"
    
    # 记录关键超参数以避免覆盖
    if experiment_type == "hyperparam":
        # 超参数实验：记录 batch_size 和 learning_rate
        exp_name += f"_bs{config.batch_size}_lr{config.learning_rate:.0e}"
    elif config.learning_rate != 1e-3:  # 非默认学习率时也记录
        exp_name += f"_lr{config.learning_rate:.0e}"
    
    if config.batch_size != 64:  # 非默认 batch size 时记录
        if "bs" not in exp_name:  # 避免重复
            exp_name += f"_bs{config.batch_size}"
    
    print(f"实验: {exp_name}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"配置: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"layers={config.n_enc_layers}/{config.n_dec_layers}, d_ff={config.d_ff}")
    print(f"位置编码: {config.pos_encoding}, 归一化: {config.norm_type}")
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    # 计算总训练步数
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * config.epochs
    
    scheduler = get_lr_scheduler(
        optimizer, 
        warmup_steps=config.warmup_steps, 
        total_steps=total_steps,
        scheduler_type=config.scheduler_type,
        peak_lr=config.learning_rate
    )
    print(f"学习率调度: {config.scheduler_type}, 峰值lr={config.learning_rate}, warmup={config.warmup_steps}步")
    
    criterion = LabelSmoothingLoss(tgt_vocab_size, Config.PAD_IDX, config.label_smoothing)
    
    # 恢复训练
    start_epoch, best_loss = 0, float('inf')
    if checkpoint and os.path.exists(checkpoint):
        print(f"从 {checkpoint} 恢复训练...")
        ckpt = torch.load(checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt['valid_loss']
    
    # 训练循环
    patience_cnt = 0
    history = {'train_loss': [], 'valid_loss': [], 'bleu_greedy': [], 'bleu_beam': [], 'step_losses': []}
    
    for epoch in range(start_epoch, start_epoch + config.epochs):
        t0 = time.time()
        model.train()
        total_loss, count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        steps_per_epoch = len(train_loader)
        eval_interval = max(1, steps_per_epoch // 5)
        
        for step, batch in enumerate(pbar):
            src, tgt = batch['src_ids'].to(device), batch['tgt_ids'].to(device)
            optimizer.zero_grad()
            
            output = model(src, tgt[:, :-1])
            loss = criterion(output, tgt[:, 1:])
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()
            scheduler.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss * src.size(0)
            count += src.size(0)
            history['step_losses'].append(batch_loss)
            pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg': f'{total_loss/count:.4f}',
                             'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
            del output, loss
            
            # 步内评估 - 仅用于监控，不影响保存逻辑
            if valid_loader and (step + 1) % eval_interval == 0:
                val_loss = evaluate(model, valid_loader, criterion, device)
                model.train()
                pbar.write(f"  [Step {step+1}] Valid Loss: {val_loss:.4f} | Train Avg: {total_loss/count:.4f}")

        # Epoch 结束后的评估
        train_loss = total_loss / count
        valid_loss = evaluate(model, valid_loader, criterion, device) if valid_loader else train_loss
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        
        mins, secs = int((time.time() - t0) / 60), int((time.time() - t0) % 60)
        print(f"Epoch {epoch+1} | {mins}m{secs}s | Train: {train_loss:.4f} PPL: {math.exp(train_loss):.2f} | "
              f"Valid: {valid_loss:.4f} PPL: {math.exp(valid_loss):.2f}")
        
        if (epoch + 1) % 5 == 0 and valid_loader:
            bleu_g = calculate_bleu(model, valid_loader, tokenizer, device, 'greedy', max_samples=200, src_lang=src_lang)
            bleu_b = calculate_bleu(model, valid_loader, tokenizer, device, 'beam', 5, max_samples=100, src_lang=src_lang)
            history['bleu_greedy'].append(bleu_g)
            history['bleu_beam'].append(bleu_b)
            sample_src = "今天天气很好。" if src_lang == 'zh' else "The weather is nice today."
            sample_tgt = translate(model, tokenizer, sample_src, device, 'greedy', src_lang=src_lang)
            print(f"  BLEU: Greedy={bleu_g:.2f}, Beam={bleu_b:.2f}")
            print(f"  翻译示例: '{sample_src}' → '{sample_tgt}'")
        
        # 基于 train_loss 保存最佳模型 (因为 valid/train 分布差异大)
        if train_loss < best_loss:
            best_loss, patience_cnt = train_loss, 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 
                'train_loss': train_loss, 'valid_loss': valid_loss,
                'config': vars(config)
            }, save_dir / f"{exp_name}_best.pt")
            print("  ✓ 保存最佳模型 (基于 Train Loss)")
        else:
            patience_cnt += 1
            if patience_cnt >= config.patience:
                print(f"早停：{config.patience} epochs 训练损失无改善")
                break
    
    with open(save_dir / f"{exp_name}_history.json", 'w') as f:
        json.dump(history, f)
    return model, history


# ==================== 消融实验 ====================

def run_ablation_study(vocab_dir: str = "vocab", save_dir: str = "checkpoints", 
                       epochs: int = 15, src_lang: str = 'zh'):
    """消融实验：对比位置编码和归一化方法"""
    experiments = [
        {'pos_encoding': 'absolute', 'norm_type': 'layernorm'},
        {'pos_encoding': 'relative', 'norm_type': 'layernorm'},
        {'pos_encoding': 'absolute', 'norm_type': 'rmsnorm'},
        {'pos_encoding': 'relative', 'norm_type': 'rmsnorm'},
    ]
    
    results = {}
    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}\n消融实验 {i+1}/{len(experiments)}: {exp}\n{'='*60}")
        config = TransformerConfig(pos_encoding=exp['pos_encoding'], norm_type=exp['norm_type'], epochs=epochs)
        _, history = train(config, vocab_dir, save_dir, src_lang=src_lang, experiment_type="ablation")
        name = f"{exp['pos_encoding']}_{exp['norm_type']}"
        results[name] = {'best_valid_loss': min(history['valid_loss']),
                        'final_bleu': history['bleu_greedy'][-1] if history['bleu_greedy'] else None}
    
    print(f"\n{'='*60}\n消融实验结果汇总\n{'='*60}")
    for name, res in results.items():
        print(f"{name:<30} Loss: {res['best_valid_loss']:.4f} BLEU: {res['final_bleu'] or 'N/A'}")
    
    with open(Path(save_dir) / "ablation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    return results


def run_hyperparameter_study(vocab_dir: str = "vocab", save_dir: str = "checkpoints",
                             epochs: int = 10, src_lang: str = 'zh'):
    """超参数研究：batch size, lr, model size"""
    experiments = [
        {'name': 'batch_32', 'batch_size': 32, 'd_model': 256, 'learning_rate': 1e-3},
        {'name': 'batch_64', 'batch_size': 64, 'd_model': 256, 'learning_rate': 1e-3},
        {'name': 'batch_128', 'batch_size': 128, 'd_model': 256, 'learning_rate': 1e-3},
        {'name': 'lr_5e-4', 'batch_size': 64, 'd_model': 256, 'learning_rate': 5e-4},
        {'name': 'lr_1e-3', 'batch_size': 64, 'd_model': 256, 'learning_rate': 1e-3},
        {'name': 'lr_3e-3', 'batch_size': 64, 'd_model': 256, 'learning_rate': 3e-3},
        {'name': 'small_d128', 'batch_size': 64, 'd_model': 128, 'd_ff': 512, 'n_heads': 4},
        {'name': 'base_d256', 'batch_size': 64, 'd_model': 256, 'd_ff': 1024, 'n_heads': 4},
        {'name': 'large_d512', 'batch_size': 64, 'd_model': 512, 'd_ff': 2048, 'n_heads': 8},
    ]
    
    results = {}
    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}\n超参数实验 {i+1}/{len(experiments)}: {exp['name']}\n{'='*60}")
        config = TransformerConfig(
            d_model=exp.get('d_model', 256), n_heads=exp.get('n_heads', 8),
            d_ff=exp.get('d_ff', 1024), batch_size=exp.get('batch_size', 64),
            learning_rate=exp.get('learning_rate', 1e-4), epochs=epochs
        )
        try:
            _, history = train(config, vocab_dir, save_dir, src_lang=src_lang, experiment_type="hyperparam")
            results[exp['name']] = {'best_valid_loss': min(history['valid_loss']),
                                   'final_bleu': history['bleu_greedy'][-1] if history['bleu_greedy'] else None}
        except RuntimeError as e:
            print(f"实验失败: {e}")
            results[exp['name']] = {'error': str(e)[:100]}
    
    print(f"\n{'='*60}\n超参数实验结果汇总\n{'='*60}")
    for name, res in results.items():
        if 'error' in res:
            print(f"{name:<20} ERROR")
        else:
            print(f"{name:<20} Loss: {res['best_valid_loss']:.4f} BLEU: {res['final_bleu'] or 'N/A'}")
    
    with open(Path(save_dir) / "hyperparameter_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description="Transformer NMT 训练脚本")
    
    parser.add_argument('--mode', default='train', 
                        choices=['train', 'ablation', 'hyperparameter', 'evaluate', 'interactive'])
    
    # 模型配置
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--d_ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--pos_encoding', default='absolute', choices=['absolute', 'relative'])
    parser.add_argument('--norm_type', default='layernorm', choices=['layernorm', 'rmsnorm'])
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3, help='峰值学习率')
    parser.add_argument('--warmup_steps', type=int, default=300)
    parser.add_argument('--scheduler_type', default='cosine', 
                        choices=['cosine', 'linear', 'inverse_sqrt', 'constant'],
                        help='学习率调度器类型')
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # 预训练词向量
    parser.add_argument('--src_embedding', default=None, help='源语言预训练词向量路径')
    parser.add_argument('--tgt_embedding', default=None, help='目标语言预训练词向量路径')
    parser.add_argument('--embedding_binary', action='store_true', help='词向量是否为二进制格式')
    parser.add_argument('--freeze_embeddings', action='store_true', help='是否冻结词向量')
    
    # 路径
    parser.add_argument('--vocab_dir', default='vocab')
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--src_lang', default='zh', choices=['zh', 'en'])
    parser.add_argument('--device', default=None)
    parser.add_argument('--decode_method', default='greedy', choices=['greedy', 'beam'])
    parser.add_argument('--beam_size', type=int, default=5)
    
    # 数据配置
    parser.add_argument('--train_file', default='data/train_10k.jsonl',
                        help='训练数据文件路径（如 data/train_10k_retranslated_hunyuan.jsonl）')
    parser.add_argument('--use_hunyuan', action='store_true',
                        help='使用混元重译数据 (zh_hy 字段替代 zh)')
    
    args = parser.parse_args()
    
    if args.mode == 'ablation':
        run_ablation_study(args.vocab_dir, args.save_dir, args.epochs, args.src_lang)
    elif args.mode == 'hyperparameter':
        run_hyperparameter_study(args.vocab_dir, args.save_dir, args.epochs, args.src_lang)
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            return print("需要 --checkpoint")
        device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ckpt = torch.load(args.checkpoint, map_location=device)
        config = TransformerConfig(**ckpt['config'])
        tokenizer, _, valid_ds, test_ds = load_all(args.vocab_dir, src_lang=args.src_lang)
        ds = test_ds or valid_ds
        if not ds:
            return print("无评估数据")
        loader = create_dataloader(ds, config.batch_size, shuffle=False)
        src_vocab_size = len(tokenizer.zh_vocab) if args.src_lang == 'zh' else len(tokenizer.en_vocab)
        tgt_vocab_size = len(tokenizer.en_vocab) if args.src_lang == 'zh' else len(tokenizer.zh_vocab)
        model = TransformerNMT(src_vocab_size, tgt_vocab_size, config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"BLEU Greedy: {calculate_bleu(model, loader, tokenizer, device, 'greedy', max_samples=500, src_lang=args.src_lang):.2f}")
        print(f"BLEU Beam: {calculate_bleu(model, loader, tokenizer, device, 'beam', args.beam_size, max_samples=200, src_lang=args.src_lang):.2f}")
        samples = ["今天天气很好。", "我喜欢学习。"] if args.src_lang == 'zh' else ["The weather is nice.", "I love learning."]
        for s in samples:
            print(f"  '{s}' → '{translate(model, tokenizer, s, device, 'greedy', src_lang=args.src_lang)}'")
    elif args.mode == 'interactive':
        if not args.checkpoint:
            return print("需要 --checkpoint")
        device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        ckpt = torch.load(args.checkpoint, map_location=device)
        config = TransformerConfig(**ckpt['config'])
        tokenizer, _, _, _ = load_all(args.vocab_dir, src_lang=args.src_lang)
        src_vocab_size = len(tokenizer.zh_vocab) if args.src_lang == 'zh' else len(tokenizer.en_vocab)
        tgt_vocab_size = len(tokenizer.en_vocab) if args.src_lang == 'zh' else len(tokenizer.zh_vocab)
        model = TransformerNMT(src_vocab_size, tgt_vocab_size, config).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        src_name, tgt_name = ("中文", "英文") if args.src_lang == 'zh' else ("英文", "中文")
        print(f"\n交互式翻译 ({src_name}→{tgt_name}, 输入'quit'退出)")
        while True:
            try:
                text = input(f"\n{src_name}: ").strip()
                if text.lower() == 'quit':
                    break
                if text:
                    print(f"{tgt_name}: {translate(model, tokenizer, text, device, args.decode_method, args.beam_size, args.src_lang)}")
            except KeyboardInterrupt:
                break
    else:  # train
        config = TransformerConfig(
            d_model=args.d_model, n_heads=args.n_heads, n_enc_layers=args.n_layers,
            n_dec_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
            pos_encoding=args.pos_encoding, norm_type=args.norm_type,
            batch_size=args.batch_size, learning_rate=args.lr, warmup_steps=args.warmup_steps,
            scheduler_type=args.scheduler_type,
            epochs=args.epochs, label_smoothing=args.label_smoothing,
            src_embedding_path=args.src_embedding, tgt_embedding_path=args.tgt_embedding,
            embedding_binary=args.embedding_binary, freeze_embeddings=args.freeze_embeddings
        )
        train(config, args.vocab_dir, args.save_dir, args.checkpoint, args.device, args.src_lang, args.use_hunyuan, args.train_file)


if __name__ == "__main__":
    """
    ==================== 使用示例 ====================
    
    1. 基本训练（默认配置：绝对位置编码 + LayerNorm）
       python transformer_train.py --mode train --epochs 20
    
    2. 使用相对位置编码 + RMSNorm
       python transformer_train.py --mode train --pos_encoding relative --norm_type rmsnorm
    
    3. 使用预训练词向量（中文源语言）
       python transformer_train.py --mode train --d_model 200 \\
           --src_embedding embeddings/light_Tencent_AILab_ChineseEmbedding.bin \\
           --embedding_binary
    
    4. 运行消融实验（自动对比4种配置组合）
       python transformer_train.py --mode ablation --epochs 15
    
    5. 运行超参数研究（对比不同 batch_size, lr, model_size）
       python transformer_train.py --mode hyperparameter --epochs 10
    
    6. 评估模型
       python transformer_train.py --mode evaluate \\
           --checkpoint checkpoints/transformer_absolute_layernorm_d256_best.pt
    
    7. 交互式翻译
       python transformer_train.py --mode interactive \\
           --checkpoint checkpoints/transformer_absolute_layernorm_d256_best.pt \\
           --decode_method beam --beam_size 5
    
    8. 从 checkpoint 恢复训练
       python transformer_train.py --mode train \\
           --checkpoint checkpoints/transformer_absolute_layernorm_d256_best.pt \\
           --epochs 10
    
    ==================== 消融实验说明 ====================
    
    | 实验 | 位置编码 | 归一化 | 说明 |
    |-----|---------|-------|------|
    | 1   | absolute | layernorm | 标准 Transformer (Vaswani et al.) |
    | 2   | relative | layernorm | 相对位置偏置 (T5 风格) |
    | 3   | absolute | rmsnorm   | RMSNorm (LLaMA 风格) |
    | 4   | relative | rmsnorm   | 结合两种改进 |
    
    ==================== 超参数研究说明 ====================
    
    - Batch Size: 32, 64, 128 (影响训练稳定性和速度)
    - Learning Rate: 5e-4, 1e-3, 3e-3 (影响收敛速度和最终性能)
    - Model Size: d_model=128/256/512 (影响模型容量和推理速度)
    """
    main()
