"""
T5 微调脚本
- 基于 HuggingFace Transformers 库微调 T5 模型
- 支持中译英/英译中任务
- 与自建 Transformer 模型对比
"""
import os
import json
import time
import math
import argparse
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# HuggingFace
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    MT5Tokenizer, MT5ForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from datasets import Dataset as HFDataset

# 本地模块
from preprocess import load_jsonl, DataCleaner


# ==================== 配置 ====================

@dataclass
class T5Config:
    """T5 微调配置"""
    model_name: str = "google/mt5-small"  # 支持中文的 mT5
    max_src_len: int = 128
    max_tgt_len: int = 128
    
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 5
    
    gradient_accumulation_steps: int = 4
    fp16: bool = True  # 混合精度训练
    
    src_lang: str = 'zh'  # 源语言


# ==================== 数据集 ====================

class T5TranslationDataset(Dataset):
    """T5 翻译数据集"""
    def __init__(self, data: List[Dict], tokenizer, max_src_len: int, max_tgt_len: int, 
                 src_lang: str = 'zh', use_prefix: bool = True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.src_lang = src_lang
        self.use_prefix = use_prefix
        
        # 翻译任务前缀 - 对于 mT5，使用更简洁的格式或不使用前缀
        # mT5 没有预训练翻译任务，所以前缀主要用于区分任务类型
        if use_prefix:
            if src_lang == 'zh':
                self.prefix = "翻译成英文: "  # 使用中文前缀，mT5 更容易理解
            else:
                self.prefix = "翻译成中文: "
        else:
            self.prefix = ""
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        if self.src_lang == 'zh':
            src_text = self.prefix + item['zh']
            tgt_text = item['en']
        else:
            src_text = self.prefix + item['en']
            tgt_text = item['zh']
        
        # Tokenize
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        tgt_encoding = self.tokenizer(
            tgt_text,
            max_length=self.max_tgt_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = tgt_encoding['input_ids'].squeeze()
        # T5 使用 -100 忽略 padding token
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': src_encoding['input_ids'].squeeze(),
            'attention_mask': src_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def load_data(train_file: str, valid_file: str, test_file: str = None):
    """加载并清洗数据"""
    cleaner = DataCleaner()
    
    def process_file(path):
        if not path or not os.path.exists(path):
            return []
        data = []
        for item in load_jsonl(path):
            result = cleaner.process_pair(item.get('en', ''), item.get('zh', ''))
            if result:
                data.append({'en': result[0], 'zh': result[1]})
        return data
    
    train_data = process_file(train_file)
    valid_data = process_file(valid_file)
    test_data = process_file(test_file) if test_file else []
    
    print(f"数据量: 训练={len(train_data)}, 验证={len(valid_data)}, 测试={len(test_data)}")
    return train_data, valid_data, test_data


# ==================== BLEU 计算 ====================

def compute_bleu(refs: List[List[str]], hyps: List[str], max_n: int = 4) -> float:
    """计算 BLEU-4 分数"""
    from collections import Counter
    
    def ngrams(tokens, n):
        if n > 1:
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        return tokens
    
    precs = []
    for n in range(1, max_n + 1):
        match, total = 0.0, 0.0
        for ref_list, hyp in zip(refs, hyps):
            hyp_tokens = hyp.split() if isinstance(hyp, str) else hyp
            hyp_cnt = Counter(ngrams(hyp_tokens, n))
            max_ref = Counter()
            for ref in ref_list:
                ref_tokens = ref.split() if isinstance(ref, str) else ref
                for ng, c in Counter(ngrams(ref_tokens, n)).items():
                    max_ref[ng] = max(max_ref.get(ng, 0), c)
            match += sum(min(c, max_ref.get(ng, 0)) for ng, c in hyp_cnt.items())
            total += max(len(hyp_tokens) - n + 1, 0)
        precs.append(match / total if total > 0 else 0)
    
    # Brevity Penalty
    bp_c, bp_r = 0.0, 0.0
    for ref_list, hyp in zip(refs, hyps):
        hyp_tokens = hyp.split() if isinstance(hyp, str) else hyp
        bp_c += len(hyp_tokens)
        bp_r += min((abs(len(ref.split() if isinstance(ref, str) else ref) - len(hyp_tokens)), 
                     len(ref.split() if isinstance(ref, str) else ref)) for ref in ref_list)[1]
    bp = math.exp(1 - bp_r / bp_c) if bp_c <= bp_r and bp_c > 0 else 1.0
    
    geo = math.exp(sum(math.log(p) for p in precs) / len(precs)) if min(precs) > 0 else 0
    return bp * geo * 100


# ==================== 训练函数 ====================

def train_epoch(model, loader, optimizer, scheduler, device, gradient_accumulation_steps=1):
    model.train()
    total_loss = 0.0
    step_losses = []
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training")
    for step, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        batch_loss = outputs.loss.item()
        total_loss += batch_loss
        step_losses.append(batch_loss)
        pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
    
    return total_loss / len(loader), step_losses


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
    
    return total_loss / len(loader)


def calculate_bleu_t5(model, tokenizer, data: List[Dict], device, 
                      max_samples: int = 500, src_lang: str = 'zh',
                      num_beams: int = 4, use_prefix: bool = True, batch_size: int = 16):
    """计算 T5 模型的 BLEU 分数（批量生成优化版）"""
    model.eval()
    refs, hyps = [], []
    
    # 使用与训练一致的前缀
    if use_prefix:
        prefix = "翻译成英文: " if src_lang == 'zh' else "翻译成中文: "
    else:
        prefix = ""
    
    # **优化1: 预先构建 bad_words_ids，避免重复计算**
    bad_words_ids = []
    vocab = tokenizer.get_vocab()
    for i in range(100):
        token = f"<extra_id_{i}>"
        if token in vocab:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                bad_words_ids.append([token_id])
    
    # 构建 generate 参数
    gen_kwargs = {
        "max_length": 128,
        "num_beams": num_beams,
        "early_stopping": True,
        "decoder_start_token_id": tokenizer.pad_token_id,
    }
    
    if bad_words_ids:
        gen_kwargs["bad_words_ids"] = bad_words_ids
    
    # **优化2: 批量处理**
    data_subset = data[:max_samples]
    num_batches = (len(data_subset) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Computing BLEU"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(data_subset))
            batch_items = data_subset[start_idx:end_idx]
            
            # 准备批量输入
            src_texts = []
            ref_texts = []
            for item in batch_items:
                if src_lang == 'zh':
                    src_texts.append(prefix + item['zh'])
                    ref_texts.append(item['en'])
                else:
                    src_texts.append(prefix + item['en'])
                    ref_texts.append(item['zh'])
            
            # 批量 tokenize
            inputs = tokenizer(src_texts, return_tensors='pt', max_length=128, 
                             truncation=True, padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 批量生成
            outputs = model.generate(**inputs, **gen_kwargs)
            
            # 解码
            for i, output in enumerate(outputs):
                hyp = tokenizer.decode(output, skip_special_tokens=True)
                refs.append([ref_texts[i]])
                hyps.append(hyp)
    
    return compute_bleu(refs, hyps)


def translate_t5(model, tokenizer, text: str, device, src_lang: str = 'zh', 
                  num_beams: int = 4, use_prefix: bool = True):
    """使用 T5 进行翻译"""
    model.eval()
    
    # 使用与训练一致的前缀
    if use_prefix:
        prefix = "翻译成英文: " if src_lang == 'zh' else "翻译成中文: "
    else:
        prefix = ""
    src_text = prefix + text
    
    inputs = tokenizer(src_text, return_tensors='pt', max_length=128, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 构建 bad_words_ids 以禁止生成 sentinel tokens
    bad_words_ids = []
    vocab = tokenizer.get_vocab()
    for i in range(100):
        token = f"<extra_id_{i}>"
        if token in vocab:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:  # 确保是有效的 token
                bad_words_ids.append([token_id])
    
    # 构建 generate 参数
    gen_kwargs = {
        "max_length": 128,
        "num_beams": num_beams,
        "early_stopping": True,
        "decoder_start_token_id": tokenizer.pad_token_id,
    }
    
    # 只有当 bad_words_ids 非空时才添加
    if bad_words_ids:
        gen_kwargs["bad_words_ids"] = bad_words_ids
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ==================== 主训练函数 ====================

def train_t5(config: T5Config, train_file: str = "data/train_100k.jsonl",
             valid_file: str = "data/valid.jsonl", test_file: str = "data/test.jsonl",
             save_dir: str = "checkpoints", checkpoint: str = None):
    """T5 微调主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    train_data, valid_data, test_data = load_data(train_file, valid_file, test_file)
    
    # 加载 tokenizer 和模型
    print(f"加载模型: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    if checkpoint and os.path.exists(checkpoint):
        print(f"从 {checkpoint} 恢复...")
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    
    model = model.to(device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 创建数据集
    train_dataset = T5TranslationDataset(train_data, tokenizer, config.max_src_len, 
                                         config.max_tgt_len, config.src_lang)
    valid_dataset = T5TranslationDataset(valid_data, tokenizer, config.max_src_len,
                                         config.max_tgt_len, config.src_lang)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # 优化器和调度器
    optimizer = torch.optim.Adafactor(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    )
    
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # 混合精度
    scaler = torch.amp.GradScaler() if config.fp16 and torch.cuda.is_available() else None
    
    # 训练循环
    exp_name = f"t5_{config.model_name.split('/')[-1]}_{config.src_lang}2{'en' if config.src_lang == 'zh' else 'zh'}"
    best_loss = float('inf')
    history = {'train_loss': [], 'valid_loss': [], 'bleu': [], 'step_losses': []}
    
    # ========== 初始评估（未微调） ==========
    print("\n" + "="*60)
    print("初始评估 (未微调的预训练模型)")
    print("="*60)
    
    # 计算初始验证 loss
    initial_valid_loss = evaluate(model, valid_loader, device)
    print(f"初始验证 Loss: {initial_valid_loss:.4f}")
    
    # 计算初始 BLEU
    initial_bleu = calculate_bleu_t5(model, tokenizer, valid_data, device, 
                                     max_samples=200, src_lang=config.src_lang,
                                     batch_size=config.batch_size)
    print(f"初始 BLEU: {initial_bleu:.2f}")
    
    # 翻译示例
    sample = "今天天气很好。" if config.src_lang == 'zh' else "The weather is nice today."
    initial_translation = translate_t5(model, tokenizer, sample, device, config.src_lang)
    print(f"翻译示例: '{sample}' → '{initial_translation}'")
    
    # 保存初始指标到历史记录
    history['initial_valid_loss'] = initial_valid_loss
    history['initial_bleu'] = initial_bleu
    history['bleu'].append(initial_bleu)  # 加入 BLEU 历史，作为 epoch 0
    
    print("\n" + "="*60)
    print("开始微调训练")
    print("="*60 + "\n")
    
    for epoch in range(config.epochs):
        t0 = time.time()
        
        # 训练
        train_loss, step_losses = train_epoch(
            model, train_loader, optimizer, scheduler, device, 
            config.gradient_accumulation_steps
        )
        history['step_losses'].extend(step_losses)
        
        # 验证
        valid_loss = evaluate(model, valid_loader, device)
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        
        mins, secs = int((time.time() - t0) / 60), int((time.time() - t0) % 60)
        print(f"Epoch {epoch+1}/{config.epochs} | {mins}m{secs}s | "
              f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        
        # 计算 BLEU
        if (epoch + 1) % 2 == 0 or epoch == config.epochs - 1:
            bleu = calculate_bleu_t5(model, tokenizer, valid_data, device, 
                                     max_samples=200, src_lang=config.src_lang,
                                     batch_size=config.batch_size)
            history['bleu'].append(bleu)
            
            # 翻译示例
            sample = "今天天气很好。" if config.src_lang == 'zh' else "The weather is nice today."
            translation = translate_t5(model, tokenizer, sample, device, config.src_lang)
            print(f"  BLEU: {bleu:.2f}")
            print(f"  示例: '{sample}' → '{translation}'")
        
        # 保存最佳模型
        if valid_loss < best_loss:
            best_loss = valid_loss
            model.save_pretrained(save_dir / f"{exp_name}_best")
            tokenizer.save_pretrained(save_dir / f"{exp_name}_best")
            print("  ✓ 保存最佳模型")
    
    # 保存历史
    with open(save_dir / f"{exp_name}_history.json", 'w') as f:
        json.dump(history, f)
    
    # ========== 训练总结 ==========
    print("\n" + "="*60)
    print("训练完成总结")
    print("="*60)
    print(f"初始状态 (未微调):")
    print(f"  验证 Loss: {history['initial_valid_loss']:.4f}")
    print(f"  BLEU 分数: {history['initial_bleu']:.2f}")
    print(f"\n最终状态 (微调后):")
    print(f"  最佳验证 Loss: {best_loss:.4f} (↓ {history['initial_valid_loss'] - best_loss:.4f})")
    print(f"  最终 BLEU: {history['bleu'][-1]:.2f} (↑ {history['bleu'][-1] - history['initial_bleu']:.2f})")
    
    # 计算改进百分比
    loss_improve = ((history['initial_valid_loss'] - best_loss) / history['initial_valid_loss']) * 100
    bleu_improve = ((history['bleu'][-1] - history['initial_bleu']) / max(history['initial_bleu'], 0.01)) * 100
    print(f"\n改进:")
    print(f"  Loss 降低: {loss_improve:.1f}%")
    print(f"  BLEU 提升: {bleu_improve:.1f}%")
    print("="*60 + "\n")
    
    return model, tokenizer, history


# ==================== 模型对比 ====================

def compare_models(t5_checkpoint: str, transformer_checkpoint: str,
                   test_file: str = "data/test.jsonl", vocab_dir: str = "vocab",
                   src_lang: str = 'zh'):
    """对比 T5 和自建 Transformer 模型"""
    from transformer_train import Transformer, TransformerConfig, translate, calculate_bleu
    from preprocess import load_all, create_dataloader, Config
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载测试数据
    cleaner = DataCleaner()
    test_data = []
    for item in load_jsonl(test_file):
        result = cleaner.process_pair(item.get('en', ''), item.get('zh', ''))
        if result:
            test_data.append({'en': result[0], 'zh': result[1]})
    
    print(f"测试数据量: {len(test_data)}")
    
    results = {}
    
    # ========== T5 评估 ==========
    print("\n" + "="*50)
    print("评估 T5 模型")
    print("="*50)
    
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_checkpoint)
    t5_model = AutoModelForSeq2SeqLM.from_pretrained(t5_checkpoint).to(device)
    
    t5_bleu = calculate_bleu_t5(t5_model, t5_tokenizer, test_data, device, 
                                 max_samples=500, src_lang=src_lang, batch_size=32)
    results['T5'] = {'BLEU': t5_bleu}
    print(f"T5 BLEU: {t5_bleu:.2f}")
    
    # 示例翻译
    samples = ["今天天气很好。", "我喜欢学习。", "机器翻译是一个有趣的任务。"] if src_lang == 'zh' else \
              ["The weather is nice.", "I love learning.", "Machine translation is interesting."]
    
    print("\nT5 翻译示例:")
    for s in samples:
        print(f"  '{s}' → '{translate_t5(t5_model, t5_tokenizer, s, device, src_lang)}'")
    
    # ========== 自建 Transformer 评估 ==========
    print("\n" + "="*50)
    print("评估自建 Transformer 模型")
    print("="*50)
    
    # 加载 checkpoint
    ckpt = torch.load(transformer_checkpoint, map_location=device)
    cfg_dict = ckpt['config']
    config = TransformerConfig(**cfg_dict)
    
    # 加载 tokenizer
    tokenizer, _, _, test_ds = load_all(vocab_dir, src_lang=src_lang)
    
    if src_lang == 'zh':
        src_vocab_size, tgt_vocab_size = len(tokenizer.zh_vocab), len(tokenizer.en_vocab)
    else:
        src_vocab_size, tgt_vocab_size = len(tokenizer.en_vocab), len(tokenizer.zh_vocab)
    
    # 创建模型
    transformer_model = Transformer(src_vocab_size, tgt_vocab_size, config).to(device)
    transformer_model.load_state_dict(ckpt['model_state_dict'])
    
    # 计算 BLEU
    if test_ds:
        test_loader = create_dataloader(test_ds, config.batch_size, shuffle=False)
        transformer_bleu = calculate_bleu(transformer_model, test_loader, tokenizer, device,
                                          'beam', max_samples=500, src_lang=src_lang)
    else:
        transformer_bleu = 0.0
    
    results['Transformer'] = {'BLEU': transformer_bleu}
    print(f"Transformer BLEU: {transformer_bleu:.2f}")
    
    print("\nTransformer 翻译示例:")
    for s in samples:
        print(f"  '{s}' → '{translate(transformer_model, tokenizer, s, device, 'beam', src_lang=src_lang)}'")
    
    # ========== 结果对比 ==========
    print("\n" + "="*50)
    print("模型对比结果")
    print("="*50)
    print(f"{'模型':<20} {'BLEU':<10}")
    print("-"*30)
    for name, res in results.items():
        print(f"{name:<20} {res['BLEU']:.2f}")
    
    winner = max(results.items(), key=lambda x: x[1]['BLEU'])
    print(f"\n最佳模型: {winner[0]} (BLEU={winner[1]['BLEU']:.2f})")
    
    return results


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(description="T5 微调脚本")
    
    parser.add_argument('--mode', default='train', 
                        choices=['train', 'evaluate', 'interactive', 'compare'],
                        help='运行模式')
    
    # 模型配置
    parser.add_argument('--model_name', default='google/mt5-small',
                        help='预训练模型名称 (mt5-small, mt5-base, t5-small 等)')
    parser.add_argument('--max_src_len', type=int, default=256)
    parser.add_argument('--max_tgt_len', type=int, default=256)
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--fp16', action='store_true', default=True)
    
    # 数据和路径
    parser.add_argument('--train_file', default='data/train_100k.jsonl')
    parser.add_argument('--valid_file', default='data/valid.jsonl')
    parser.add_argument('--test_file', default='data/test.jsonl')
    parser.add_argument('--save_dir', default='checkpoints')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--vocab_dir', default='vocab')
    
    # 翻译方向
    parser.add_argument('--src_lang', default='zh', choices=['zh', 'en'])
    
    # 对比实验
    parser.add_argument('--t5_checkpoint', default=None, help='T5 checkpoint 路径')
    parser.add_argument('--transformer_checkpoint', default=None, help='Transformer checkpoint 路径')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        config = T5Config(
            model_name=args.model_name,
            max_src_len=args.max_src_len,
            max_tgt_len=args.max_tgt_len,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            epochs=args.epochs,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16=args.fp16,
            src_lang=args.src_lang
        )
        train_t5(config, args.train_file, args.valid_file, args.test_file, 
                 args.save_dir, args.checkpoint)
    
    elif args.mode == 'evaluate':
        if not args.checkpoint:
            print("需要 --checkpoint")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
        
        # 加载测试数据
        cleaner = DataCleaner()
        test_data = []
        for item in load_jsonl(args.test_file):
            result = cleaner.process_pair(item.get('en', ''), item.get('zh', ''))
            if result:
                test_data.append({'en': result[0], 'zh': result[1]})
        
        bleu = calculate_bleu_t5(model, tokenizer, test_data, device, 
                                  max_samples=500, src_lang=args.src_lang, batch_size=32)
        print(f"BLEU: {bleu:.2f}")
        
        samples = ["今天天气很好。", "我喜欢学习。", "这是一个测试。"] if args.src_lang == 'zh' else \
                  ["The weather is nice.", "I love learning.", "This is a test."]
        print("\n翻译示例:")
        for s in samples:
            print(f"  '{s}' → '{translate_t5(model, tokenizer, s, device, args.src_lang)}'")
    
    elif args.mode == 'interactive':
        if not args.checkpoint:
            print("需要 --checkpoint")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint).to(device)
        
        src_name = "中文" if args.src_lang == 'zh' else "英文"
        tgt_name = "英文" if args.src_lang == 'zh' else "中文"
        print(f"\nT5 交互式翻译 ({src_name}→{tgt_name}, 输入'quit'退出)")
        
        while True:
            try:
                text = input(f"\n{src_name}: ").strip()
                if text.lower() == 'quit':
                    break
                if text:
                    result = translate_t5(model, tokenizer, text, device, args.src_lang)
                    print(f"{tgt_name}: {result}")
            except KeyboardInterrupt:
                break
    
    elif args.mode == 'compare':
        if not args.t5_checkpoint or not args.transformer_checkpoint:
            print("需要 --t5_checkpoint 和 --transformer_checkpoint")
            return
        compare_models(args.t5_checkpoint, args.transformer_checkpoint,
                      args.test_file, args.vocab_dir, args.src_lang)


if __name__ == "__main__":
    main()

