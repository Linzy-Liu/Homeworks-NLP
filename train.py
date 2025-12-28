"""机器翻译模型训练脚本"""
import os, json, time, math, argparse
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from preprocess import Config, TranslationTokenizer, create_dataloader, load_all, check_preprocessed_exists, preprocess_pipeline
from model import create_model, Seq2Seq


class TrainConfig:
    EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT = 200, 400, 2, 0.4
    CLIP_GRAD, PATIENCE = 1.0, 10
    
    # 预训练词向量配置 (设为 None 则不使用)
    EN_EMBEDDING_PATH = None  # 英文词向量路径, 如 "embeddings/glove.6B.200d.txt"
    EN_EMBEDDING_BINARY = False  # 英文词向量是否为二进制格式
    ZH_EMBEDDING_PATH = "embeddings/light_Tencent_AILab_ChineseEmbedding.bin"  # 腾讯中文词向量
    ZH_EMBEDDING_BINARY = True  # 中文词向量是否为二进制格式 (.bin)
    FREEZE_EMBEDDINGS = False  # 是否冻结词向量 (不参与训练)


def get_device(device_str=None):
    if device_str: 
        return torch.device(device_str)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loader, optimizer, criterion, clip, tf_ratio, device):
    model.train()
    total_loss, count = 0.0, 0
    step_losses = []
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        src, tgt = batch['src_ids'].to(device), batch['tgt_ids'].to(device)
        optimizer.zero_grad()
        
        # return_attention=False 节省显存
        outputs, _ = model(src, tgt, batch['src_lengths'], tf_ratio, return_attention=False)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        # 记录并显示当前 loss
        batch_loss = loss.item()
        total_loss += batch_loss * src.size(0)
        count += src.size(0)
        step_losses.append(batch_loss)
        pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg': f'{total_loss/count:.4f}'})
        
        # 清理显存（每个batch后释放中间变量）
        del outputs, loss
        
    return total_loss / count, step_losses


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, count = 0.0, 0
    pbar = tqdm(loader, desc="Evaluating")
    with torch.no_grad():
        for batch in pbar:
            src, tgt = batch['src_ids'].to(device), batch['tgt_ids'].to(device)
            outputs, _ = model(src, tgt, batch['src_lengths'], 0.0, return_attention=False)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            batch_loss = loss.item()
            total_loss += batch_loss * src.size(0)
            count += src.size(0)
            pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg': f'{total_loss/count:.4f}'})
    
    return total_loss / count


def compute_bleu(refs, hyps, max_n=4):
    """
    计算 BLEU-4 分数
    
    BLEU-4 = BP * exp(1/4 * sum(log(p_n) for n in 1..4))
    其中:
        - p_n: n-gram 精确率 (clipped precision)
        - BP: 短句惩罚因子 (brevity penalty)
    
    Args:
        refs: 参考翻译列表，每个元素是 [ref_tokens_list, ...]（支持多参考）
        hyps: 模型翻译列表，每个元素是 hyp_tokens_list
        max_n: 最大 n-gram，默认4（即 BLEU-4）
    
    Returns:
        BLEU-4 分数（百分制，0-100）
    """
    from collections import Counter
    def ngrams(t, n): 
        if n > 1:
            return [tuple(t[i:i+n]) for i in range(len(t) - n + 1)]
        else:
            return t
    
    # 计算 1-gram 到 max_n-gram 的精确率
    precs, bp_c, bp_r = [], 0.0, 0.0
    for n in range(1, max_n + 1):
        match, total = 0.0, 0.0
        for ref_list, hyp in zip(refs, hyps):
            hyp_cnt = Counter(ngrams(hyp, n))
            # 取所有参考翻译中 n-gram 的最大计数
            max_ref = Counter()
            for ref in ref_list:
                for ng, c in Counter(ngrams(ref, n)).items():
                    max_ref[ng] = max(max_ref.get(ng, 0), c)
            # Clipped count: min(候选计数, 参考最大计数)
            match += sum(min(c, max_ref.get(ng, 0)) for ng, c in hyp_cnt.items())
            total += max(len(hyp) - n + 1, 0)
        precs.append(match / total if total > 0 else 0)
    
    # 计算 Brevity Penalty (BP)
    for ref_list, hyp in zip(refs, hyps):
        bp_c += len(hyp)  # 候选翻译总长度
        bp_r += min((abs(len(r) - len(hyp)), len(r)) for r in ref_list)[1]  # 最接近的参考长度
    bp = math.exp(1 - bp_r / bp_c) if bp_c <= bp_r and bp_c > 0 else 1.0
    
    # BLEU-4 = BP * 几何平均(p1, p2, p3, p4)
    geo = math.exp(sum(math.log(p) for p in precs) / len(precs)) if min(precs) > 0 else 0
    return bp * geo * 100  # 返回百分制


def calculate_bleu(model, loader, tokenizer, device, decode='greedy', beam_size=5, max_samples=500, src_lang='zh'):
    """计算 BLEU 分数（适配中译英或英译中）"""
    model.eval()
    refs, hyps, count = [], [], 0
    # 根据源语言确定目标词表
    tgt_vocab = tokenizer.en_vocab if src_lang == 'zh' else tokenizer.zh_vocab
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"BLEU ({decode})"):
            if count >= max_samples: break
            src, tgt = batch['src_ids'].to(device), batch['tgt_ids']
            if decode == 'greedy':
                preds, _ = model.greedy_decode(src, batch['src_lengths'], 100, tgt_vocab.bos_idx, tgt_vocab.eos_idx)
            else:
                preds, _ = model.beam_search_decode(src, batch['src_lengths'], 100, beam_size, tgt_vocab.bos_idx, tgt_vocab.eos_idx)
            for i in range(src.size(0)):
                if count >= max_samples: break
                ref = [t for t in tgt_vocab.ids_to_tokens(tgt[i].tolist()) if t not in Config.SPECIAL_TOKENS]
                hyp = [t for t in tgt_vocab.ids_to_tokens(preds[i].tolist()) if t not in Config.SPECIAL_TOKENS]
                refs.append([ref])
                hyps.append(hyp)
                count += 1
    return compute_bleu(refs, hyps)


def translate(model, tokenizer, text, device, decode='greedy', beam_size=5, src_lang='zh'):
    """翻译函数（支持中译英或英译中）
    
    Args:
        src_lang: 源语言，'zh'=中译英，'en'=英译中
    """
    model.eval()
    # 根据源语言选择编码/解码函数
    if src_lang == 'zh':
        ids = tokenizer.encode_chinese(text)
        tgt_vocab = tokenizer.en_vocab
        decode_fn = tokenizer.decode_english
    else:
        ids = tokenizer.encode_english(text)
        tgt_vocab = tokenizer.zh_vocab
        decode_fn = tokenizer.decode_chinese
    
    src = torch.tensor([ids], device=device)
    with torch.no_grad():
        if decode == 'greedy':
            preds, _ = model.greedy_decode(src, torch.tensor([len(ids)]), 100, tgt_vocab.bos_idx, tgt_vocab.eos_idx)
        else:
            preds, _ = model.beam_search_decode(src, torch.tensor([len(ids)]), 100, beam_size, tgt_vocab.bos_idx, tgt_vocab.eos_idx)
    return decode_fn(preds[0].tolist())


def train(attention_type='additive', tf_ratio=1.0, tf_decay=0.0, rnn_type='GRU',
          num_epochs=20, batch_size=64, eval_batch_size=None, lr=1e-3, save_dir="checkpoints", vocab_dir="vocab",
          checkpoint=None, device=None, src_lang='zh', use_hunyuan=False, train_file='data/train_10k.jsonl'):
    """主训练函数，支持从checkpoint恢复
    
    Args:
        src_lang: 源语言，'zh'=中译英（默认），'en'=英译中
        use_hunyuan: 是否使用混元重译数据 (zh_hy 字段)
        train_file: 训练数据文件路径
    """
    device = get_device(device)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据（中译英：src=zh, tgt=en）
    if check_preprocessed_exists(vocab_dir):
        tokenizer, train_ds, valid_ds, _ = load_all(vocab_dir, src_lang=src_lang, use_hunyuan=use_hunyuan)
    else:
        tokenizer, train_ds, valid_ds, _ = preprocess_pipeline(train_file=train_file, save_dir=vocab_dir, src_lang=src_lang, use_hunyuan=use_hunyuan)
    
    if eval_batch_size is None:
        eval_batch_size = batch_size
    train_loader = create_dataloader(train_ds, batch_size, shuffle=True)
    valid_loader = create_dataloader(valid_ds, eval_batch_size, shuffle=False) if valid_ds else None
    
    # 根据翻译方向确定词表大小
    # 中译英: src=中文, tgt=英文
    # 英译中: src=英文, tgt=中文
    if src_lang == 'zh':
        src_vocab_size, tgt_vocab_size = len(tokenizer.zh_vocab), len(tokenizer.en_vocab)
        src_vocab, tgt_vocab = tokenizer.zh_vocab, tokenizer.en_vocab
    else:
        src_vocab_size, tgt_vocab_size = len(tokenizer.en_vocab), len(tokenizer.zh_vocab)
        src_vocab, tgt_vocab = tokenizer.en_vocab, tokenizer.zh_vocab
    
    # 创建或加载模型
    start_epoch, best_loss = 0, float('inf')
    if checkpoint and os.path.exists(checkpoint):
        print(f"从 {checkpoint} 恢复训练...")
        ckpt = torch.load(checkpoint, map_location=device)
        cfg = ckpt['config']
        attention_type, rnn_type = cfg['attention_type'], cfg['rnn_type']
        tf_ratio = cfg['teacher_forcing_ratio']
        src_lang = cfg.get('src_lang', 'zh')  # 兼容旧版本
        model = create_model(cfg['src_vocab_size'], cfg['tgt_vocab_size'], cfg['embed_dim'], cfg['hidden_dim'],
                           cfg['num_layers'], TrainConfig.DROPOUT, rnn_type, attention_type).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt['valid_loss']
    else:
        model = create_model(src_vocab_size, tgt_vocab_size, TrainConfig.EMBED_DIM,
                           TrainConfig.HIDDEN_DIM, TrainConfig.NUM_LAYERS, TrainConfig.DROPOUT,
                           rnn_type, attention_type).to(device)
        
        # 加载预训练词向量 (如果配置了)
        # 中译英: encoder=中文词向量, decoder=英文词向量
        if TrainConfig.ZH_EMBEDDING_PATH or TrainConfig.EN_EMBEDDING_PATH:
            from embeddings import load_pretrained_vectors
            
            # 源语言词向量 (encoder)
            if src_lang == 'zh':
                src_emb_path, src_binary = TrainConfig.ZH_EMBEDDING_PATH, TrainConfig.ZH_EMBEDDING_BINARY
            else:
                src_emb_path, src_binary = TrainConfig.EN_EMBEDDING_PATH, TrainConfig.EN_EMBEDDING_BINARY
            
            if src_emb_path and os.path.exists(src_emb_path):
                src_weights, _, _ = load_pretrained_vectors(src_emb_path, src_vocab.token2idx, TrainConfig.EMBED_DIM, binary=src_binary)
                model.encoder.load_pretrained_embeddings(src_weights.to(device), TrainConfig.FREEZE_EMBEDDINGS)
            
            # 目标语言词向量 (decoder)
            if src_lang == 'zh':
                tgt_emb_path, tgt_binary = TrainConfig.EN_EMBEDDING_PATH, TrainConfig.EN_EMBEDDING_BINARY
            else:
                tgt_emb_path, tgt_binary = TrainConfig.ZH_EMBEDDING_PATH, TrainConfig.ZH_EMBEDDING_BINARY
            
            if tgt_emb_path and os.path.exists(tgt_emb_path):
                tgt_weights, _, _ = load_pretrained_vectors(tgt_emb_path, tgt_vocab.token2idx, TrainConfig.EMBED_DIM, binary=tgt_binary)
                model.decoder.load_pretrained_embeddings(tgt_weights.to(device), TrainConfig.FREEZE_EMBEDDINGS)
    
    exp_name = f"{rnn_type}_{attention_type}_tf{tf_ratio}" + (f"_decay{tf_decay}" if tf_decay > 0 else "")
    print(f"实验: {exp_name}, 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"翻译方向: {'中文→英文' if src_lang == 'zh' else '英文→中文'}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 2)
    
    if checkpoint and os.path.exists(checkpoint):
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    
    config = {'attention_type': attention_type, 'rnn_type': rnn_type, 'teacher_forcing_ratio': tf_ratio,
              'embed_dim': TrainConfig.EMBED_DIM, 'hidden_dim': TrainConfig.HIDDEN_DIM,
              'num_layers': TrainConfig.NUM_LAYERS, 'src_vocab_size': src_vocab_size,
              'tgt_vocab_size': tgt_vocab_size, 'src_lang': src_lang}
    
    patience_cnt, curr_tf = 0, tf_ratio
    history = {'train_loss': [], 'valid_loss': [], 'bleu_greedy': [], 'bleu_beam': [], 'step_losses': []}
    
    # 训练循环
    for epoch in range(start_epoch, start_epoch + num_epochs):
        t0 = time.time()
        model.train()
        total_loss, count = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        # 确定评估间隔 (每 20% steps)
        steps_per_epoch = len(train_loader)
        eval_interval = max(1, steps_per_epoch // 4)
        
        for step, batch in enumerate(pbar):
            src, tgt = batch['src_ids'].to(device), batch['tgt_ids'].to(device)
            optimizer.zero_grad()
            
            outputs, _ = model(src, tgt, batch['src_lengths'], curr_tf)
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            
            nn.utils.clip_grad_norm_(model.parameters(), TrainConfig.CLIP_GRAD)
            optimizer.step()
            
            # 记录 Loss
            batch_loss = loss.item()
            total_loss += batch_loss * src.size(0)
            count += src.size(0)
            history['step_losses'].append(batch_loss)
            pbar.set_postfix({'loss': f'{batch_loss:.4f}', 'avg': f'{total_loss/count:.4f}'})
            del outputs, loss
            
            # 步内评估 (每 25%) - 仅用于监控，不影响保存逻辑
            if valid_loader and (step + 1) % eval_interval == 0:
                val_loss = evaluate(model, valid_loader, criterion, device)
                model.train() # 切回训练模式
                pbar.write(f"  [Step {step+1}] Valid Loss: {val_loss:.4f} | Train Avg: {total_loss/count:.4f}")

        # Epoch 结束后的常规评估
        train_loss = total_loss / count
        valid_loss = evaluate(model, valid_loader, criterion, device) if valid_loader else train_loss
        scheduler.step(train_loss)  # 调度器也基于 train_loss

        # 针对高显存峰值的主动清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        
        mins, secs = int((time.time() - t0) / 60), int((time.time() - t0) % 60)
        print(f"Epoch {epoch+1} Summary | {mins}m{secs}s | Train: {train_loss:.4f} | Valid: {valid_loss:.4f} | TF: {curr_tf:.2f}")
        
        # 每个 epoch 都计算 BLEU 分数
        if valid_loader:
            bleu_g = calculate_bleu(model, valid_loader, tokenizer, device, 'greedy', max_samples=200, src_lang=src_lang)
            bleu_b = calculate_bleu(model, valid_loader, tokenizer, device, 'beam', 5, max_samples=100, src_lang=src_lang)
            history['bleu_greedy'].append(bleu_g)
            history['bleu_beam'].append(bleu_b)
            # 翻译示例：中译英
            sample_src = "今天天气很好。" if src_lang == 'zh' else "The weather is nice today."
            sample_tgt = translate(model, tokenizer, sample_src, device, 'greedy', src_lang=src_lang)
            print(f"  BLEU: Greedy={bleu_g:.2f}, Beam={bleu_b:.2f}")
            print(f"  翻译示例: '{sample_src}' → '{sample_tgt}'")
        
        # Epoch 结束时的保存逻辑 (基于 train_loss，因为 valid/train 分布差异大)
        if train_loss < best_loss:
            best_loss, patience_cnt = train_loss, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(), 
                       'train_loss': train_loss, 'valid_loss': valid_loss, 'config': config},
                      save_dir / f"{exp_name}_best.pt")
            print("  ✓ 保存最佳模型 (基于 Train Loss)")
        else:
            patience_cnt += 1
            if patience_cnt >= TrainConfig.PATIENCE:
                print(f"早停：{TrainConfig.PATIENCE} epochs 训练损失无改善")
                break
        
        curr_tf = max(0.0, curr_tf - tf_decay)
    
    with open(save_dir / f"{exp_name}_history.json", 'w') as f:
        json.dump(history, f)
    return model, history


def load_checkpoint(path, vocab_dir="vocab", device=None):
    """加载checkpoint用于推理"""
    device = get_device(device)
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt['config']
    src_lang = cfg.get('src_lang', 'zh')  # 兼容旧版本
    tokenizer, _, _, _ = load_all(vocab_dir, src_lang=src_lang)
    model = create_model(cfg['src_vocab_size'], cfg['tgt_vocab_size'], cfg['embed_dim'], cfg['hidden_dim'],
                        cfg['num_layers'], 0.0, cfg['rnn_type'], cfg['attention_type']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, tokenizer, cfg


def interactive_translate(checkpoint, vocab_dir="vocab", decode='greedy', beam_size=5, device=None):
    model, tokenizer, cfg = load_checkpoint(checkpoint, vocab_dir, device)
    device = get_device(device)
    src_lang = cfg.get('src_lang', 'zh')
    src_name = "中文" if src_lang == 'zh' else "英文"
    tgt_name = "英文" if src_lang == 'zh' else "中文"
    print(f"\n交互式翻译 ({src_name}→{tgt_name}, 输入'quit'退出, 解码={decode})")
    while True:
        try:
            text = input(f"\n{src_name}: ").strip()
            if text.lower() == 'quit': break
            if text: print(f"{tgt_name}: {translate(model, tokenizer, text, device, decode, beam_size, src_lang)}")
        except KeyboardInterrupt:
            break


def evaluate_checkpoint(checkpoint, vocab_dir="vocab", batch_size=64, device=None):
    model, tokenizer, cfg = load_checkpoint(checkpoint, vocab_dir, device)
    device = get_device(device)
    src_lang = cfg.get('src_lang', 'zh')
    _, _, valid_ds, test_ds = load_all(vocab_dir, src_lang=src_lang)
    ds = test_ds or valid_ds
    if not ds: return print("无评估数据")
    loader = create_dataloader(ds, batch_size, shuffle=False)
    print(f"翻译方向: {'中文→英文' if src_lang == 'zh' else '英文→中文'}")
    print(f"BLEU Greedy: {calculate_bleu(model, loader, tokenizer, device, 'greedy', max_samples=500, src_lang=src_lang):.2f}")
    print(f"BLEU Beam-5: {calculate_bleu(model, loader, tokenizer, device, 'beam', 5, max_samples=200, src_lang=src_lang):.2f}")
    
    # 测试翻译示例
    if src_lang == 'zh':
        samples = ["今天天气很好。", "我喜欢学习。", "这是一个测试。"]
    else:
        samples = ["The weather is nice.", "I love learning.", "This is a test."]
    
    print("\n翻译示例:")
    for s in samples:
        greedy_out = translate(model, tokenizer, s, device, 'greedy', src_lang=src_lang)
        beam_out = translate(model, tokenizer, s, device, 'beam', 5, src_lang=src_lang)
        print(f"  '{s}'")
        print(f"    Greedy: {greedy_out}")
        print(f"    Beam:   {beam_out}")


def compare_experiments(experiments, vocab_dir="vocab", save_dir="checkpoints", num_epochs=15, batch_size=64, src_lang='zh'):
    results = {}
    for i, cfg in enumerate(experiments):
        print(f"\n{'#'*50}\n# 实验 {i+1}/{len(experiments)}: {cfg}\n{'#'*50}")
        # 注意：这里我们只关心history，不关心model
        _, hist = train(cfg.get('attention_type', 'additive'), cfg.get('teacher_forcing_ratio', 1.0),
                       cfg.get('tf_decay', 0.0), cfg.get('rnn_type', 'GRU'), num_epochs, batch_size,
                       save_dir=save_dir, vocab_dir=vocab_dir, src_lang=src_lang)
        name = f"{cfg.get('rnn_type','GRU')}_{cfg.get('attention_type','additive')}_tf{cfg.get('teacher_forcing_ratio',1.0)}"
        results[name] = {'best_loss': min(hist['valid_loss']), 
                        'bleu_greedy': hist['bleu_greedy'][-1] if hist['bleu_greedy'] else None}
    
    print(f"\n{'='*60}\n实验对比\n{'='*60}")
    for n, r in results.items():
        print(f"{n:<35} Loss: {r['best_loss']:.4f} BLEU: {r['bleu_greedy'] or 'N/A'}")
    with open(Path(save_dir) / "comparison.json", 'w') as f:
        json.dump(results, f)
    return results


def main():
    """
    命令行入口函数
    
    使用示例:
        # 训练模型（默认中译英）
        python train.py --mode train --attention additive --epochs 20
        
        # 从checkpoint恢复训练
        python train.py --mode train --checkpoint checkpoints/xxx_best.pt --epochs 10
        
        # 运行对比实验（比较不同attention和TF策略）
        python train.py --mode compare --epochs 15
        
        # 评估模型性能
        python train.py --mode evaluate --checkpoint checkpoints/xxx_best.pt
        
        # 交互式翻译（中文→英文）
        python train.py --mode interactive --checkpoint checkpoints/xxx_best.pt --decode_method beam
    """
    parser = argparse.ArgumentParser(description="RNN-based NMT 训练脚本")
    
    # ===== 运行模式 =====
    parser.add_argument('--mode', default='train', choices=['train', 'compare', 'evaluate', 'interactive'],
                        help='运行模式: train=训练, compare=对比实验, evaluate=评估, interactive=交互翻译')
    
    # ===== 模型架构参数 =====
    parser.add_argument('--attention', default='additive', choices=['dot', 'multiplicative', 'additive'],
                        help='注意力机制类型: dot=点积, multiplicative=乘性(Luong), additive=加性(Bahdanau)')
    parser.add_argument('--rnn_type', default='GRU', choices=['GRU', 'LSTM'],
                        help='RNN单元类型')
    
    # ===== 训练超参数 =====
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批大小')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='验证/评估批大小（默认同训练批大小）')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--tf_ratio', type=float, default=1.0,
                        help='Teacher Forcing比例: 1.0=完全TF, 0.0=完全Free Running')
    parser.add_argument('--tf_decay', type=float, default=0.0,
                        help='每个epoch TF比例的衰减量，用于渐进式训练策略')
    
    # ===== 路径配置 =====
    parser.add_argument('--vocab_dir', default='vocab',
                        help='词表和预处理数据目录')
    parser.add_argument('--save_dir', default='checkpoints',
                        help='模型checkpoint保存目录')
    parser.add_argument('--checkpoint', default=None,
                        help='checkpoint路径，用于恢复训练或推理')
    
    # ===== 解码配置（用于evaluate/interactive模式）=====
    parser.add_argument('--decode_method', default='greedy', choices=['greedy', 'beam'],
                        help='解码策略: greedy=贪婪解码, beam=束搜索')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam Search的束宽度')
    
    # ===== 设备配置 =====
    parser.add_argument('--device', default=None,
                        help='运行设备: cuda/cpu，默认自动检测')
    
    # ===== 翻译方向 =====
    parser.add_argument('--src_lang', default='zh', choices=['zh', 'en'],
                        help='源语言: zh=中译英（默认）, en=英译中')
    
    # ===== 数据配置 =====
    parser.add_argument('--train_file', default='data/train_10k.jsonl',
                        help='训练数据文件路径（如 data/train_10k_retranslated_hunyuan.jsonl）')
    parser.add_argument('--use_hunyuan', action='store_true',
                        help='使用混元重译数据 (zh_hy 字段替代 zh)')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        compare_experiments([
            # {'attention_type': 'dot', 'teacher_forcing_ratio': 0.8},
            # {'attention_type': 'multiplicative', 'teacher_forcing_ratio': 0.8},
            {'attention_type': 'additive', 'teacher_forcing_ratio': 1.0},
            {'attention_type': 'additive', 'teacher_forcing_ratio': 0.8},
            {'attention_type': 'additive', 'teacher_forcing_ratio': 0.5},
            {'attention_type': 'additive', 'teacher_forcing_ratio': 0.3},
            {'attention_type': 'additive', 'teacher_forcing_ratio': 0.0},
            # {'attention_type': 'multiplicative', 'teacher_forcing_ratio': 1.0, 'tf_decay': 0.05},
            # {'attention_type': 'additive', 'teacher_forcing_ratio': 1.0, 'tf_decay': 0.05},
        ], args.vocab_dir, args.save_dir, args.epochs, args.batch_size, args.src_lang)
    elif args.mode == 'evaluate':
        if not args.checkpoint: return print("需要 --checkpoint")
        evaluate_checkpoint(args.checkpoint, args.vocab_dir, args.batch_size, args.device)
    elif args.mode == 'interactive':
        if not args.checkpoint: return print("需要 --checkpoint")
        interactive_translate(args.checkpoint, args.vocab_dir, args.decode_method, args.beam_size, args.device)
    else:
        train(args.attention, args.tf_ratio, args.tf_decay, args.rnn_type, args.epochs, args.batch_size,
              args.eval_batch_size, args.lr, args.save_dir, args.vocab_dir, args.checkpoint, args.device, 
              args.src_lang, args.use_hunyuan, args.train_file)


if __name__ == "__main__":
    main()
