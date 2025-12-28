"""
预训练词向量加载模块
支持: GloVe, Word2Vec, FastText 格式
"""
import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm


def load_pretrained_vectors(
    vector_path: str,
    vocab: Dict[str, int],
    embed_dim: int,
    binary: bool = False
) -> Tuple[torch.Tensor, int, int]:
    """
    加载预训练词向量
    
    Args:
        vector_path: 词向量文件路径 (支持 GloVe/Word2Vec/FastText 文本格式)
        vocab: 词表 {token: idx}
        embed_dim: 词向量维度
        binary: 是否为二进制格式 (Word2Vec binary)
    
    Returns:
        embedding_matrix: [vocab_size, embed_dim] 词向量矩阵
        found_count: 找到的词数量
        oov_count: 未找到的词数量 (使用随机初始化)
    """
    vocab_size = len(vocab)
    
    # 初始化词向量矩阵 (使用正态分布随机初始化)
    embedding_matrix = torch.randn(vocab_size, embed_dim) * 0.1
    
    # 特殊 token 初始化为零向量
    special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
    for token in special_tokens:
        if token in vocab:
            embedding_matrix[vocab[token]] = torch.zeros(embed_dim)
    
    found_count = 0
    
    print(f"加载预训练词向量: {vector_path}")
    
    if binary:
        # Word2Vec 二进制格式
        from gensim.models import KeyedVectors
        word_vectors = KeyedVectors.load_word2vec_format(vector_path, binary=True)
        pretrained_dim = word_vectors.vector_size
        
        # 如果维度不匹配，调整 embedding_matrix
        if pretrained_dim != embed_dim:
            print(f"警告: 预训练向量维度 {pretrained_dim} != 目标维度 {embed_dim}")
            print(f"将使用原始维度 {pretrained_dim}，请在模型中使用投影层")
            # 重新初始化为预训练维度
            embedding_matrix = torch.randn(vocab_size, pretrained_dim) * 0.1
            for token in special_tokens:
                if token in vocab:
                    embedding_matrix[vocab[token]] = torch.zeros(pretrained_dim)
        
        for word, idx in tqdm(vocab.items(), desc="匹配词向量"):
            if word in word_vectors:
                embedding_matrix[idx] = torch.from_numpy(word_vectors[word].copy())
                found_count += 1
    else:
        # 文本格式 (GloVe / Word2Vec text / FastText text)
        with open(vector_path, 'r', encoding='utf-8', errors='ignore') as f:
            # 跳过可能的头部行 (Word2Vec/FastText 格式第一行是 vocab_size dim)
            first_line = f.readline().strip().split()
            if len(first_line) == 2:
                # Word2Vec/FastText 格式，第一行是元信息
                file_dim = int(first_line[1])
                if file_dim != embed_dim:
                    print(f"警告: 文件维度 {file_dim} != 指定维度 {embed_dim}")
            else:
                # GloVe 格式，第一行就是词向量
                f.seek(0)
            
            for line in tqdm(f, desc="加载词向量"):
                parts = line.rstrip().split(' ')
                if len(parts) < embed_dim + 1:
                    continue
                word = parts[0]
                if word in vocab:
                    try:
                        vector = torch.tensor([float(x) for x in parts[1:embed_dim+1]])
                        embedding_matrix[vocab[word]] = vector
                        found_count += 1
                    except ValueError:
                        continue
    
    oov_count = vocab_size - found_count - len(special_tokens)
    print(f"词向量加载完成: 找到 {found_count}/{vocab_size} ({found_count/vocab_size*100:.1f}%)")
    print(f"OOV 词使用随机初始化: {oov_count} 个")
    
    return embedding_matrix, found_count, oov_count


def download_glove(save_dir: str = "embeddings", dim: int = 100) -> str:
    """
    下载 GloVe 词向量 (英文)
    
    Args:
        save_dir: 保存目录
        dim: 维度 (50, 100, 200, 300)
    
    Returns:
        词向量文件路径
    """
    import urllib.request
    import zipfile
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    glove_file = save_dir / f"glove.6B.{dim}d.txt"
    if glove_file.exists():
        print(f"GloVe 已存在: {glove_file}")
        return str(glove_file)
    
    zip_path = save_dir / "glove.6B.zip"
    url = "http://nlp.stanford.edu/data/glove.6B.zip"
    
    print(f"下载 GloVe 词向量 ({dim}d)...")
    print(f"URL: {url}")
    print("注意: 文件较大 (~862MB)，请耐心等待...")
    
    urllib.request.urlretrieve(url, zip_path)
    
    print("解压中...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    
    os.remove(zip_path)
    print(f"完成: {glove_file}")
    
    return str(glove_file)


def create_embedding_layer(
    vocab_size: int,
    embed_dim: int,
    pretrained_weights: Optional[torch.Tensor] = None,
    padding_idx: int = 0,
    freeze: bool = False
) -> nn.Embedding:
    """
    创建词嵌入层，支持预训练权重
    
    Args:
        vocab_size: 词表大小
        embed_dim: 嵌入维度
        pretrained_weights: 预训练权重 [vocab_size, embed_dim]
        padding_idx: padding token 索引
        freeze: 是否冻结词向量 (不参与训练)
    
    Returns:
        nn.Embedding 层
    """
    embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
    
    if pretrained_weights is not None:
        assert pretrained_weights.shape == (vocab_size, embed_dim), \
            f"权重形状不匹配: {pretrained_weights.shape} vs ({vocab_size}, {embed_dim})"
        embedding.weight.data.copy_(pretrained_weights)
        print(f"已加载预训练词向量: {vocab_size} x {embed_dim}")
    
    if freeze:
        embedding.weight.requires_grad = False
        print("词向量已冻结，不参与训练")
    
    return embedding


# ==================== 常用中文词向量 ====================
"""
常用中文预训练词向量下载地址:

1. Tencent AI Lab 中文词向量 (推荐)
   - 维度: 200
   - 词数: 8M
   - 下载: https://ai.tencent.com/ailab/nlp/en/embedding.html
   - 文件: Tencent_AILab_ChineseEmbedding.txt (~16GB)

2. Chinese Word Vectors
   - GitHub: https://github.com/Embedding/Chinese-Word-Vectors
   - 多种语料和维度可选

3. FastText 中文
   - 下载: https://fasttext.cc/docs/en/crawl-vectors.html
   - 文件: cc.zh.300.vec.gz

使用方法:
    # 加载中文词向量
    zh_vectors, _, _ = load_pretrained_vectors(
        "path/to/chinese_vectors.txt",
        tokenizer.zh_vocab.token2idx,
        embed_dim=200
    )
    
    # 加载英文词向量 (GloVe)
    en_vectors, _, _ = load_pretrained_vectors(
        "embeddings/glove.6B.100d.txt",
        tokenizer.en_vocab.token2idx,
        embed_dim=100
    )
"""

