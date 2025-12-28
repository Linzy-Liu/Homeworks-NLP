import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
from tqdm import tqdm

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

# 1. 配置与模型加载
# LaBSE 专门用于计算跨语言句子的相似度
model = SentenceTransformer('sentence-transformers/LaBSE')

# 数据路径配置
train_path = "data/train_100k.jsonl"  # 训练集
valid_path = "data/valid.jsonl"        # 验证集（可选）

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# 2. 提取数据
print("加载训练集...")
df_train = load_data(train_path)
print(f"成功加载训练集，共 {len(df_train)} 条样本")

# 尝试加载验证集
try:
    print("加载验证集...")
    df_valid = load_data(valid_path)
    print(f"成功加载验证集，共 {len(df_valid)} 条样本")
    has_valid = True
except FileNotFoundError:
    print("未找到验证集，仅分析训练集")
    df_valid = None
    has_valid = False

en_sentences_train = df_train['en'].tolist()
zh_sentences_train = df_train['zh'].tolist()

if has_valid:
    en_sentences_valid = df_valid['en'].tolist()
    zh_sentences_valid = df_valid['zh'].tolist()

# 3. 计算嵌入向量 (Embeddings)
print("\n正在计算训练集编码（这可能需要几分钟，建议使用 GPU）...")
en_embeddings_train = model.encode(en_sentences_train, convert_to_tensor=True, show_progress_bar=True)
zh_embeddings_train = model.encode(zh_sentences_train, convert_to_tensor=True, show_progress_bar=True)

if has_valid:
    print("\n正在计算验证集编码...")
    en_embeddings_valid = model.encode(en_sentences_valid, convert_to_tensor=True, show_progress_bar=True)
    zh_embeddings_valid = model.encode(zh_sentences_valid, convert_to_tensor=True, show_progress_bar=True)

# 4. 计算语义相似度 (Cosine Similarity)
print("\n正在计算语义相似度...")
cosine_scores_train = util.cos_sim(en_embeddings_train, zh_embeddings_train)
similarities_train = torch.diagonal(cosine_scores_train).cpu().numpy()
df_train['similarity'] = similarities_train

if has_valid:
    cosine_scores_valid = util.cos_sim(en_embeddings_valid, zh_embeddings_valid)
    similarities_valid = torch.diagonal(cosine_scores_valid).cpu().numpy()
    df_valid['similarity'] = similarities_valid

# 5. 可视化 - 相似度直方图（美化版）
print("\n生成相似度分布图...")
fig, ax = plt.subplots(figsize=(12, 7))

# 训练集分布（半透明）
sns.histplot(
    df_train['similarity'], 
    bins=50, 
    kde=True, 
    color='#3498db',  # 蓝色
    alpha=0.6,        # 透明度
    label='Train Set',
    stat='density',
    ax=ax
)

# 验证集分布（如果存在）
if has_valid:
    sns.histplot(
        df_valid['similarity'], 
        bins=30, 
        kde=True, 
        color='#e74c3c',  # 红色
        alpha=0.5,         # 透明度
        label='Valid Set',
        stat='density',
        ax=ax
    )

# 添加统计信息
mean_train = df_train['similarity'].mean()
std_train = df_train['similarity'].std()
ax.axvline(x=mean_train, color='#2980b9', linestyle='--', linewidth=2, 
           label=f'Train Mean: {mean_train:.3f}')

if has_valid:
    mean_valid = df_valid['similarity'].mean()
    ax.axvline(x=mean_valid, color='#c0392b', linestyle='--', linewidth=2,
               label=f'Valid Mean: {mean_valid:.3f}')

# 质量阈值线
ax.axvline(x=0.7, color='#95a5a6', linestyle=':', linewidth=2.5, 
           label='Quality Threshold (0.7)')

# 美化
ax.set_title("Semantic Similarity Distribution (LaBSE)", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Cosine Similarity Score", fontsize=13, fontweight='bold')
ax.set_ylabel("Density", fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(0, 1)

plt.tight_layout()
plt.savefig("similarity_distribution.png", dpi=300, bbox_inches='tight')
print("已保存: similarity_distribution.png")
plt.show()

# 6. 可视化 - t-SNE 空间分布对比（美化版）
print("\n正在进行 t-SNE 降维可视化...")
n_samples_train = min(800, len(df_train))
n_samples_valid = min(200, len(df_valid)) if has_valid else 0

# 训练集采样
indices_train = np.random.choice(len(df_train), n_samples_train, replace=False)
en_subset_train = en_embeddings_train[indices_train].cpu().numpy()
zh_subset_train = zh_embeddings_train[indices_train].cpu().numpy()

# 验证集采样（如果存在）
if has_valid:
    indices_valid = np.random.choice(len(df_valid), n_samples_valid, replace=False)
    en_subset_valid = en_embeddings_valid[indices_valid].cpu().numpy()
    zh_subset_valid = zh_embeddings_valid[indices_valid].cpu().numpy()
    
    # 合并所有数据一起降维
    combined = np.concatenate([
        en_subset_train, zh_subset_train,
        en_subset_valid, zh_subset_valid
    ])
else:
    combined = np.concatenate([en_subset_train, zh_subset_train])

# t-SNE 降维
print("执行 t-SNE 降维（可能需要1-2分钟）...")
tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42, perplexity=30)
reduced = tsne.fit_transform(combined)

# 分离结果
offset = 0
en_reduced_train = reduced[offset:offset+n_samples_train]
offset += n_samples_train
zh_reduced_train = reduced[offset:offset+n_samples_train]
offset += n_samples_train

if has_valid:
    en_reduced_valid = reduced[offset:offset+n_samples_valid]
    offset += n_samples_valid
    zh_reduced_valid = reduced[offset:offset+n_samples_valid]

# 绘图
fig, ax = plt.subplots(figsize=(14, 10))

# 训练集 - 圆点
ax.scatter(en_reduced_train[:, 0], en_reduced_train[:, 1], 
           c='#3498db', alpha=0.5, s=30, label='Train EN', marker='o', edgecolors='none')
ax.scatter(zh_reduced_train[:, 0], zh_reduced_train[:, 1], 
           c='#2ecc71', alpha=0.5, s=30, label='Train ZH', marker='o', edgecolors='none')

# 验证集 - × 标记
if has_valid:
    ax.scatter(en_reduced_valid[:, 0], en_reduced_valid[:, 1], 
               c='#e74c3c', alpha=0.7, s=80, label='Valid EN', marker='x', linewidths=2)
    ax.scatter(zh_reduced_valid[:, 0], zh_reduced_valid[:, 1], 
               c='#f39c12', alpha=0.7, s=80, label='Valid ZH', marker='x', linewidths=2)

# 连线：展示平行句对之间的对齐程度
n_connections = min(15, n_samples_train)
for i in range(n_connections):
    ax.plot([en_reduced_train[i, 0], zh_reduced_train[i, 0]], 
            [en_reduced_train[i, 1], zh_reduced_train[i, 1]], 
            color='gray', linewidth=0.5, alpha=0.2, zorder=0)

if has_valid:
    n_connections_valid = min(10, n_samples_valid)
    for i in range(n_connections_valid):
        ax.plot([en_reduced_valid[i, 0], zh_reduced_valid[i, 0]], 
                [en_reduced_valid[i, 1], zh_reduced_valid[i, 1]], 
                color='#e74c3c', linewidth=0.8, alpha=0.3, linestyle='--', zorder=0)

# 美化
ax.set_title("Embedding Space Distribution: EN vs ZH (t-SNE)", 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("t-SNE Dimension 1", fontsize=13, fontweight='bold')
ax.set_ylabel("t-SNE Dimension 2", fontsize=13, fontweight='bold')
ax.legend(loc='best', fontsize=11, framealpha=0.9, markerscale=1.2)
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig("embedding_space.png", dpi=300, bbox_inches='tight')
print("已保存: embedding_space.png")
plt.show()

# 7. 筛选出低质量数据
print("\n" + "="*60)
print("数据质量分析报告")
print("="*60)

# 训练集统计
bad_data_train = df_train[df_train['similarity'] < 0.7].sort_values(by='similarity')
print(f"\n【训练集】")
print(f"  总样本数: {len(df_train)}")
print(f"  平均相似度: {df_train['similarity'].mean():.4f} ± {df_train['similarity'].std():.4f}")
print(f"  低质量样本 (< 0.7): {len(bad_data_train)} ({len(bad_data_train)/len(df_train)*100:.2f}%)")

# 验证集统计
if has_valid:
    bad_data_valid = df_valid[df_valid['similarity'] < 0.7].sort_values(by='similarity')
    print(f"\n【验证集】")
    print(f"  总样本数: {len(df_valid)}")
    print(f"  平均相似度: {df_valid['similarity'].mean():.4f} ± {df_valid['similarity'].std():.4f}")
    print(f"  低质量样本 (< 0.7): {len(bad_data_valid)} ({len(bad_data_valid)/len(df_valid)*100:.2f}%)")

# 显示最差的样本
print(f"\n【最差样本示例（训练集）】")
print(bad_data_train[['en', 'zh', 'similarity']].head(5).to_string(index=False))

# 保存清洗建议
if len(bad_data_train) > 0:
    bad_data_train.to_csv("misaligned_train_report.csv", index=False)
    print(f"\n已保存训练集低质量样本报告: misaligned_train_report.csv")

if has_valid and len(bad_data_valid) > 0:
    bad_data_valid.to_csv("misaligned_valid_report.csv", index=False)
    print(f"已保存验证集低质量样本报告: misaligned_valid_report.csv")

print("\n分析完成！")