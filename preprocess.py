"""机器翻译数据预处理模块"""
import os, re, json, pickle, shutil
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 强制使用 HuggingFace datasets
from datasets import Dataset as HFDataset, DatasetDict, load_from_disk


class Config:
    """预处理配置"""
    MAX_LENGTH = 512
    MIN_FREQ = 4
    BPE_VOCAB_SIZE = 8000
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<bos>", "<eos>", "<unk>"
    SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
    PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


class DataCleaner:
    """数据清洗器"""
    HTML_PATTERN = re.compile(r'<[^>]+>') # HTML标签正则
    
    CONTROL_CHAR_PATTERN = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')# 控制字符正则（保留换行、制表符等常用控制符）
    
    # 非法字符正则（保留中英文、数字、常用标点）
    # 中文范围：\u4e00-\u9fff（基本汉字）、\u3000-\u303f（中文标点）、\uff00-\uffef（全角字符）
    # 英文范围：a-zA-Z、常用标点
    VALID_CHAR_PATTERN = re.compile(
        r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef'  # 中文相关
        r'a-zA-Z0-9'                                   # 英文字母和数字
        r'\s'                                          # 空白字符
        r'.,!?;:\'\"()\[\]{}<>@#$%^&*+=\-_/\\|`~'     # 英文标点
        r'。，！？；：''""（）【】《》、·…—]'           # 中文标点
    )
    
    def __init__(self, max_length: int = Config.MAX_LENGTH):
        self.max_length = max_length
    
    def clean_text(self, text: str) -> str:
        if not text: return ""
        text = self.HTML_PATTERN.sub('', text)
        text = self.CONTROL_CHAR_PATTERN.sub('', text)
        text = self.VALID_CHAR_PATTERN.sub('', text)
        return re.sub(r'\s+', ' ', text).strip()
    
    def process_pair(self, en: str, zh: str) -> Optional[Tuple[str, str]]:
        en, zh = self.clean_text(en), self.clean_text(zh)
        if not en or not zh: return None
        return (en[:self.max_length], zh[:self.max_length])


class ChineseTokenizer:
    """中文分词器 - HanLP"""
    def __init__(self):
        self._tok = None
    
    @property
    def tokenizer(self):
        if self._tok is None:
            import hanlp
            self._tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        return self._tok
    
    def tokenize(self, text: str) -> List[str]:
        """对中文文本进行分词"""
        if not text:
            return []
        try:
            tokens = self.tokenizer(text)
            return tokens
        except Exception as e:
            print(f"中文分词错误: {e}, 文本: {text[:50]}...")
            # 回退到字符级分词
            return list(text)
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """批量分词"""
        results = []
        for text in tqdm(texts, desc="中文分词"):
            results.append(self.tokenize(text))
        return results


class EnglishBPETokenizer:
    """英文BPE分词器"""
    def __init__(self, vocab_size: int = Config.BPE_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self._tok = None
        self.is_trained = False
    
    def train(self, texts: List[str], save_path: str = None):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
        
        tok = Tokenizer(BPE(unk_token=Config.UNK_TOKEN))
        tok.pre_tokenizer = Whitespace()
        tok.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
        tok.train_from_iterator(texts, BpeTrainer(vocab_size=self.vocab_size, special_tokens=Config.SPECIAL_TOKENS, show_progress=True))
        self._tok, self.is_trained = tok, True
        if save_path: 
            tok.save(save_path)
    
    def load(self, path: str):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)
        self.is_trained = True
    
    def tokenize(self, text: str) -> List[str]:
        return self._tok.encode(text).tokens if text else []
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenize(t) for t in tqdm(texts, desc="英文BPE分词")]


class Vocabulary:
    """词表类"""
    def __init__(self, min_freq: int = Config.MIN_FREQ):
        self.min_freq = min_freq
        self.token2idx = {t: i for i, t in enumerate(Config.SPECIAL_TOKENS)}
        self.idx2token = {i: t for i, t in enumerate(Config.SPECIAL_TOKENS)}
        self.token_freqs = Counter()
    
    @property
    def pad_idx(self): return Config.PAD_IDX
    @property
    def bos_idx(self): return Config.BOS_IDX
    @property
    def eos_idx(self): return Config.EOS_IDX
    @property
    def unk_idx(self): return Config.UNK_IDX
    def __len__(self): return len(self.token2idx)
    
    def build_from_tokens(self, token_lists: List[List[str]]):
        for tokens in token_lists: 
            self.token_freqs.update(tokens)
        idx = len(Config.SPECIAL_TOKENS)
        for token, freq in self.token_freqs.items():
            if freq >= self.min_freq and token not in self.token2idx:
                self.token2idx[token], self.idx2token[idx] = idx, token
                idx += 1
        print(f"词表大小: {len(self)}")
    
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.token2idx.get(t, self.unk_idx) for t in tokens]
    
    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.idx2token.get(i, Config.UNK_TOKEN) for i in ids]
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'token2idx': self.token2idx, 'idx2token': self.idx2token, 
                        'token_freqs': dict(self.token_freqs), 'min_freq': self.min_freq}, f)
    
    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        with open(path, 'rb') as f: 
            data = pickle.load(f)
        vocab = cls(data['min_freq'])
        vocab.token2idx, vocab.idx2token = data['token2idx'], {int(k): v for k, v in data['idx2token'].items()}
        vocab.token_freqs = Counter(data['token_freqs'])
        return vocab


class TranslationTokenizer:
    """机器翻译Tokenizer"""
    def __init__(self, zh_vocab: Vocabulary, en_vocab: Vocabulary, 
                 zh_tokenizer: ChineseTokenizer, en_tokenizer: EnglishBPETokenizer,
                 max_length: int = Config.MAX_LENGTH):
        self.zh_vocab, self.en_vocab = zh_vocab, en_vocab
        self.zh_tokenizer, self.en_tokenizer = zh_tokenizer, en_tokenizer
        self.max_length = max_length
    
    def _encode(self, text: str, tokenizer, vocab, add_special: bool = True) -> List[int]:
        ids = vocab.tokens_to_ids(tokenizer.tokenize(text))
        if add_special: 
            ids = [vocab.bos_idx] + ids + [vocab.eos_idx]
        if len(ids) > self.max_length: ids = ids[:self.max_length-1] + [vocab.eos_idx]
        return ids
    
    def encode_chinese(self, text: str, add_special: bool = True) -> List[int]:
        return self._encode(text, self.zh_tokenizer, self.zh_vocab, add_special)
    
    def encode_english(self, text: str, add_special: bool = True) -> List[int]:
        return self._encode(text, self.en_tokenizer, self.en_vocab, add_special)
    
    def decode_chinese(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = self.zh_vocab.ids_to_tokens(ids)
        if skip_special: 
            tokens = [t for t in tokens if t not in Config.SPECIAL_TOKENS]
        return ''.join(tokens)
    
    def decode_english(self, ids: List[int], skip_special: bool = True) -> str:
        tokens = self.en_vocab.ids_to_tokens(ids)
        if skip_special: 
            tokens = [t for t in tokens if t not in Config.SPECIAL_TOKENS]
        return ' '.join(tokens).replace(' ##', '').replace('##', '')
    
    def save(self, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.zh_vocab.save(str(save_dir / 'zh_vocab.pkl'))
        self.en_vocab.save(str(save_dir / 'en_vocab.pkl'))
        if self.en_tokenizer.is_trained:
            self.en_tokenizer._tok.save(str(save_dir / 'en_bpe.json'))
    
    @classmethod
    def load(cls, load_dir: str, max_length: int = Config.MAX_LENGTH) -> 'TranslationTokenizer':
        load_dir = Path(load_dir)
        zh_vocab, en_vocab = Vocabulary.load(str(load_dir / 'zh_vocab.pkl')), Vocabulary.load(str(load_dir / 'en_vocab.pkl'))
        zh_tok, en_tok = ChineseTokenizer(), EnglishBPETokenizer()
        bpe_path = load_dir / 'en_bpe.json'
        if bpe_path.exists(): en_tok.load(str(bpe_path))
        return cls(zh_vocab, en_vocab, zh_tok, en_tok, max_length)


class TranslationDataset(Dataset):
    """机器翻译数据集 (基于 HuggingFace Dataset)"""
    def __init__(self, data: 'HFDataset', tokenizer: TranslationTokenizer, src_lang: str = 'en', use_hunyuan: bool = False):
        """
        Args:
            data: HuggingFace Dataset
            tokenizer: 翻译分词器
            src_lang: 源语言 ('en' 或 'zh')
            use_hunyuan: 是否使用 zh_hy 字段（混元重译中文）代替 zh
        """
        self.data = data
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.use_hunyuan = use_hunyuan
        
        # 检查是否支持 hunyuan 字段
        if use_hunyuan and 'zh_hy' not in data.column_names:
            print("警告: 数据集中不存在 zh_hy 字段，将使用 zh 字段")
            self.use_hunyuan = False
    
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        # HF Dataset 支持索引访问，并使用内存映射 (Arrow)
        item = self.data[idx]
        en = item['en']
        # 根据 use_hunyuan 选择中文字段
        zh = item.get('zh_hy', item['zh']) if self.use_hunyuan else item['zh']
            
        if self.src_lang == 'en':
            src_ids, tgt_ids = self.tokenizer.encode_english(en), self.tokenizer.encode_chinese(zh)
        else:
            src_ids, tgt_ids = self.tokenizer.encode_chinese(zh), self.tokenizer.encode_english(en)
        return {'src_ids': torch.tensor(src_ids), 'tgt_ids': torch.tensor(tgt_ids),
                'src_text': en if self.src_lang == 'en' else zh, 'tgt_text': zh if self.src_lang == 'en' else en}


def collate_fn(batch: List[Dict], pad_idx: int = Config.PAD_IDX) -> Dict:
    src_ids, tgt_ids = [b['src_ids'] for b in batch], [b['tgt_ids'] for b in batch]
    src_pad = pad_sequence(src_ids, batch_first=True, padding_value=pad_idx)
    tgt_pad = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_idx)
    return {'src_ids': src_pad, 'tgt_ids': tgt_pad, 'src_mask': (src_pad != pad_idx).long(),
            'tgt_mask': (tgt_pad != pad_idx).long(), 'src_lengths': torch.tensor([len(s) for s in src_ids]),
            'tgt_lengths': torch.tensor([len(t) for t in tgt_ids])}


def create_dataloader(dataset: TranslationDataset, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    # Windows 下多进程读取可能不稳定，设为 0；Linux/Mac 下设为 4 加速
    # 为避免 HanLP 在多进程 Worker 中重复加载导致显存泄漏，这里强制设为 0
    num_workers = 0 
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=lambda b: collate_fn(b, Config.PAD_IDX), 
                      pin_memory=torch.cuda.is_available())


def load_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def preprocess_pipeline(train_file: str = "data/train_10k.jsonl", valid_file: str = "data/valid.jsonl",
                       test_file: str = "data/test.jsonl", save_dir: str = "vocab",
                       min_freq: int = Config.MIN_FREQ, max_length: int = Config.MAX_LENGTH,
                       src_lang: str = 'zh', use_hunyuan: bool = False):
    """完整预处理流程
    
    Args:
        use_hunyuan: 是否使用 zh_hy 字段（混元重译中文）代替 zh，
                     要求训练数据文件包含 zh_hy 字段
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    cleaner = DataCleaner(max_length)
    
    def clean_file(path):
        if not os.path.exists(path): return []
        # 返回 [{'en': ..., 'zh': ..., 'zh_hy': ...}, ...] 方便转 HF Dataset
        results = []
        for d in load_jsonl(path):
            # use_hunyuan 时优先使用 zh_hy 字段（对所有数据集生效）
            if use_hunyuan and 'zh_hy' in d:
                zh_text = d.get('zh_hy', d.get('zh', ''))
            else:
                zh_text = d.get('zh', '')
            
            res = cleaner.process_pair(d.get('en',''), zh_text)
            if res:
                item = {'en': res[0], 'zh': res[1]}
                # 如果存在 zh_hy，也保留（用于后续灵活选择）
                if 'zh_hy' in d:
                    zh_hy_clean = cleaner.clean_text(d['zh_hy'])
                    if zh_hy_clean:
                        item['zh_hy'] = zh_hy_clean[:max_length]
                results.append(item)
        return results
    
    print("正在清洗数据...")
    if use_hunyuan:
        print("使用混元重译数据 (zh_hy 字段) - 对 train/valid/test 全部生效")
    train_data = clean_file(train_file)
    valid_data = clean_file(valid_file)
    test_data = clean_file(test_file)
    print(f"数据量: 训练={len(train_data)}, 验证={len(valid_data)}, 测试={len(test_data)}")
    print(f"翻译方向: {'中文→英文' if src_lang == 'zh' else '英文→中文'}")
    
    # 提取文本用于训练 Tokenizer
    train_en = [d['en'] for d in train_data]
    train_zh = [d['zh'] for d in train_data]
    
    # 训练/初始化 Tokenizer
    zh_tok, en_tok = ChineseTokenizer(), EnglishBPETokenizer()
    en_tok.train(train_en, str(save_dir / 'en_bpe.json'))
    
    print("构建词表...")
    zh_tokens = zh_tok.tokenize_batch(train_zh)
    en_tokens = en_tok.tokenize_batch(train_en)
    
    zh_vocab, en_vocab = Vocabulary(min_freq), Vocabulary(min_freq)
    zh_vocab.build_from_tokens(zh_tokens)
    en_vocab.build_from_tokens(en_tokens)
    
    tokenizer = TranslationTokenizer(zh_vocab, en_vocab, zh_tok, en_tok, max_length)
    tokenizer.save(str(save_dir))
    
    # 保存 HuggingFace Dataset 到磁盘
    print("保存 HuggingFace Dataset 到磁盘...")
    dataset_dict = DatasetDict()
    if train_data: dataset_dict['train'] = HFDataset.from_list(train_data)
    if valid_data: dataset_dict['valid'] = HFDataset.from_list(valid_data)
    if test_data: dataset_dict['test'] = HFDataset.from_list(test_data)
    
    dataset_path = save_dir / 'hf_dataset'
    if dataset_path.exists(): shutil.rmtree(dataset_path)
    dataset_dict.save_to_disk(str(dataset_path))
    
    # 保存 meta info json
    with open(save_dir / 'meta_info.json', 'w') as f:
        json.dump({'src_lang': src_lang, 'use_hunyuan': use_hunyuan}, f)
        
    # use_hunyuan 对 train/valid/test 全部生效
    return (tokenizer, 
            TranslationDataset(dataset_dict['train'], tokenizer, src_lang=src_lang, use_hunyuan=use_hunyuan),
            TranslationDataset(dataset_dict['valid'], tokenizer, src_lang=src_lang, use_hunyuan=use_hunyuan) if valid_data else None,
            TranslationDataset(dataset_dict['test'], tokenizer, src_lang=src_lang, use_hunyuan=use_hunyuan) if test_data else None)


def check_preprocessed_exists(save_dir: str = "vocab") -> bool:
    save_dir = Path(save_dir)
    # 检查 Tokenizer 文件
    tokens_exist = all((save_dir / f).exists() for f in ['zh_vocab.pkl', 'en_vocab.pkl', 'en_bpe.json'])
    # 检查数据文件 (必须是 HF Dataset 目录)
    data_exist = (save_dir / 'hf_dataset').exists()
    return tokens_exist and data_exist


def load_all(save_dir: str = "vocab", max_length: int = Config.MAX_LENGTH, src_lang: str = None, use_hunyuan: bool = None):
    """加载预处理结果
    
    Args:
        use_hunyuan: 是否使用 zh_hy 字段。如果为 None，则从 meta_info.json 读取
    """
    save_dir = Path(save_dir)
    tokenizer = TranslationTokenizer.load(str(save_dir), max_length)
    
    print("加载 HuggingFace Dataset (内存映射)...")
    if not (save_dir / 'hf_dataset').exists():
        raise FileNotFoundError(f"未找到数据集目录: {save_dir / 'hf_dataset'}")
    
    dataset_dict = load_from_disk(str(save_dir / 'hf_dataset'))
    
    if (save_dir / 'meta_info.json').exists():
        with open(save_dir / 'meta_info.json', 'r') as f:
            meta = json.load(f)
            if src_lang is None: src_lang = meta.get('src_lang', 'zh')
            # 如果未指定 use_hunyuan，从 meta 读取
            if use_hunyuan is None: use_hunyuan = meta.get('use_hunyuan', False)
    
    # 默认值
    if src_lang is None: src_lang = 'zh'
    if use_hunyuan is None: use_hunyuan = False
    
    # use_hunyuan 对 train/valid/test 全部生效
    train_ds = TranslationDataset(dataset_dict['train'], tokenizer, src_lang=src_lang, use_hunyuan=use_hunyuan) if 'train' in dataset_dict else None
    valid_ds = TranslationDataset(dataset_dict['valid'], tokenizer, src_lang=src_lang, use_hunyuan=use_hunyuan) if 'valid' in dataset_dict else None
    test_ds = TranslationDataset(dataset_dict['test'], tokenizer, src_lang=src_lang, use_hunyuan=use_hunyuan) if 'test' in dataset_dict else None
    
    print(f"翻译方向: {'中文→英文' if src_lang == 'zh' else '英文→中文'}")
    if use_hunyuan:
        print("使用混元重译数据 (zh_hy 字段)")
    return tokenizer, train_ds, valid_ds, test_ds


if __name__ == "__main__":
    save_dir = "vocab"
    src_lang = 'zh'  # 中译英
    
    # 强制重新生成以测试新流程
    if check_preprocessed_exists(save_dir):
        print("发现已有数据，加载中...")
        tokenizer, train_ds, valid_ds, test_ds = load_all(save_dir, src_lang=src_lang)
    else:
        print("开始预处理...")
        tokenizer, train_ds, valid_ds, test_ds = preprocess_pipeline(src_lang=src_lang)
    
    loader = create_dataloader(train_ds, batch_size=32)
    batch = next(iter(loader))
    print(f"Batch: src={batch['src_ids'].shape}, tgt={batch['tgt_ids'].shape}")
    print(f"翻译方向: {'中文→英文' if src_lang == 'zh' else '英文→中文'}")
