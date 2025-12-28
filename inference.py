"""
One-click Batch Inference Script for NMT Project (RNN, Transformer, T5)
"""
import torch
import os
from pathlib import Path

# 导入自定义模块与组件
from model import create_model  #
from train import translate as rnn_translate
from transformer_train import translate as transformer_translate
from transformer_train import load_all
from transformer_train import TransformerNMT, TransformerConfig  #
from preprocess import TranslationTokenizer  #
from transformers import MT5ForConditionalGeneration, T5Tokenizer  #

# ==================== 模型加载路径与配置 (请在此处修改) ====================
MODEL_PATHS = {
    "rnn": "checkpoints/GRU_multiplicative_tf1.0_decay0.05_best.pt",
    "transformer": "checkpoints/ablation_transformer_absolute_rmsnorm_d256_best.pt",
    "t5": "checkpoints/t5_mt5-small_zh2en_best",
    "vocab_dir": "vocab"
}

# 预设测试用例
TEST_SENTENCES = [
    "这是一个关于机器翻译的测试。",
    "今天天气不错，我们去散步吧。",
    "我正在学习如何使用深度学习模型进行翻译。",
    "人工智能正在改变我们的生活方式。"
]
# =====================================================================

class BatchTranslator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing Translator on {self.device}...")
        
        # 1. 加载词表和基础 Tokenizer
        self.tokenizer_scratch = TranslationTokenizer.load(MODEL_PATHS["vocab_dir"])
        
        # 2. 初始化并加载 RNN 模型
        ckpt = torch.load(MODEL_PATHS["rnn"], map_location=self.device)
        cfg = ckpt['config']
        cfg['dropout'] = 0.4
        attention_type, rnn_type = cfg['attention_type'], cfg['rnn_type']
        model = create_model(cfg['src_vocab_size'], cfg['tgt_vocab_size'], cfg['embed_dim'], cfg['hidden_dim'],
                           cfg['num_layers'], cfg['dropout'], rnn_type, attention_type).to(self.device)
        model.load_state_dict(ckpt['model_state_dict'])
        tokenizer, _, _, _ = load_all(MODEL_PATHS["vocab_dir"], src_lang='zh')
        self.tokenizer_scratch = tokenizer
        self.rnn = model
        self.rnn.eval()
        
        # 3. 初始化并加载 Transformer 模型
        ckpt = torch.load(MODEL_PATHS["transformer"], map_location=self.device)
        config_dict = ckpt['config']
        config_dict['dropout'] = 0.3
        config = TransformerConfig.from_dict(config_dict)
        self.transformer = TransformerNMT(4931, 6803, config).to(self.device)
        self.transformer.load_state_dict(ckpt['model_state_dict'])
        self.transformer.eval()
        
        # 4. 初始化并加载 mT5 模型
        self.t5_tokenizer = T5Tokenizer.from_pretrained(MODEL_PATHS["t5"])
        self.t5_model = MT5ForConditionalGeneration.from_pretrained(MODEL_PATHS["t5"]).to(self.device)
        self.t5_model.eval()
        print("[*] All models loaded successfully.\n")

    def translate_all(self, text):
        results = {}
        
        # RNN 推理 
        rnn_preds = rnn_translate(self.rnn, self.tokenizer_scratch, text, self.device)
        results["RNN"] = rnn_preds
        
        # Transformer 推理 
        
        tf_preds = transformer_translate(self.transformer, self.tokenizer_scratch, text, self.device)
        results["Transformer"] = tf_preds
        
        # T5 推理
        t5_inputs = self.t5_tokenizer(f"translate Chinese to English: {text}", return_tensors="pt").to(self.device)
        t5_outputs = self.t5_model.generate(t5_inputs.input_ids, max_length=100, num_beams=1)
        results["T5"] = self.t5_tokenizer.decode(t5_outputs[0], skip_special_tokens=True)
        
        return results

def main():
    translator = BatchTranslator()
    line_width = 110
    print("=" * line_width)
    print(f"{'Source Chinese Text':<40} | {'Model':<12} | {'English Translation'}")
    print("-" * line_width)
        
    for sentence in TEST_SENTENCES:
        # 运行一次，获取三个模型的结果
        translations = translator.translate_all(sentence)
            
        # 直接循环输出结果，不再使用 first 逻辑
        for model_name, result in translations.items():
            print(f"{sentence:<40} | {model_name:<12} | {result}")
            
        print("-" * line_width)

if __name__ == "__main__":
    main()
    print("此处由于找不到原本Transformer的词表了（在后续测试中被覆盖了），所以Transformer的翻译结果可能不太准确")