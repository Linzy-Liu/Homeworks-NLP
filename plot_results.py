import json
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# Ensure output directories exist
os.makedirs('fig', exist_ok=True)
os.makedirs('tables', exist_ok=True)

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

def load_history(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return {}
    with open(filepath, 'r') as f:
        return json.load(f)

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_metric(data_map, metric_type, title, filename, ylabel=None, smooth_factor=None, figsize=(10, 6), fontsize=12):
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': fontsize})
    
    # Determine if we are plotting steps or epochs
    is_step = False
    
    for label, history in data_map.items():
        if metric_type == 'step_losses' and 'step_losses' in history:
            is_step = True
            values = history['step_losses']
            if smooth_factor:
                values = smooth_curve(values, smooth_factor)
            x_axis = range(1, len(values) + 1)
            plt.plot(x_axis, values, label=label, linewidth=1.5, alpha=0.9)
            
        elif metric_type == 'valid_loss' and 'valid_loss' in history:
            values = history['valid_loss']
            x_axis = range(1, len(values) + 1)
            plt.plot(x_axis, values, label=label, marker='o', markersize=4)

    plt.title(title)
    plt.xlabel('Steps' if is_step else 'Epochs')
    plt.ylabel(ylabel if ylabel else 'Loss')
    
    # Force integer ticks for Epochs
    if not is_step:
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(prop={'size': fontsize})
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join('fig', filename), dpi=300)
    plt.close()

def save_bleu_table(bleu_data, filename):
    # bleu_data: list of dicts {'Experiment': name, 'BLEU': score}
    df = pd.DataFrame(bleu_data)
    # Save as CSV
    csv_path = os.path.join('tables', filename + '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved BLEU table to {csv_path}")
    
    # Also save as a simple image for report inclusion if needed, or just Markdown
    # For now, just CSV is fine, but user asked to "summarize into a table".
    # I'll create a simple text table in a file too.
    txt_path = os.path.join('tables', filename + '.txt')
    with open(txt_path, 'w') as f:
        f.write(df.to_markdown(index=False))

def run_experiments():
    # 1. RNN Attention
    rnn_att_files = {
        'Additive': 'checkpoints/GRU_additive_tf0.8_history.json',
        'Multiplicative': 'checkpoints/GRU_multiplicative_tf0.8_history.json',
        'Dot Product': 'checkpoints/GRU_dot_tf0.8_history.json'
    }
    rnn_att_data = {k: load_history(v) for k, v in rnn_att_files.items()}
    plot_metric(rnn_att_data, 'step_losses', 'RNN Attention - Training Loss (Step)', 'rnn_attention_train_loss_step.png', 
                smooth_factor=0.95, figsize=(8, 6), fontsize=14)
    plot_metric(rnn_att_data, 'valid_loss', 'RNN Attention - Validation Loss', 'rnn_attention_valid_loss.png', 
                figsize=(8, 6), fontsize=14)
    
    # RNN Attention BLEU Table
    att_bleu = []
    for k, v in rnn_att_data.items():
        greedy_score = v.get('bleu_greedy', [0])[-1] if 'bleu_greedy' in v else v.get('bleu', [0])[-1]
        beam_score = v.get('bleu_beam', [0])[-1] if 'bleu_beam' in v else None
        
        if beam_score is not None:
            att_bleu.append({'Model': k, 'BLEU (Greedy)': greedy_score, 'BLEU (Beam)': beam_score})
        else:
            att_bleu.append({'Model': k, 'Final BLEU': greedy_score})
    save_bleu_table(att_bleu, 'rnn_attention_bleu_summary')

    # 2. RNN Teacher Forcing
    tf_ratios = ['0.0', '0.3', '0.5', '0.8', '1.0']
    tf_data = {}
    for r in tf_ratios:
        path = f'checkpoints/GRU_additive_tf{r}_history.json'
        tf_data[f'TF {r}'] = load_history(path)
    # Add decay
    decay_path = 'checkpoints/GRU_additive_tf1.0_decay0.05_history.json'
    tf_data['TF 1.0 Decay 0.05'] = load_history(decay_path)
    
    plot_metric(tf_data, 'step_losses', 'RNN Teacher Forcing - Training Loss (Step)', 'rnn_tf_train_loss_step.png', smooth_factor=0.95)
    plot_metric(tf_data, 'valid_loss', 'RNN Teacher Forcing - Validation Loss', 'rnn_tf_valid_loss.png')
    
    # TF BLEU Table
    tf_bleu = []
    
    # Manually adding multiplicative decay for comparison as requested
    mul_decay_path = 'checkpoints/GRU_multiplicative_tf1.0_decay0.05_history.json'
    if os.path.exists(mul_decay_path):
        tf_data['Multiplicative TF 1.0 Decay 0.05'] = load_history(mul_decay_path)

    plot_metric(tf_data, 'step_losses', 'RNN Teacher Forcing - Training Loss (Step)', 'rnn_tf_train_loss_step.png', 
                smooth_factor=0.95, figsize=(8, 6), fontsize=14)
    plot_metric(tf_data, 'valid_loss', 'RNN Teacher Forcing - Validation Loss', 'rnn_tf_valid_loss.png', 
                figsize=(8, 6), fontsize=14)

    for k, v in tf_data.items():
        greedy_score = v.get('bleu_greedy', [0])[-1] if 'bleu_greedy' in v else v.get('bleu', [0])[-1]
        beam_score = v.get('bleu_beam', [0])[-1] if 'bleu_beam' in v else None
        
        if beam_score is not None:
            tf_bleu.append({'Model': k, 'BLEU (Greedy)': greedy_score, 'BLEU (Beam)': beam_score})
        else:
            tf_bleu.append({'Model': k, 'Final BLEU': greedy_score})
    save_bleu_table(tf_bleu, 'rnn_tf_bleu_summary')

    # 3. Transformer Ablation
    ablation_files = {
        'Abs + LayerNorm': 'checkpoints/ablation_transformer_absolute_layernorm_d256_history.json',
        'Abs + RMSNorm': 'checkpoints/ablation_transformer_absolute_rmsnorm_d256_history.json',
        'Rel + LayerNorm': 'checkpoints/ablation_transformer_relative_layernorm_d256_history.json',
        'Rel + RMSNorm': 'checkpoints/ablation_transformer_relative_rmsnorm_d256_history.json',
    }
    ablation_data = {k: load_history(v) for k, v in ablation_files.items()}
    plot_metric(ablation_data, 'step_losses', 'Transformer Ablation - Training Loss (Step)', 'transformer_ablation_train_loss_step.png', 
                smooth_factor=0.9, figsize=(5, 5), fontsize=14)
    plot_metric(ablation_data, 'valid_loss', 'Transformer Ablation - Validation Loss', 'transformer_ablation_valid_loss.png', 
                figsize=(8, 5), fontsize=14)

    # Ablation BLEU
    ablation_bleu = []
    for k, v in ablation_data.items():
        greedy_score = v.get('bleu_greedy', [0])[-1] if 'bleu_greedy' in v else v.get('bleu', [0])[-1]
        beam_score = v.get('bleu_beam', [0])[-1] if 'bleu_beam' in v else None
        
        if beam_score is not None:
            ablation_bleu.append({'Model': k, 'BLEU (Greedy)': greedy_score, 'BLEU (Beam)': beam_score})
        else:
            ablation_bleu.append({'Model': k, 'Final BLEU': greedy_score})
    save_bleu_table(ablation_bleu, 'transformer_ablation_bleu_summary')

    # 4. Transformer Hyperparams
    # Focusing on Valid Loss for comparisons, but can do step loss if desired.
    # Grouping by param type for clarity.
    
    # Learning Rate
    lr_files = {
        'LR 1e-4': 'checkpoints/hyperparam_transformer_absolute_layernorm_d256_bs64_lr1e-04_history.json',
        'LR 5e-4': 'checkpoints/hyperparam_transformer_absolute_layernorm_d256_bs64_lr5e-04_history.json',
        'LR 1e-3': 'checkpoints/hyperparam_transformer_absolute_layernorm_d256_bs64_lr1e-03_history.json',
        'LR 3e-3': 'checkpoints/hyperparam_transformer_absolute_layernorm_d256_bs64_lr3e-03_history.json',
    }
    lr_data = {k: load_history(v) for k, v in lr_files.items()}
    # plot_metric(lr_data, 'step_losses', 'Transformer LR - Training Loss', 'transformer_lr_train_loss_step.png', smooth_factor=0.9)
    plot_metric(lr_data, 'valid_loss', 'Transformer LR - Validation Loss', 'transformer_lr_valid_loss.png',
                figsize=(5, 4), fontsize=14)
    
    # Batch Size
    bs_files = {
        'BS 32': 'checkpoints/hyperparam_transformer_absolute_layernorm_d256_bs32_lr1e-03_history.json',
        'BS 64': 'checkpoints/hyperparam_transformer_absolute_layernorm_d256_bs64_lr1e-03_history.json',
        'BS 128': 'checkpoints/hyperparam_transformer_absolute_layernorm_d256_bs128_lr1e-03_history.json',
    }
    bs_data = {k: load_history(v) for k, v in bs_files.items()}
    plot_metric(bs_data, 'valid_loss', 'Transformer BS - Validation Loss', 'transformer_bs_valid_loss.png',
                figsize=(5, 4), fontsize=14)
    
    # Scale
    scale_files = {
        'd_model 128': 'checkpoints/hyperparam_transformer_absolute_layernorm_d128_bs64_lr1e-04_history.json',
        'd_model 256': 'checkpoints/hyperparam_transformer_absolute_layernorm_d256_bs64_lr1e-04_history.json',
        'd_model 512': 'checkpoints/hyperparam_transformer_absolute_layernorm_d512_bs32_lr1e-04_history.json',
    }
    scale_data = {k: load_history(v) for k, v in scale_files.items()}
    plot_metric(scale_data, 'valid_loss', 'Transformer Scale - Validation Loss', 'transformer_scale_valid_loss.png',
                figsize=(5, 4), fontsize=14)

    # Hyperparams BLEU Summary (Single Table for all hyperparams)
    hyper_bleu = []
    for d_map, category in [(lr_data, 'LR'), (bs_data, 'BS'), (scale_data, 'Scale')]:
        for k, v in d_map.items():
            greedy_score = v.get('bleu_greedy', [0])[-1] if 'bleu_greedy' in v else v.get('bleu', [0])[-1]
            beam_score = v.get('bleu_beam', [0])[-1] if 'bleu_beam' in v else None
            
            if beam_score is not None:
                hyper_bleu.append({'Category': category, 'Model': k, 'BLEU (Greedy)': greedy_score, 'BLEU (Beam)': beam_score})
            else:
                hyper_bleu.append({'Category': category, 'Model': k, 'Final BLEU': greedy_score})
    save_bleu_table(hyper_bleu, 'transformer_hyperparams_bleu_summary')

    # 5. T5 Comparison (Cleaned vs Original)
    t5_files = {
        'T5 Original': 'checkpoints/t5_mt5-small_zh2en_history.json',
        'T5 Cleaned': 'checkpoints/t5_mt5-small_zh2en_cleaned_history.json'
    }
    t5_data = {k: load_history(v) for k, v in t5_files.items()}
    plot_metric(t5_data, 'step_losses', 'T5 - Training Loss (Step)', 't5_comparison_train_loss_step.png', 
                smooth_factor=0.99, figsize=(6, 5), fontsize=14) # T5 has many steps
    plot_metric(t5_data, 'valid_loss', 'T5 - Validation Loss', 't5_comparison_valid_loss.png',
                figsize=(6, 5), fontsize=14)
    
    t5_bleu = []
    for k, v in t5_data.items():
        greedy_score = v.get('bleu_greedy', [0])[-1] if 'bleu_greedy' in v else v.get('bleu', [0])[-1]
        beam_score = v.get('bleu_beam', [0])[-1] if 'bleu_beam' in v else None
        
        if beam_score is not None:
            t5_bleu.append({'Model': k, 'BLEU (Greedy)': greedy_score, 'BLEU (Beam)': beam_score})
        else:
            t5_bleu.append({'Model': k, 'Final BLEU': greedy_score})
    save_bleu_table(t5_bleu, 't5_bleu_summary')

    # 6. Decoding Strategies (Transformer)
    decoding_path = 'checkpoints/transformer_absolute_layernorm_d256_history.json'
    if not os.path.exists(decoding_path):
        decoding_path = 'checkpoints/ablation_transformer_absolute_layernorm_d256_history.json'
    
    if os.path.exists(decoding_path):
        d_hist = load_history(decoding_path)
        greedy = d_hist.get('bleu_greedy', [0])[-1]
        beam = d_hist.get('bleu_beam', [0])[-1]
        
        dec_table = [
            {'Strategy': 'Greedy Search', 'Final BLEU': greedy},
            {'Strategy': 'Beam Search', 'Final BLEU': beam}
        ]
        save_bleu_table(dec_table, 'decoding_strategies_bleu_summary')

    # 7. Overall Comparison (Valid Loss Only as Step Loss scales differ too much)
    # RNN, Transformer, T5 (Cleaned)
    comp_files = {
        'RNN (GRU Additive)': 'checkpoints/GRU_additive_tf0.8_history.json',
        'Transformer (Base)': 'checkpoints/transformer_absolute_layernorm_d256_history.json',
        'T5 (Cleaned)': 'checkpoints/t5_mt5-small_zh2en_cleaned_history.json'
    }
    # Fallback for transformer
    if not os.path.exists(comp_files['Transformer (Base)']):
        comp_files['Transformer (Base)'] = 'checkpoints/ablation_transformer_absolute_layernorm_d256_history.json'
        
    comp_data = {k: load_history(v) for k, v in comp_files.items()}
    plot_metric(comp_data, 'valid_loss', 'Model Comparison - Validation Loss', 'comparison_valid_loss.png',
                figsize=(12, 5), fontsize=16)
    
    comp_bleu = []
    for k, v in comp_data.items():
        greedy_score = v.get('bleu_greedy', [0])[-1] if 'bleu_greedy' in v else v.get('bleu', [0])[-1]
        beam_score = v.get('bleu_beam', [0])[-1] if 'bleu_beam' in v else None
        
        if beam_score is not None:
            comp_bleu.append({'Model': k, 'BLEU (Greedy)': greedy_score, 'BLEU (Beam)': beam_score})
        else:
            comp_bleu.append({'Model': k, 'Final BLEU': greedy_score})
    save_bleu_table(comp_bleu, 'overall_comparison_bleu_summary')

if __name__ == '__main__':
    print("Generating plots and tables...")
    run_experiments()
    print("Done.")
