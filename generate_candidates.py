# generate_candidates.py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from models import TCRTransformer
from inference import generate_tcr, decode_sequence

def load_model(model_path, vocab_dict, device,
               d_model=128, nhead=4, 
               num_encoder_layers=2, num_decoder_layers=2,
               dim_feedforward=512,
               max_src_len=55, max_tgt_len=28):
    model = TCRTransformer(
        vocab_size=len(vocab_dict),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def main():
    # 1. 加载词典
    vocab_dict = np.load('vocab_dict.npy', allow_pickle=True).item()
    for sp in ['<PAD>', '<UNK>', '<SEP>', '<SOS>', '<EOS>']:
        if sp not in vocab_dict:
            vocab_dict[sp] = len(vocab_dict)
    rev_vocab_dict = {v: k for k, v in vocab_dict.items()}

    # 2. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        model_path='best_model.pth',
        vocab_dict=vocab_dict,
        device=device,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        max_src_len=55,  # 与训练时一致
        max_tgt_len=28   # 与训练时一致
    )

    # 3. 读取需要生成 TCR 的输入 CSV
    # 假设列名是: "MHC", "Target Peptide"
    input_csv = 'tumor.csv'
    df_input = pd.read_csv(input_csv).dropna()

    results = []
    N = 20  # 每条 (MHC, Peptide) 生成候选 TCR 的条数

    for _, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Generating Candidates"):
        
        mhc = row['MHC_sequence']
        peptide = row['Peptide Sequence']
        name = row['Name1']+'_'+row['Name2']
        HLA_type = row['MHC Type']
        # 对每条输入生成 N 个候选 TCR，每个候选单独占一行
        for i in range(N):
            gen_indices = generate_tcr(
                model=model,
                mhc=mhc,
                peptide=peptide,
                vocab_dict=vocab_dict,
                device=device,
                temperature=0.8,  # 可调
                top_k=5,          # 可调
                max_len=28
            )
            tcr_str = decode_sequence(gen_indices, rev_vocab_dict)

            # 每行只有一个候选 TCR
            results.append({
                'name':name,
                'HLA_type':HLA_type,
                'MHC': mhc,
                'Target Peptide': peptide,
                'Generated_TCR': tcr_str
            })

    df_out = pd.DataFrame(results)
    output_csv = 'candidates_result.csv'
    df_out.to_csv(output_csv, index=False)
    print(f"Done! {N} candidate TCR(s) per input, saved to {output_csv}")

if __name__ == "__main__":
    main()
