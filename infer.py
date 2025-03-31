import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import os
import re

def create_vocab():
    """创建统一的氨基酸词汇表，包含标准氨基酸和特殊 token"""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY-"
    # 先建立基本词汇
    vocab = {aa: i for i, aa in enumerate(amino_acids)}
    # 加入特殊 token
    special_tokens = ['<PAD>', '<UNK>', '<SEP>', '< SOS >', '<EOS>']
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab

def get_amino_acid_features():
    """返回氨基酸理化性质特征字典（包含分子量信息，归一化后约在0~1之间）"""
    features = {
        'A': [0, 0, 0, 0, 89.1/200],
        'C': [0, 0, 0, 0, 121.2/200],
        'D': [0, 1, 0, 0, 133.1/200],
        'E': [0, 1, 0, 0, 147.1/200],
        'F': [1, 0, 0, 0, 165.2/200],
        'G': [0, 0, 0, 0, 75.1/200],
        'H': [0, 0, 1, 0, 155.2/200],
        'I': [0, 0, 0, 0, 131.2/200],
        'K': [0, 0, 1, 0, 146.2/200],
        'L': [0, 0, 0, 0, 131.2/200],
        'M': [0, 0, 0, 0, 149.2/200],
        'N': [0, 0, 0, 1, 132.1/200],
        'P': [0, 0, 0, 0, 115.1/200],
        'Q': [0, 0, 0, 1, 146.1/200],
        'R': [0, 0, 1, 0, 174.2/200],
        'S': [0, 0, 0, 1, 105.1/200],
        'T': [0, 0, 0, 1, 119.1/200],
        'V': [0, 0, 0, 0, 117.1/200],
        'W': [1, 0, 0, 0, 204.2/200],
        'Y': [1, 0, 0, 1, 181.2/200],
        '-': [0, 0, 0, 0, 0],
        '<PAD>': [0, 0, 0, 0, 0],
        '<UNK>': [0, 0, 0, 0, 0],
        '<SEP>': [0, 0, 0, 0, 0],
        '< SOS >': [0, 0, 0, 0, 0],
        '<EOS>': [0, 0, 0, 0, 0],
    }
    return features

class InputEncoder(nn.Module):
    """
    将氨基酸序列的各种特征编码映射到统一的表示空间，
    使用门控机制融合 token 嵌入、理化特征嵌入和位置嵌入。
    """
    def __init__(self, vocab_size, d_model, max_len=1000):
        super(InputEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token嵌入：输出维度为 d_model // 2
        self.token_embedding = nn.Embedding(vocab_size, d_model // 2)
        # 理化性质嵌入：输入维度为 5（包含分子量），输出为 d_model // 4
        self.property_embedding = nn.Linear(5, d_model // 4)
        # 位置嵌入：输出维度为 d_model // 4
        self.position_embedding = nn.Embedding(max_len, d_model // 4)
        
        # 进行特征融合，先通过门控网络再映射到最终表示空间
        self.gate = nn.Linear(d_model, d_model)
        self.fusion_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, tokens, properties, positions):
        """
        参数:
            tokens: [batch_size, seq_len] - 序列的 token ID
            properties: [batch_size, seq_len, 5] - 每个 token 的理化性质（包含分子量）
            positions: [batch_size, seq_len] - 位置 ID
        """
        token_emb = self.token_embedding(tokens)              # [B, L, d_model//2]
        property_emb = self.property_embedding(properties)      # [B, L, d_model//4]
        position_emb = self.position_embedding(positions)       # [B, L, d_model//4]
        
        # 拼接得到总特征 [B, L, d_model]
        combined = torch.cat([token_emb, property_emb, position_emb], dim=2)
        # 门控机制：计算gate权重 (范围0~1)
        gate_weights = torch.sigmoid(self.gate(combined))
        fused = gate_weights * combined
        output = self.fusion_linear(fused)
        output = self.dropout(output)
        output = self.layer_norm(output)
        return output

class TCRTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, max_mhc_len, max_peptide_len, max_tcr_len, vocab_dict):
        super(TCRTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_mhc_len = max_mhc_len
        self.max_peptide_len = max_peptide_len
        self.max_src_len = max_mhc_len + max_peptide_len + 1  # 加上 <SEP> 分隔符
        self.max_tcr_len = max_tcr_len
        self.max_tgt_len = max_tcr_len + 2  # 加上 < SOS > 和 <EOS>
        self.vocab_dict = vocab_dict
        self.d_model = d_model
        
        self.encoder_input_processor = InputEncoder(vocab_size, d_model, max_len=self.max_src_len)
        self.decoder_input_processor = InputEncoder(vocab_size, d_model, max_len=self.max_tgt_len)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, tgt, src_properties, tgt_properties, 
                src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        batch_size = src.size(0)
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        
        src_positions = torch.arange(src_len, device=src.device).unsqueeze(0).expand(batch_size, -1)
        tgt_positions = torch.arange(tgt_len, device=tgt.device).unsqueeze(0).expand(batch_size, -1)
        
        src_emb = self.encoder_input_processor(src, src_properties, src_positions)
        tgt_emb = self.decoder_input_processor(tgt, tgt_properties, tgt_positions)
        
        # Transformer要求的输入格式：[seq_len, batch_size, d_model]
        src_emb = src_emb.transpose(0, 1)
        tgt_emb = tgt_emb.transpose(0, 1)
        
        output = self.transformer(src_emb, tgt_emb, 
                                  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask, 
                                  tgt_mask=tgt_mask)
        
        output = output.transpose(0, 1)
        return self.fc_out(output)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def generate_tcr_beam(model, mhc, peptide, vocab_dict, device, beam_width=5, max_len=28, temperature=0.8):
    aa_features = get_amino_acid_features()
    
    # 构造源序列及对应理化特征
    mhc_encoded = [vocab_dict.get(aa, vocab_dict['<UNK>']) for aa in mhc]
    peptide_encoded = [vocab_dict.get(aa, vocab_dict['<UNK>']) for aa in peptide]
    src_sequence = mhc_encoded + [vocab_dict['<SEP>']] + peptide_encoded
    src_raw = list(mhc) + ['<SEP>'] + list(peptide)
    src_properties = [aa_features.get(aa, aa_features['<UNK>']) for aa in src_raw]
    
    src_tensor = torch.tensor([src_sequence], dtype=torch.long).to(device)
    src_properties = torch.tensor([src_properties], dtype=torch.float).to(device)
    if src_tensor.size(1) < model.max_src_len:
        pad_size = model.max_src_len - src_tensor.size(1)
        padding = torch.full((1, pad_size), vocab_dict['<PAD>'], dtype=torch.long).to(device)
        src_tensor = torch.cat([src_tensor, padding], dim=1)
        prop_padding = torch.zeros((1, pad_size, src_properties.size(-1)), dtype=torch.float).to(device)
        src_properties = torch.cat([src_properties, prop_padding], dim=1)
    
    sos_token = vocab_dict['< SOS >']
    eos_token = vocab_dict['<EOS>']
    init_seq = [sos_token]
    init_prop = [aa_features['< SOS >']]
    candidates = [{'seq': init_seq, 'log_prob': 0.0, 'tgt_properties': torch.tensor(init_prop, dtype=torch.float).to(device)}]
    completed_candidates = []
    
    for step in range(1, max_len + 1):
        all_candidates = []
        alive_candidates = []
        for cand in candidates:
            if cand['seq'][-1] == eos_token:
                completed_candidates.append(cand)
            else:
                alive_candidates.append(cand)
        if len(alive_candidates) == 0:
            break
        
        batch_seq = [cand['seq'] for cand in alive_candidates]
        batch_props = [cand['tgt_properties'] for cand in alive_candidates]
        batch_seq_tensor = torch.tensor(batch_seq, dtype=torch.long).to(device)
        L = batch_seq_tensor.size(1)
        tgt_positions = torch.arange(L, device=device).unsqueeze(0).expand(len(batch_seq), L)
        batch_props_tensor = torch.stack(batch_props, dim=0)
        tgt_mask = model.generate_square_subsequent_mask(L).to(device)
        tgt_padding_mask = (batch_seq_tensor == vocab_dict['<PAD>'])
        src_padding_mask = (src_tensor == vocab_dict['<PAD>'])
        batch_size = batch_seq_tensor.size(0)
        expanded_src = src_tensor.expand(batch_size, -1)
        expanded_src_props = src_properties.expand(batch_size, -1, src_properties.size(-1))
        
        with torch.no_grad():
            output = model(expanded_src, batch_seq_tensor, expanded_src_props, batch_props_tensor,
                           src_key_padding_mask=src_padding_mask.expand(batch_size, -1), 
                           tgt_key_padding_mask=tgt_padding_mask,
                           tgt_mask=tgt_mask)
        last_step_logits = output[:, -1, :] / temperature
        log_probs = torch.log_softmax(last_step_logits, dim=-1)
        topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
        rev_vocab_dict = {v: k for k, v in vocab_dict.items()}
        for i, cand in enumerate(alive_candidates):
            for j in range(beam_width):
                new_token = topk_indices[i, j].item()
                new_log_prob = cand['log_prob'] + topk_log_probs[i, j].item()
                new_seq = cand['seq'] + [new_token]
                new_token_aa = rev_vocab_dict.get(new_token, '<UNK>')
                new_prop = aa_features.get(new_token_aa, aa_features['<UNK>'])
                new_tgt_props = torch.cat([cand['tgt_properties'], torch.tensor([new_prop], dtype=torch.float).to(device)], dim=0)
                new_cand = {'seq': new_seq, 'log_prob': new_log_prob, 'tgt_properties': new_tgt_props}
                all_candidates.append(new_cand)
        all_candidates = sorted(all_candidates, key=lambda x: x['log_prob'], reverse=True)
        candidates = all_candidates[:beam_width]
    
    completed_candidates.extend(candidates)
    finished = [c for c in completed_candidates if c['seq'][-1] == eos_token]
    if finished:
        best_candidate = max(finished, key=lambda x: x['log_prob'])
    else:
        best_candidate = max(completed_candidates, key=lambda x: x['log_prob'])
    
    result_seq = best_candidate['seq']
    # 去除初始的 < SOS >，并取到 <EOS> 前的序列
    result_seq = result_seq[1:]
    if eos_token in result_seq:
        eos_index = result_seq.index(eos_token)
        result_seq = result_seq[:eos_index]
    return result_seq

def generate_multiple_tcrs(model, mhc, peptide, vocab_dict, device, num_tcrs=1, max_len=28, 
                          temperatures=None, beam_widths=None):
    """为同一个peptide-MHC对生成多个TCR序列"""
    if temperatures is None:
        # 为了增加多样性，使用不同的温度值
        base_temp = 0.8
        temperatures = [base_temp + 0.1 * (i - num_tcrs // 2) / (num_tcrs // 2 + 1) for i in range(num_tcrs)]
    
    if beam_widths is None:
        # 使用不同的beam宽度
        beam_widths = [max(3, min(5 + i // 2, 10)) for i in range(num_tcrs)]
    
    # 确保温度和beam宽度数量匹配生成数量
    if len(temperatures) < num_tcrs:
        temperatures.extend([0.8] * (num_tcrs - len(temperatures)))
    if len(beam_widths) < num_tcrs:
        beam_widths.extend([5] * (num_tcrs - len(beam_widths)))
    
    rev_vocab_dict = {v: k for k, v in vocab_dict.items()}
    generated_tcrs = []
    
    for i in range(num_tcrs):
        # 使用不同的温度和beam宽度以增加多样性
        temp = temperatures[i]
        beam_width = beam_widths[i]
        
        tcr_tokens = generate_tcr_beam(model, mhc, peptide, vocab_dict, device, 
                                        beam_width=beam_width, max_len=max_len, 
                                        temperature=temp)
        
        # 解码TCR序列
        tcr_sequence = decode_sequence(tcr_tokens, rev_vocab_dict)
        generated_tcrs.append(tcr_sequence)
    
    return generated_tcrs

def decode_sequence(seq, rev_vocab_dict):
    """将token ID转换回氨基酸序列"""
    decoded = []
    for token in seq:
        aa = rev_vocab_dict.get(token, '<UNK>')
        if aa == '<EOS>':
            break
        if aa not in ['<PAD>', '<UNK>', '< SOS >']:
            decoded.append(aa)
    return ''.join(decoded) if decoded else "<empty>"

def extract_sequence_lengths_from_error(error_msg):
    """从错误消息中提取序列长度信息"""
    encoder_match = re.search(r'encoder_input_processor\.position_embedding\.weight: copying a param with shape torch\.Size\(\[(\d+), \d+\]\)', error_msg)
    decoder_match = re.search(r'decoder_input_processor\.position_embedding\.weight: copying a param with shape torch\.Size\(\[(\d+), \d+\]\)', error_msg)
    
    encoder_len = int(encoder_match.group(1)) if encoder_match else None
    decoder_len = int(decoder_match.group(1)) if decoder_match else None
    
    return encoder_len, decoder_len

def main():
    parser = argparse.ArgumentParser(description='TCR序列生成推理脚本')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径，包含peptide和mhc列')
    parser.add_argument('--output', type=str, required=True, help='输出CSV文件路径')
    parser.add_argument('--model', type=str, required=True, help='模型检查点文件路径')
    parser.add_argument('--num_tcrs', type=int, default=1, help='每个peptide-MHC对生成的TCR数量')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='使用的设备(cuda或cpu)')
    parser.add_argument('--max_len', type=int, default=28, help='生成TCR的最大长度')
    parser.add_argument('--temperatures', type=float, nargs='+', help='生成多样化TCR的温度参数，数量应与num_tcrs匹配')
    parser.add_argument('--beam_widths', type=int, nargs='+', help='beam搜索宽度，数量应与num_tcrs匹配')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='模型嵌入维度')
    parser.add_argument('--nhead', type=int, default=2, help='Transformer注意力头数')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--num_decoder_layers', type=int, default=2, help='解码器层数')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='前馈网络维度')
    
    # 特别注意：这些长度参数必须与训练时使用的完全匹配
    parser.add_argument('--max_mhc_len', type=int, default=40, help='MHC序列最大长度')
    parser.add_argument('--max_peptide_len', type=int, default=14, help='肽段序列最大长度')
    parser.add_argument('--max_tcr_len', type=int, default=26, help='TCR序列最大长度')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("警告：CUDA不可用，将使用CPU进行推理")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    # 创建词汇表
    vocab_dict = create_vocab()
    vocab_size = len(vocab_dict)
    
    # 加载输入数据
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    
    input_data = pd.read_csv(args.input)
    if 'peptide' not in input_data.columns or 'mhc' not in input_data.columns:
        raise ValueError("输入CSV必须包含'peptide'和'mhc'列")
    
    # 修正匹配训练模型的序列长度 - 基于原始训练时的确切值
    # 从错误信息中我们知道:
    # encoder_seq_len = 55 (max_src_len)
    # decoder_seq_len = 28 (max_tgt_len)
    src_seq_len = 55
    tgt_seq_len = 28
    
    # 由于 max_src_len = max_mhc_len + max_peptide_len + 1
    # 而 max_tgt_len = max_tcr_len + 2
    # 我们可以反推:
    max_tcr_len = tgt_seq_len - 2  # 26
    
    # 对于MHC和peptide的长度，我们需要一个合理的分配
    # 这里我们根据经验分配：MHC通常比肽段长
    max_mhc_len = 40
    max_peptide_len = src_seq_len - max_mhc_len - 1  # 14
    
    print(f"使用序列长度参数: max_mhc_len={max_mhc_len}, max_peptide_len={max_peptide_len}, max_tcr_len={max_tcr_len}")
    print(f"对应的 max_src_len={max_mhc_len + max_peptide_len + 1}, max_tgt_len={max_tcr_len + 2}")
    
    # 初始化模型 - 使用计算得到的正确长度
    model = TCRTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        max_mhc_len=max_mhc_len,
        max_peptide_len=max_peptide_len,
        max_tcr_len=max_tcr_len,
        vocab_dict=vocab_dict
    ).to(device)
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(args.model, map_location=device))
        print(f"成功加载模型 {args.model}")
    except Exception as e:
        error_msg = str(e)
        print(f"错误: 无法加载模型。详细信息: {error_msg}")
        print("尝试从错误信息中推断正确的序列长度...")
        
        try:
            encoder_len, decoder_len = extract_sequence_lengths_from_error(error_msg)
            if encoder_len and decoder_len:
                print(f"从错误信息检测到序列长度: encoder={encoder_len}, decoder={decoder_len}")
                # 反推具体参数
                max_tcr_len = decoder_len - 2
                max_mhc_len = 40  # 合理估计
                max_peptide_len = encoder_len - max_mhc_len - 1
                
                print(f"调整序列长度参数: max_mhc_len={max_mhc_len}, max_peptide_len={max_peptide_len}, max_tcr_len={max_tcr_len}")
                
                # 使用推断的参数重新创建模型
                model = TCRTransformer(
                    vocab_size=vocab_size,
                    d_model=args.d_model,
                    nhead=args.nhead,
                    num_encoder_layers=args.num_encoder_layers,
                    num_decoder_layers=args.num_decoder_layers,
                    dim_feedforward=args.dim_feedforward,
                    max_mhc_len=max_mhc_len,
                    max_peptide_len=max_peptide_len,
                    max_tcr_len=max_tcr_len,
                    vocab_dict=vocab_dict
                ).to(device)
                
                # 再次尝试加载
                model.load_state_dict(torch.load(args.model, map_location=device))
                print("使用调整后的参数成功加载模型")
            else:
                raise ValueError("无法从错误信息中提取序列长度")
        except Exception as e2:
            print(f"自动调整参数失败: {str(e2)}")
            # 给用户提供明确指示
            print("\n请尝试使用以下命令，手动指定正确的序列长度:")
            print("python infer.py --input tumor.csv --output results.csv --model best_model1.pth --num_tcrs 5 " + 
                  "--max_mhc_len 40 --max_peptide_len 14 --max_tcr_len 26")
            return
    
    # 设置为评估模式
    model.eval()
    
    # 准备输出数据结构
    output_data = []
    
    # 处理每个输入样本
    print(f"开始处理 {len(input_data)} 个样本，每个样本生成 {args.num_tcrs} 个TCR序列...")
    for idx, row in tqdm(input_data.iterrows(), total=len(input_data)):
        peptide = row['peptide']
        mhc = row['mhc']
        
        # 生成多个TCR
        tcrs = generate_multiple_tcrs(
            model, mhc, peptide, vocab_dict, device, 
            num_tcrs=args.num_tcrs,
            max_len=args.max_len,
            temperatures=args.temperatures,
            beam_widths=args.beam_widths
        )
        
        # 将结果添加到输出数据中
        for tcr in tcrs:
            output_data.append({
                'peptide': peptide,
                'mhc': mhc,
                'tcr': tcr
            })
    
    # 保存结果到CSV
    output_df = pd.DataFrame(output_data)
    output_df.to_csv(args.output, index=False)
    print(f"处理完成，结果已保存到 {args.output}")
    print(f"生成了 {len(output_data)} 条TCR序列")

if __name__ == "__main__":
    main()