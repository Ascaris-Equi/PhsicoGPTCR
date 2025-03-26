# inference.py
import torch
import numpy as np
import pandas as pd
from models import TCRTransformer

def load_model(model_path, vocab_dict, device, 
               d_model=128, nhead=2, 
               num_encoder_layers=2, num_decoder_layers=2, 
               dim_feedforward=512, 
               max_src_len=55, max_tgt_len=28):
    vocab_size = len(vocab_dict)
    model = TCRTransformer(
        vocab_size=vocab_size,
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

# 以下函数保持不变……
def generate_tcr(model, mhc: str, peptide: str, vocab_dict: dict, device,
                 temperature: float = 1.0, top_k: int = 5, max_len: int = 30):
    PAD = vocab_dict['<PAD>']
    SOS = vocab_dict['<SOS>']
    EOS = vocab_dict['<EOS>']
    SEP = vocab_dict['<SEP>']
    UNK = vocab_dict['<UNK>']

    mhc_encoded = [vocab_dict.get(aa, UNK) for aa in mhc]
    pep_encoded = [vocab_dict.get(aa, UNK) for aa in peptide]
    src_sequence = mhc_encoded + [SEP] + pep_encoded

    if len(src_sequence) > model.max_src_len:
        src_sequence = src_sequence[:model.max_src_len]
    else:
        src_sequence += [PAD] * (model.max_src_len - len(src_sequence))
    src_tensor = torch.tensor([src_sequence], dtype=torch.long, device=device)

    tgt_indices = [SOS]
    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long, device=device)
        tgt_mask = model.generate_square_subsequent_mask(tgt_tensor.size(1)).to(device)

        src_key_padding_mask = (src_tensor == PAD)
        tgt_key_padding_mask = (tgt_tensor == PAD)

        with torch.no_grad():
            output = model(
                src=src_tensor,
                tgt=tgt_tensor,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask
            )
            logits = output[0, -1, :] / temperature
        top_k_logits, top_k_indices = torch.topk(logits, k=top_k)
        probs = torch.softmax(top_k_logits, dim=-1)
        next_token = top_k_indices[torch.multinomial(probs, 1)].item()
        if next_token == EOS:
            break
        tgt_indices.append(next_token)
    return tgt_indices

def decode_sequence(seq_indices, rev_vocab_dict):
    special_tokens = {'<PAD>', '<SOS>', '<EOS>', '<SEP>', '<UNK>'}
    decoded = []
    for idx in seq_indices:
        token = rev_vocab_dict.get(idx, '<UNK>')
        if token == '<EOS>':
            break
        if token in special_tokens:
            continue
        decoded.append(token)
    return "".join(decoded)
