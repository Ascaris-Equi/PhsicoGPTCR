# models.py
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        position_embeddings = self.position_embeddings(position_ids)
        return x + position_embeddings

class TCRTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward, max_src_len, max_tgt_len):
        super(TCRTransformer, self).__init__()

        self.vocab_size = vocab_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max(max_src_len, max_tgt_len))

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None,
                tgt_key_padding_mask=None, tgt_mask=None):
        # src: [batch_size, src_seq_len]
        # tgt: [batch_size, tgt_seq_len]
        src_emb = self.embedding(src)  # [batch_size, src_seq_len, d_model]
        tgt_emb = self.embedding(tgt)  # [batch_size, tgt_seq_len, d_model]

        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)

        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask
        )
        logits = self.fc_out(output)
        return logits

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
