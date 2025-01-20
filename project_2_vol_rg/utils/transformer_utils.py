import torch


def generate_square_subsequent_mask(size):
    mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)
    return mask


def generate_text_transformer(model, start_phrase, symbol_to_index, index_to_symbol, generate_length, device):
    model.eval()
    input_seq = [symbol_to_index[char] for char in start_phrase]
    input_seq = torch.tensor(input_seq, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(generate_length):
        tgt_mask = generate_square_subsequent_mask(input_seq.size(1)).to(device)
        output = model(tgt=input_seq, memory=None, tgt_mask=tgt_mask)
        next_token = output[:, -1, :].argmax(dim=-1).item()
        input_seq = torch.cat([input_seq, torch.tensor([[next_token]], device=device)], dim=1)
        if index_to_symbol[next_token] == "<EOS>":
            break

    return "".join([index_to_symbol[idx] for idx in input_seq.squeeze().tolist()])
