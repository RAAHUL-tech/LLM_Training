import torch

def sequence_logprob(model, input_ids, attention_mask):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    logits = outputs.logits[:, :-1]
    labels = input_ids[:, 1:]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logprobs = log_probs.gather(
        2, labels.unsqueeze(-1)
    ).squeeze(-1)

    mask = attention_mask[:, 1:]
    return (token_logprobs * mask).sum(dim=1)
