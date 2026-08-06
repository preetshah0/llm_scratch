import torch
import torch.nn.functional as F

probs = F.softmax(logits, dim=-1)
pred_token_ids = torch.argmax(probs, dim=-1)

next_tokens = pred_token_ids[:, -1]
next_confidences = probs[torch.arange(logits.size(0)), -1, next_tokens]

for idx, (tok_id, conf) in enumerate(zip(next_tokens, next_confidences)):
    print(f"Batch {idx}: Predicted Token ID = {tok_id.item()} | Confidence = {conf.item()*100:.4f}%")

def compute_accuracy(pred_ids, target_ids):
    correct = (pred_ids == target_ids).float()
    return correct.mean().item() * 100

if 'target_ids' in locals():
    acc = compute_accuracy(pred_token_ids, target_ids)
    print(f"Model Accuracy: {acc:.2f}%")
