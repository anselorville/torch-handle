import torch
from typing import List, Tuple


def chunk_iter(seq: List[int], max_len: int, stride: int) -> List[List[int]]:
    if max_len <= 0:
        return [seq]
    chunks = []
    start = 0
    while start < len(seq):
        end = min(start + max_len, len(seq))
        chunks.append(seq[start:end])
        if end == len(seq):
            break
        start = end - stride if stride > 0 else end
        start = max(0, start)
    return chunks


@torch.no_grad()
def hierarchical_summarize(
    model,
    tokenizer,
    doc_ids: List[int],
    max_src_len: int = 8192,
    stride: int = 1024,
    max_tgt_len: int = 256,
    round2_max_src_len: int = 4096,
    device: torch.device = None,
) -> Tuple[str, List[str]]:
    """
    层级汇总：
    1) 将长文切块生成块摘要；
    2) 将块摘要拼接为第二轮输入，再生成最终摘要。
    返回：(final_summary, chunk_summaries)
    """
    device = device or next(model.parameters()).device
    chunks = chunk_iter(doc_ids, max_len=max_src_len, stride=stride)
    chunk_summaries: List[str] = []
    bos, eos = tokenizer.convert_tokens_to_ids('<bos>'), tokenizer.convert_tokens_to_ids('<eos>')
    for ch in chunks:
        src = torch.tensor([ch], dtype=torch.long, device=device)
        # 简化：贪心生成
        tgt = torch.tensor([[bos]], dtype=torch.long, device=device)
        for _ in range(max_tgt_len):
            logits = model(src, tgt)
            next_id = int(logits[:, -1, :].argmax(dim=-1))
            tgt = torch.cat([tgt, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == eos:
                break
        chunk_summaries.append(tokenizer.decode(tgt[0].tolist(), skip_special_tokens=True))

    # 第二轮
    concat_ids: List[int] = []
    for s in chunk_summaries:
        concat_ids.extend(tokenizer.encode(s))
    concat_ids = concat_ids[:round2_max_src_len]
    src2 = torch.tensor([concat_ids], dtype=torch.long, device=device)
    tgt2 = torch.tensor([[bos]], dtype=torch.long, device=device)
    for _ in range(max_tgt_len):
        logits = model(src2, tgt2)
        next_id = int(logits[:, -1, :].argmax(dim=-1))
        tgt2 = torch.cat([tgt2, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos:
            break
    final_summary = tokenizer.decode(tgt2[0].tolist(), skip_special_tokens=True)
    return final_summary, chunk_summaries


