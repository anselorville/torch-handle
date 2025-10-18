import typing as _t

try:
    from rouge_score import rouge_scorer
except Exception as _e:  # noqa: F401
    rouge_scorer = None


def compute_rouge(preds: _t.List[str], refs: _t.List[str], use_stemmer: bool = True) -> dict:
    if rouge_scorer is None:
        raise RuntimeError('rouge-score 未安装，请先安装 requirements。')
    assert len(preds) == len(refs)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
    agg = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    n = len(preds)
    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        for k in agg:
            agg[k] += scores[k].fmeasure
    for k in agg:
        agg[k] /= max(1, n)
    return agg


