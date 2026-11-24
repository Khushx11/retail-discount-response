"""Keep a rolling evaluation only if its summaries and predictions agree."""

MODELS = {
    "LightGBM Tweedie", "MLP Poisson", "Piecewise monotone", "Paired free control",
    "Naive last week", "Naive mean so far",
}


def load_complete(summaries, predictions):
    completed = []
    frames = []
    if "origin" not in predictions or not MODELS.issubset(predictions.columns):
        return completed, frames
    for origin in dict.fromkeys(row["origin"] for row in summaries):
        rows = [row for row in summaries if row["origin"] == origin]
        if len(rows) != len(MODELS) or {row["model"] for row in rows} != MODELS:
            continue
        part = predictions.loc[predictions.origin == origin]
        if not len(part) or {row["test_rows"] for row in rows} != {len(part)}:
            continue
        if part[sorted(MODELS)].isna().to_numpy().any():
            continue
        completed.extend(rows)
        frames.append(part)
    return completed, frames
