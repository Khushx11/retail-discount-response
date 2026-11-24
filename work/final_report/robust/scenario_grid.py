"""Replay the model checkpoints over the 0–60% discount range on one known context."""
import json

import numpy as np
import pandas as pd
import study
import torch


def replay():
    pipeline = study.p
    columns = [*pipeline.NUM, *pipeline.CAT, "disc_prev", "forecast_date", "y"]
    rows = pd.read_parquet(
        study.ROOT / "out/features.parquet",
        columns=list(dict.fromkeys(columns)),
        filters=[("split", "==", "test")],
    )
    eligible = rows.loc[rows.disc_missing.eq(0)]
    if eligible.empty:
        raise ValueError("No historical test context has a recorded discount.")
    context = eligible.iloc[[0]]
    schema = json.loads((study.O / "schema.json").read_text())
    configs = json.loads((study.O / "selections.json").read_text())
    historical = float(context.disc.iloc[0])
    scenarios = pd.concat([context] * 61, ignore_index=True)
    scenarios["disc"] = np.arange(61) / 100
    scenarios["disc_change"] = scenarios.disc - scenarios.disc_prev

    tree = study.lgb.Booster(model_file=str(study.O / "lightgbm_tweedie_0.txt"))
    design = pipeline.tree_frame(scenarios, pipeline.NUM + pipeline.CAT, schema)
    tweedie = tree.predict(design, num_threads=2)
    net = study.Piecewise(schema, configs["Piecewise monotone"]["config"])
    net.load_state_dict(torch.load(
        study.O / "piecewise_monotone_0.pt", map_location="cpu", weights_only=True
    ))
    monotone = pipeline.nn_predict(net, pipeline.tensors(scenarios, schema), "poisson")
    if not np.isfinite(tweedie).all() or (np.diff(monotone) < -1e-5).any():
        raise ValueError("Checkpoint outputs failed the finite or monotonicity check.")
    return {
        "forecast_date": str(context.forecast_date.iloc[0].date()),
        "observed_discount": historical,
        "observed_units": float(context.y.iloc[0]),
        "forecast": [
            {"discount": percent, "tweedie": float(tree_result), "monotone": float(net_result)}
            for percent, (tree_result, net_result) in enumerate(zip(tweedie, monotone))
        ],
    }


if __name__ == "__main__":
    print(json.dumps(replay(), allow_nan=False))
