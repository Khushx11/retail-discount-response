# Retail Discount Response

Khush Bhavin Maru · COMP5064 / PAIML2

This project forecasts the next week’s observed unit sales for a fashion product at a store after its first week on sale. It compares sales-history and seasonal features, start-of-week stock estimates and planned discounts across LightGBM, neural networks and simple baselines. A piecewise neural model constrains predicted sales to increase or stay flat as the supplied discount rises, with other inputs fixed.

## Interview dashboard

Run `python portfolio_demo.py` from the repository root and open http://127.0.0.1:8002. The aggregate mode uses the reported model comparisons and discount-support checks and runs with the Python standard library.

With local access to the original evidence, install the modelling dependencies and launch:

```sh
python portfolio_demo.py --evidence /path/to/COMP5064_Evidence
```

The dashboard loads two seed-zero checkpoints and evaluates them at 61 discount settings on the first historical test context with a recorded discount. It keeps the other inputs fixed. The observed sales and discount come from that historical context. The support chart describes a separate sample of 10,000 test contexts and covers seven tested levels. It should not be read as a support rating for the example. [Reproduction and access requirements](docs/reproduce.md)

## Results

On 405,992 historical product–store–week rows, LightGBM Tweedie had the lowest mean WAPE among the evaluated models.

- **LightGBM Tweedie** · 67.48% WAPE
- **Poisson MLP** · 69.50% WAPE
- **Monotone piecewise network** · 72.92% WAPE
- **Last-week baseline** · 97.31% WAPE

Model-family scores average three seeds. WAPE is the sum of absolute unit errors divided by the sum of observed units. The conservative-input tree scored 76.31% WAPE in a single run with target-week discount inputs and Google Trends removed.

![Forecast error after removing feature groups, and feature importance for the selected tree](docs/figures/ablations.png)

See [experiment notes](docs/experiments.md) for the split, feature availability, ablations, scenario figures and limitations. See [reproduction steps](docs/reproduce.md) for the environment, data inputs and commands.

## Dataset

Get VISUELLE 2.0 from the publisher. The [dataset setup and verification steps](docs/reproduce.md#verify-the-source-csvs) cover the four required files and the exact build used in this study.

## Credits

Coursework by Khush Bhavin Maru. The preserved source files were imported from the supplied final evidence snapshot. New repository packaging and documentation were prepared with Codex.

Dataset from Humatics Lab and Nuna Lie. Cite Geri Skenderi, Christian Joppi, Matteo Denitto, Berniero Scarpa and Marco Cristani. *The Multi-Modal Universe of Fast-Fashion: The Visuelle 2.0 Benchmark*, CVPR Workshops, 2022, pp. 2241–2246. [Dataset and access form](https://humaticslab.github.io/forecasting/visuelle) · [Paper](https://openaccess.thecvf.com/content/CVPR2022W/CVFAD/html/Skenderi_The_Multi-Modal_Universe_of_Fast-Fashion_The_Visuelle_2.0_Benchmark_CVPRW_2022_paper.html)
