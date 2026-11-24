"""Builds the long product-store-week table used by both models.

One row = one product in one store in one week of its life (weeks 1 to 11).
Week 0 is dropped as a target because it has no sales history to lean on.
"""
import numpy as np, pandas as pd

W = [str(i) for i in range(12)]

def load(data='data'):
    s = pd.read_csv(f'{data}/sales.csv', index_col=0)
    d = pd.read_csv(f'{data}/price_discount_series.csv')
    r = pd.read_csv(f'{data}/restocks.csv')
    g = pd.read_csv(f'{data}/vis2_gtrends_data.csv', parse_dates=['date'])
    return s, d, r, g

def stock_matrix(s, r):
    """Units delivered to each series per life-week, from the weekly restock log.
    Deliveries before launch (t<0) get folded into week 0 as opening stock."""
    s = s.reset_index(drop=True)
    key = s[['external_code', 'retail']].copy(); key['row'] = np.arange(len(s))
    key['rd'] = pd.to_datetime(s.release_date)
    r = r.copy()
    r['wk'] = pd.to_datetime(r.year.astype(str) + '-' + r.week.astype(str) + '-1', format='%G-%V-%u')
    m = r.merge(key, on=['external_code', 'retail'])
    m['t'] = (m.wk - m.rd).dt.days // 7
    m = m[m.t <= 11]
    m['t'] = m.t.clip(lower=0)
    deliv = np.zeros((len(s), 12))
    np.add.at(deliv, (m.row.values, m.t.values), m.qty.values)
    return deliv

def build(data='data'):
    s, d, r, g = load(data)
    s = s.reset_index(drop=True)
    df = s.merge(d, on=['external_code', 'retail'], suffixes=('', '_d'), how='left', validate='one_to_one', indicator=True)
    assert df['_merge'].eq('both').all(), 'Unmatched discount keys'
    assert len(df) == len(s)
    assert df[['external_code', 'retail']].equals(s[['external_code', 'retail']])
    S_raw = df[W].values.astype(float)
    S = np.clip(S_raw, 0, None)     # 0.3% of weeks are net returns (-1 to -3), treat as zero demand
    D = np.nan_to_num(df[[w + '_d' for w in W]].values.astype(float))
    deliv = stock_matrix(s, r)

    cum_deliv = np.cumsum(deliv, 1)                      # delivered up to and incl. week t
    cum_sales_prev = np.concatenate([np.zeros((len(S), 1)), np.cumsum(S_raw, 1)[:, :-1]], 1)
    # Forecast at start of target week: realized deliveries DURING that week are unknown.
    deliveries_before = np.concatenate([np.zeros((len(S), 1)), cum_deliv[:, :-1]], 1)
    inv = deliveries_before - cum_sales_prev


    rd = pd.to_datetime(df.release_date)
    # google trends for the category, 52 weeks before release (VISUELLE convention)
    g = g.set_index('date').sort_index()

    rows = []
    for t in range(1, 12):
        f = pd.DataFrame({
            'row': np.arange(len(df)), 't': t,
            'y': S[:, t],
            'disc': D[:, t], 'disc_prev': D[:, t - 1],
            'disc_change': D[:, t] - D[:, t - 1],
            'disc_max_prev': D[:, :t].max(1),
            'weeks_on_disc_prev': (D[:, :t] > 0).sum(1),
            'lag1': S[:, t - 1],
            'lag2': S[:, t - 2] if t >= 2 else np.nan,
            'lag3': S[:, t - 3] if t >= 3 else np.nan,
            'mean_sofar': S[:, :t].mean(1),
            'cum_sales': S[:, :t].sum(1),
            'inv_start': inv[:, t], 'deliv_prev': deliv[:, t - 1],
            'opening_stock': deliv[:, 0],
            'stockout_start': (inv[:, t] <= 0).astype(int),
        })
        rows.append(f)
    long = pd.concat(rows, ignore_index=True)
    meta = df[['external_code', 'retail', 'season', 'category', 'color', 'fabric', 'price']].copy()
    meta['release_date'] = rd
    meta['rel_month'] = rd.dt.month
    meta['rel_woy'] = rd.dt.isocalendar().week.astype(int)
    meta['season_type'] = df.season.str[:2]
    long = long.join(meta, on='row')
    long['forecast_date'] = long.release_date + pd.to_timedelta(long.t * 7, unit='D')
    long['label_end'] = long.forecast_date + pd.Timedelta(days=7)
    long['cal_woy'] = long.forecast_date.dt.isocalendar().week.astype(int)
    long['disc_missing'] = df[[w + '_d' for w in W]].isna().to_numpy()[long.row.to_numpy(), long.t.to_numpy()].astype(int)

    # trend: mean google trends level of the category over the 52 weeks pre-release
    cat_tr = {}
    for c in df.category.unique():
        if c in g.columns:
            cat_tr[c] = g[c]
    def trend(rel, cat):
        if cat not in cat_tr: return np.nan
        x = cat_tr[cat]; x = x[(x.index < rel) & (x.index >= rel - pd.Timedelta(weeks=52))]
        return x.mean() if len(x) else np.nan
    pairs = meta[['release_date', 'category']].drop_duplicates()
    pairs['gtrend'] = [trend(a, b) for a, b in zip(pairs.release_date, pairs.category)]
    long = long.merge(pairs, on=['release_date', 'category'], how='left')

    # Retrospective fixed chronological split; every preceding target week is complete.
    long['split'] = np.where(long.forecast_date < '2018-11-05', 'train',
                    np.where(long.forecast_date < '2019-05-06', 'val', 'test'))
    for a, b in [('train', 'val'), ('val', 'test')]:
        assert long.loc[long.split == a, 'label_end'].max() <= long.loc[long.split == b, 'forecast_date'].min()
    assert not {'prod_lag1', 'prod_mean_sofar', 'inv', 'deliv_t', 'stockout'} & set(long.columns)
    return long
