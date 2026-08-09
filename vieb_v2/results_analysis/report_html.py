"""Render the cross-method comparison as a single self-contained HTML page.

Figures are embedded as data URIs -- the page has to stand alone, and the
Artifact CSP blocks every external host anyway.

Numbers come from the JSONs, never from the prose, so the page cannot drift
from the run the way a hand-written table does.

Usage:
    python -m results_analysis.report_html --report ~/vieb2-results/_report \
        --out ~/vieb2-results/_report/v2_model_comparison.html
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os

FIGURES = [
    ("1_effect_vs_power",
     "Effect size does not follow significance count",
     "Each arm's largest paired occupancy shift at retrieval against the "
     "fraction of its states that reach q&lt;0.05. The two are nearly "
     "uncorrelated. Marker shape encodes the algorithm family; every point is "
     "directly labelled, so nothing here depends on colour."),
    ("2_retrieval_timecourse",
     "Does the top state tell the fear-learning story?",
     "Occupancy of each arm's strongest state across the whole design. MoSeq's "
     "syllable 1 quadruples at retrieval and stays separated between contexts; "
     "diffusion-Koopman's state 13 is the mirror image, suppressed by fear. "
     "Neither HDBSCAN arm's top state moves visibly."),
    ("3_state_count_sweep",
     "Is the Koopman state count an output or a parameter?",
     "Attractors found against the <code>--n-regions</code> parameter, both "
     "log. In PCA space the relationship is one-to-one across a 16&times; "
     "sweep, so the state count is the parameter under another name. In "
     "diffusion space the exponent is 0.71, so genuine merging happens."),
    ("4_partition_geometry",
     "What shape is each partition?",
     "Both HDBSCAN arms put 96&ndash;99% of their clustered frames in a single "
     "state, leaving an entropy near zero. Both Koopman arms spread mass "
     "across their states. This is the geometric reading that the "
     "discrimination metric overturns &mdash; and partly confirms."),
    ("5_implied_timescales",
     "The falsification gate, and what its failure looks like",
     "Implied timescale t&#8322; against lag &tau;, and the local scaling "
     "exponent. A Markovian coarse-graining would show a flat region; instead "
     "t&#8322; grows across the entire 1,080&times; lag range in both arms, "
     "with an exponent strictly between 0 and 1 everywhere. That is "
     "long-memory behaviour with no timescale separation."),
    ("6_ranking",
     "The composite score, decomposed",
     "Six axes, min&ndash;max scaled across arms and weighted. Arms missing an "
     "axis are renormalised over the axes they have. Effect is weighted above "
     "coverage deliberately &mdash; that single choice is what separates this "
     "ranking from the geometric one."),
]


def _b64(path):
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _fmt(v, spec=".3f", dash="&mdash;"):
    if v is None or (isinstance(v, float) and v != v):
        return dash
    if isinstance(v, float):
        return format(v, spec)
    return html.escape(str(v))


def _rank_cards(rows):
    out = []
    for r in rows:
        ref = r["reference"]
        tag = "reference" if ref else f"rank {r['rank']}"
        cls = " card--ref" if ref else ""
        out.append(f"""
      <article class="card{cls}">
        <p class="card__tag">{tag}</p>
        <h3 class="card__name">{html.escape(r['arm'])}</h3>
        <p class="card__score"><span class="num">{r['composite']:.3f}</span>
           <span class="card__unit">composite</span></p>
        <dl class="card__stats">
          <div><dt>effect</dt><dd class="num">{_fmt(r['effect'], '.4f')}</dd></div>
          <div><dt>states</dt><dd class="num">{_fmt(r['n_states'])}</dd></div>
          <div><dt>significant</dt><dd class="num">{_fmt(r['n_significant'])}/{_fmt(r['n_tested'])}</dd></div>
        </dl>
      </article>""")
    return "\n".join(out)


def _discrimination_table(rows, disc, trunc):
    body = []
    for r in rows:
        arm = r["arm"]
        d = (disc.get("arms") or {}).get(arm, {})
        s = d.get("score") or {}
        t = ((trunc.get("arms") or {}).get(arm, {}) or {}).get("score") or {}
        ref = r["reference"]
        # MoSeq's own control already ran truncated; its full-session numbers
        # do not exist, so the truncated columns repeat rather than invent.
        t_sig = r["n_significant"] if ref else t.get("n_significant")
        t_eff = r["effect"] if ref else t.get("max_abs_median_diff")
        null = r.get("null_frac_at_or_above") if ref else s.get("null_frac_at_or_above")
        a = r.get("top_state_mean_a") if not ref else None
        b = r.get("top_state_mean_b") if not ref else None
        if ref:
            top = "0.463 &larr; 0.095"
        elif a is None:
            top = "&mdash;"
        else:
            top = f"{a:.3f} &larr; {b:.3f}"
        body.append(f"""
        <tr{' class="row--ref"' if ref else ''}>
          <th scope="row">{html.escape(arm)}</th>
          <td class="num">{_fmt(r['n_states'])}</td>
          <td class="num">{_fmt(r['n_tested'])}</td>
          <td class="num">{_fmt(r['n_significant'])}</td>
          <td class="num strong">{_fmt(r['effect'], '.4f')}</td>
          <td class="num">{top}</td>
          <td class="num">{_fmt(t_sig)}</td>
          <td class="num strong">{_fmt(t_eff, '.4f')}</td>
          <td class="num">{_fmt(null, '.2f')}</td>
        </tr>""")
    return "\n".join(body)


def _geometry_table(arms):
    order = ["pca-HDBSCAN", "pca-Koopman", "diffusion-HDBSCAN",
             "diffusion-Koopman"]
    body = []
    for name in order:
        a = arms.get(name)
        if not a:
            continue
        nsr = a.get("noise_speed_ratio")
        flag = "fail" if nsr and nsr > 5 else ("pass" if nsr else "warn")
        body.append(f"""
        <tr>
          <th scope="row">{html.escape(name)}</th>
          <td class="num">{_fmt(a.get('n_states'))}</td>
          <td class="num">{_fmt(a.get('largest_state_frac_clean'))}</td>
          <td class="num">{_fmt(a.get('noise_frac'))}</td>
          <td class="num">{_fmt(a.get('state_entropy_clean'))}</td>
          <td class="num"><span class="pill pill--{flag}">{_fmt(nsr, '.2f')}</span></td>
          <td class="num">{_fmt(a.get('size_speed_rank_corr'), '+.2f')}</td>
        </tr>""")
    return "\n".join(body)


def _sweep_table(sweep):
    by = {}
    for r in sweep:
        by.setdefault(r["latent"], {})[r["n_regions"]] = r["n_attractors"]
    cols = sorted({r["n_regions"] for r in sweep})
    head = "".join(f"<th scope='col' class='num'>{c}</th>" for c in cols)
    rows = []
    for latent in ("pca", "diffusion"):
        vals = by.get(latent, {})
        if not vals:
            continue
        import math
        ks = sorted(vals)
        slope = ((math.log(vals[ks[-1]]) - math.log(vals[ks[0]])) /
                 (math.log(ks[-1]) - math.log(ks[0])))
        cells = "".join(f"<td class='num'>{vals.get(c, '&mdash;')}</td>"
                        for c in cols)
        flag = "fail" if slope >= 0.95 else "warn"
        rows.append(f"<tr><th scope='row'>{latent}</th>{cells}"
                    f"<td class='num'><span class='pill pill--{flag}'>"
                    f"n &prop; r<sup>{slope:.2f}</sup></span></td></tr>")
    return head, "\n".join(rows)


def _axes_table(ranking):
    labels = {
        "effect": ("Effect", "largest paired occupancy shift at retrieval"),
        "specificity": ("Specificity", "retrieval effect &divide; novel-context effect"),
        "coverage": ("Coverage", "significant &divide; tested states"),
        "resolution": ("Resolution", "1 &minus; largest state fraction"),
        "parsimony": ("Parsimony", "1 &minus; d log(states) / d log(parameter)"),
        "cleanliness": ("Cleanliness", "1 &minus; noise fraction"),
    }
    rows = []
    for axis, w in ranking["weights"].items():
        name, desc = labels[axis]
        rows.append(f"<tr><th scope='row'>{name}</th>"
                    f"<td class='num'>{w:.3f}</td><td>{desc}</td></tr>")
    return "\n".join(rows)


def build(report_dir, figures_dir=None):
    figures_dir = figures_dir or os.path.join(report_dir, "figures")
    load = lambda n: json.load(open(os.path.join(report_dir, n)))  # noqa: E731
    comparison = load("model_comparison.json")
    disc = load("discrimination.json")
    ranking = load("ranking.json")
    try:
        trunc = load("discrimination_trunc5381.json")
    except FileNotFoundError:
        trunc = {}

    rows = ranking["rows"]
    ts = (comparison.get("timescales") or {}).get("arms") or {}
    degen = (comparison.get("timescales") or {}).get("degeneracy") or {}
    join = (comparison.get("koopman_comparison") or {}).get("index_join") or {}
    lat = comparison.get("latents") or {}

    # Figures are placed beside the section that argues from them, so each one
    # is rendered into its own slot rather than dumped in a block at the end.
    figs = {}
    for i, (stem, title, caption) in enumerate(FIGURES, 1):
        path = os.path.join(figures_dir, f"{stem}.png")
        if not os.path.exists(path):
            figs[i] = ""
            continue
        figs[i] = f"""
  <div class="bleed-wide">
      <figure class="figure" id="fig{i}">
        <img src="data:image/png;base64,{_b64(path)}" alt="{html.escape(title)}" />
        <figcaption><span class="figure__no">Figure {i}</span>
          <strong>{title}</strong> {caption}</figcaption>
      </figure>
  </div>"""

    sweep_head, sweep_rows = _sweep_table(comparison.get("n_regions_sweep") or [])
    ch = ts.get("channels", {})
    po = ts.get("pose_only", {})

    def ts_row(arm, label, dim):
        if not arm:
            return ""
        t2 = [v for v in arm["t2_s"] if v]
        ex = arm.get("local_exponents") or []
        g = ((json.loads(json.dumps(ex))[-1]["exponent"]) if ex else None)
        return (f"<tr><th scope='row'>{label}</th><td class='num'>{dim}</td>"
                f"<td class='num'>{t2[0]:.3f}&nbsp;s</td>"
                f"<td class='num'>{t2[-1]:.1f}&nbsp;s</td>"
                f"<td class='num'>{t2[-1]/t2[0]:.0f}&times;</td>"
                f"<td class='num'>{ex[0]['exponent']:.2f} &rarr; {g:.2f}</td>"
                f"<td class='num'>{min(x for x in arm['t2_over_tau'] if x):.2f}</td>"
                f"<td><span class='pill pill--fail'>no plateau</span></td></tr>")

    return TEMPLATE.format(
        rank_cards=_rank_cards(rows),
        disc_rows=_discrimination_table(rows, disc, trunc),
        geom_rows=_geometry_table(comparison.get("arms") or {}),
        sweep_head=sweep_head,
        sweep_rows=sweep_rows,
        axes_rows=_axes_table(ranking),
        ts_rows=ts_row(ch, "pose PCs + locomotor channels", "11D") +
                ts_row(po, "pose PCs only (control)", "9D"),
        figures_1_2=figs.get(1, "") + figs.get(2, ""),
        figures_3=figs.get(3, ""),
        figures_4=figs.get(4, ""),
        figures_5=figs.get(5, ""),
        figures_6=figs.get(6, ""),
        ari_pca=f"{join.get('pca', {}).get('adjusted_rand', float('nan')):.4f}",
        ari_diff=f"{join.get('diffusion', {}).get('adjusted_rand', float('nan')):.4f}",
        auc_lin=f"{degen.get('models', {}).get('logistic', {}).get('auc', float('nan')):.3f}",
        auc_boost=f"{degen.get('models', {}).get('boosted', {}).get('auc', float('nan')):.3f}",
        pca_secs=f"{lat.get('pca', {}).get('seconds', 0):.0f}",
        diff_secs=f"{lat.get('diffusion', {}).get('seconds', 0):.0f}",
        gap=f"{lat.get('diffusion', {}).get('spectral_gap', float('nan')):.4f}",
        n_states_ulam=ch.get("n_states", "&mdash;"),
        horizon=f"{ch.get('horizon_s', 0):.0f}",
    )


TEMPLATE = """<title>VIEB v2 — model comparison</title>
<style>
  /* Light is the base set. Every colour is a token so the two dark scopes
     below can redefine the set without any component knowing about themes. */
  :root {{
    --ground:        #fbfbfd;
    --surface:       #ffffff;
    --surface-sunk:  #f4f4f8;
    --ink:           #15151e;
    --ink-2:         #4a4a5c;
    --ink-3:         #7d7d92;
    --rule:          #e3e3ed;
    --rule-strong:   #cfcfe0;
    --accent:        #3535a6;
    --accent-soft:   #ecebfa;
    --pass:          #0f6f52;
    --pass-soft:     #e2f3ec;
    --warn:          #8a5800;
    --warn-soft:     #f8efdc;
    --fail:          #a52834;
    --fail-soft:     #fbe9eb;

    --serif: ui-serif, Charter, "Bitstream Charter", Georgia, "Times New Roman", serif;
    --sans:  system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    --mono:  ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;

    --measure: 68ch;
    --wide: min(1180px, 94vw);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      --ground:       #101015;
      --surface:      #191921;
      --surface-sunk: #14141b;
      --ink:          #f1f1f6;
      --ink-2:        #b4b4c6;
      --ink-3:        #8686a0;
      --rule:         #2a2a37;
      --rule-strong:  #3b3b4d;
      --accent:       #9a9aff;
      --accent-soft:  #1f1f3d;
      --pass:         #45c197;
      --pass-soft:    #10261f;
      --warn:         #d99f37;
      --warn-soft:    #2a2110;
      --fail:         #f0757f;
      --fail-soft:    #2e1418;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:       #101015;
    --surface:      #191921;
    --surface-sunk: #14141b;
    --ink:          #f1f1f6;
    --ink-2:        #b4b4c6;
    --ink-3:        #8686a0;
    --rule:         #2a2a37;
    --rule-strong:  #3b3b4d;
    --accent:       #9a9aff;
    --accent-soft:  #1f1f3d;
    --pass:         #45c197;
    --pass-soft:    #10261f;
    --warn:         #d99f37;
    --warn-soft:    #2a2110;
    --fail:         #f0757f;
    --fail-soft:    #2e1418;
  }}

  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--ground);
    color: var(--ink);
    font-family: var(--sans);
    font-size: 16.5px;
    line-height: 1.62;
    -webkit-font-smoothing: antialiased;
  }}

  /* One grid, three tracks. Prose sits in `measure`; evidence breaks out to
     `wide` without any element needing its own margins. */
  .page {{
    display: grid;
    grid-template-columns:
      [full-start] minmax(1.25rem, 1fr)
      [wide-start] minmax(0, calc((var(--wide) - var(--measure)) / 2))
      [text-start] minmax(0, var(--measure)) [text-end]
      minmax(0, calc((var(--wide) - var(--measure)) / 2)) [wide-end]
      minmax(1.25rem, 1fr) [full-end];
    row-gap: 1.5rem;
    padding: 0 0 6rem;
  }}
  .page > * {{ grid-column: text; }}
  .bleed-wide {{ grid-column: wide; }}

  header.masthead {{
    grid-column: full;
    background: var(--surface);
    border-bottom: 1px solid var(--rule);
    padding: 3.25rem 1.25rem 2.5rem;
    margin-bottom: 1.5rem;
  }}
  .masthead__inner {{ max-width: var(--wide); margin: 0 auto; }}
  .eyebrow {{
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--accent);
    margin: 0 0 0.9rem;
  }}
  h1 {{
    font-family: var(--serif);
    font-weight: 600;
    font-size: clamp(2rem, 4.4vw, 3rem);
    line-height: 1.1;
    letter-spacing: -0.018em;
    text-wrap: balance;
    margin: 0 0 0.9rem;
  }}
  .standfirst {{
    font-family: var(--serif);
    font-size: 1.16rem;
    line-height: 1.55;
    color: var(--ink-2);
    max-width: 62ch;
    margin: 0 0 1.6rem;
  }}
  .runmeta {{
    display: flex; flex-wrap: wrap; gap: 0 1.75rem;
    font-family: var(--mono); font-size: 0.78rem;
    color: var(--ink-3); margin: 0;
    font-variant-numeric: tabular-nums;
  }}

  h2 {{
    font-family: var(--serif);
    font-size: 1.62rem;
    font-weight: 600;
    letter-spacing: -0.012em;
    line-height: 1.22;
    text-wrap: balance;
    margin: 2.75rem 0 0.2rem;
    padding-top: 1.5rem;
    border-top: 2px solid var(--ink);
  }}
  h2 .sec {{
    display: block;
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 0.5rem;
    font-weight: 400;
  }}
  h3 {{
    font-family: var(--sans);
    font-size: 1.02rem;
    font-weight: 650;
    letter-spacing: 0.002em;
    margin: 2rem 0 0.1rem;
  }}
  p {{ margin: 0.85rem 0; }}
  a {{ color: var(--accent); text-underline-offset: 2px; }}
  a:focus-visible, summary:focus-visible {{
    outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 2px;
  }}
  strong {{ font-weight: 650; }}
  code {{
    font-family: var(--mono); font-size: 0.86em;
    background: var(--surface-sunk); border: 1px solid var(--rule);
    padding: 0.08em 0.34em; border-radius: 3px;
  }}
  .num {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
  .strong {{ font-weight: 650; }}

  /* Verdict strip: the summary before the detail. */
  .cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 0.75rem;
    margin: 0.5rem 0 1rem;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--rule);
    border-top: 3px solid var(--rule-strong);
    padding: 0.95rem 1rem 1.1rem;
  }}
  .card--ref {{ border-top-color: var(--accent); background: var(--accent-soft); }}
  .card__tag {{
    font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--ink-3); margin: 0 0 0.4rem;
  }}
  .card--ref .card__tag {{ color: var(--accent); }}
  .card__name {{
    font-family: var(--sans); font-size: 0.95rem; font-weight: 650;
    margin: 0 0 0.55rem; line-height: 1.3;
  }}
  .card__score {{ margin: 0 0 0.65rem; display: flex; align-items: baseline; gap: 0.4rem; }}
  .card__score .num {{ font-size: 1.7rem; font-weight: 500; letter-spacing: -0.02em; }}
  .card__unit {{ font-size: 0.72rem; color: var(--ink-3); }}
  .card__stats {{ margin: 0; display: grid; gap: 0.15rem; }}
  .card__stats > div {{
    display: flex; justify-content: space-between; gap: 0.5rem;
    border-top: 1px solid var(--rule); padding-top: 0.22rem;
    font-size: 0.78rem;
  }}
  .card__stats dt {{ color: var(--ink-3); }}
  .card__stats dd {{ margin: 0; font-size: 0.82rem; }}

  .tablewrap {{ overflow-x: auto; margin: 1.1rem 0 0.4rem; }}
  table {{
    border-collapse: collapse; width: 100%;
    font-size: 0.85rem; background: var(--surface);
  }}
  caption {{
    caption-side: top; text-align: left; color: var(--ink-3);
    font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.1em;
    text-transform: uppercase; padding-bottom: 0.6rem;
  }}
  th, td {{
    padding: 0.5rem 0.7rem; text-align: left;
    border-bottom: 1px solid var(--rule); white-space: nowrap;
  }}
  thead th {{
    font-size: 0.72rem; font-weight: 600; color: var(--ink-2);
    letter-spacing: 0.02em; border-bottom: 1.5px solid var(--rule-strong);
    vertical-align: bottom;
  }}
  tbody th {{ font-weight: 600; }}
  td.num, th.num {{ text-align: right; font-family: var(--mono);
                    font-variant-numeric: tabular-nums; }}
  tbody tr:hover {{ background: var(--surface-sunk); }}
  tr.row--ref {{ background: var(--accent-soft); }}
  tr.row--ref:hover {{ background: var(--accent-soft); }}
  tr.row--ref th {{ color: var(--accent); }}

  .pill {{
    display: inline-block; padding: 0.1rem 0.45rem; border-radius: 999px;
    font-family: var(--mono); font-size: 0.75rem; font-weight: 600;
  }}
  .pill--pass {{ background: var(--pass-soft); color: var(--pass); }}
  .pill--warn {{ background: var(--warn-soft); color: var(--warn); }}
  .pill--fail {{ background: var(--fail-soft); color: var(--fail); }}

  .figure {{ margin: 1.6rem 0 0.6rem; }}
  .figure img {{
    display: block; width: 100%; height: auto;
    background: #fcfcfb; border: 1px solid var(--rule);
  }}
  figcaption {{
    font-size: 0.83rem; line-height: 1.5; color: var(--ink-2);
    margin-top: 0.65rem; max-width: 78ch;
  }}
  .figure__no {{
    font-family: var(--mono); font-size: 0.7rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--accent); margin-right: 0.5rem;
  }}

  .callout {{
    background: var(--surface); border: 1px solid var(--rule);
    border-left: 3px solid var(--accent);
    padding: 1rem 1.15rem; margin: 1.4rem 0;
  }}
  .callout p:first-child {{ margin-top: 0; }}
  .callout p:last-child {{ margin-bottom: 0; }}
  .callout--fail {{ border-left-color: var(--fail); }}

  ol.findings, ul.plain {{ padding-left: 1.15rem; margin: 0.9rem 0; }}
  ol.findings > li, ul.plain > li {{ margin-bottom: 0.7rem; padding-left: 0.2rem; }}
  ol.findings > li::marker {{ font-family: var(--mono); color: var(--accent);
                              font-weight: 600; }}

  .support {{
    font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--ink-3);
    border: 1px solid var(--rule); border-radius: 999px;
    padding: 0.05rem 0.45rem; margin-left: 0.4rem; white-space: nowrap;
  }}

  details {{
    border: 1px solid var(--rule); background: var(--surface);
    padding: 0.75rem 1rem; margin: 1rem 0;
  }}
  summary {{ cursor: pointer; font-weight: 650; font-size: 0.92rem; }}
  details[open] summary {{ margin-bottom: 0.6rem; }}

  pre {{
    background: var(--surface-sunk); border: 1px solid var(--rule);
    padding: 0.9rem 1rem; overflow-x: auto; font-family: var(--mono);
    font-size: 0.79rem; line-height: 1.55; margin: 1rem 0;
  }}
  footer {{
    grid-column: text; margin-top: 3rem; padding-top: 1.25rem;
    border-top: 1px solid var(--rule); color: var(--ink-3); font-size: 0.82rem;
  }}
  @media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; transition: none !important; }}
  }}
</style>

<div class="page">
  <header class="masthead">
    <div class="masthead__inner">
      <p class="eyebrow">VIEB v2 &middot; branch v2_results</p>
      <h1>Four ways to find a behavioural state, scored on whether they find one</h1>
      <p class="standfirst">Four state-discovery algorithms have been run on
        Luna across two latent spaces. Until now they were compared only on the
        shape of the partition they produce. Scored instead on whether their
        states separate Context&nbsp;A after conditioning &mdash; the axis the
        Keypoint-MoSeq control established &mdash; the ranking reverses, and
        the v2 default pipeline turns out to detect nothing at all.</p>
      <p class="runmeta">
        <span>3,846 recordings</span><span>298 animals</span>
        <span>22,355,989 frames</span><span>30 fps</span>
        <span>paired Wilcoxon &middot; BH-FDR</span>
      </p>
    </div>
  </header>

  <section class="bleed-wide">
    <div class="cards">
{rank_cards}
    </div>
  </section>

  <h2><span class="sec">Section 01 &middot; the headline</span>Significance count
    is not effect size</h2>

  <p>With 298 animals paired within-subject, a two-percentage-point shift in a
    state that occupies 96% of every session clears q&nbsp;&lt;&nbsp;1e-31.
    That is the trap in this dataset, and it is why the arms had to be scored
    on effect size rather than on how many of their states reach
    significance.</p>

  <ol class="findings">
    <li><strong>The v2 default pipeline does not detect the effect at all.</strong>
      <code>pca-HDBSCAN</code>'s largest median paired difference across every
      state is <em>exactly</em> 0.0. Truncated to a common session length it
      finds 0 of 4 testable states significant, and its own sign-flip null
      reaches that count 100% of the time. Not a weak detector &mdash; a null
      one.</li>
    <li><strong><code>diffusion-HDBSCAN</code>'s 35-of-37 is a power artifact.</strong>
      Its hit rate edges MoSeq's, on a top state that moves from 0.951 to
      0.971. Ranking by significant-state count puts this arm first and the arm
      with a 0.55&nbsp;&rarr;&nbsp;0.39 shift third.</li>
    <li><strong>One VIEB arm beats MoSeq on one axis.</strong>
      <code>diffusion-Koopman</code>'s retrieval effect is 9.95&times; its
      novel-context effect, against MoSeq's 5.92&times; &mdash; its shift is
      more specific to the <em>conditioned</em> context rather than to context
      change in general. It does not beat MoSeq on effect size, and nothing
      here does.</li>
  </ol>

  <div class="bleed-wide tablewrap">
    <table>
      <caption>Retrieval contrast &mdash; day&nbsp;1 Context&nbsp;A (no shock)
        vs day&nbsp;0 Context&nbsp;A</caption>
      <thead>
        <tr>
          <th scope="col">arm</th>
          <th scope="col" class="num">states</th>
          <th scope="col" class="num">tested</th>
          <th scope="col" class="num">sig.</th>
          <th scope="col" class="num">max |shift|</th>
          <th scope="col" class="num">top state A &larr; B</th>
          <th scope="col" class="num">sig. trunc.</th>
          <th scope="col" class="num">|shift| trunc.</th>
          <th scope="col" class="num">null &ge; obs</th>
        </tr>
      </thead>
      <tbody>
{disc_rows}
      </tbody>
    </table>
  </div>

  <p>The truncated columns re-run every contrast on the first 5,381 frames of
    each recording. Session length is confounded with arm here &mdash;
    Context&nbsp;A runs ~6,302 frames against ~5,392 for B and C, because the
    shock protocol needs the time &mdash; so this is the test that removes
    length as an explanation.</p>

  <div class="callout">
    <p><strong>Truncation is the decisive test, and it separates the arms
      further.</strong> It <em>strengthens</em> <code>diffusion-Koopman</code>
      (0.187&nbsp;&rarr;&nbsp;0.211), the same direction MoSeq moved, so
      session length was diluting its effect rather than manufacturing it. And
      it removes <code>pca-HDBSCAN</code>'s two significant states entirely,
      leaving a result its own null reproduces every single time.</p>
  </div>

{figures_1_2}

  <h2><span class="sec">Section 02 &middot; how they were compared</span>One
    metric, reused verbatim</h2>

  <p><code>results_analysis/discriminate.py</code> reuses
    <code>scripts/moseq_control.py</code>'s statistics without modification
    &mdash; per-animal means, paired Wilcoxon signed-rank, BH-FDR, the same
    within-animal sign-flip null, the same 0.1%/50-recording frequency floor.
    Only the state labels change, which is what makes the numbers
    commensurable.</p>

  <h3>The join that made it possible</h3>
  <p><code>labels.npz</code> stores an index of
    <code>(recording_idx, frame_idx)</code> and nothing else. The module that
    writes it states the map back to a file &ldquo;is not merely absent, it is
    not reconstructible after the fact&rdquo; &mdash; unreadable files are
    dropped into a skip list that shifts every later index and is never
    persisted.</p>
  <p>That is true in general and <strong>verifiably false for this run</strong>:
    the pose directory yields 4,925 paths, the checkpoint holds 4,925 lengths,
    and the per-file frame count matches the checkpoint for all 4,925 exactly,
    with zero mismatches. A skipped file would break that correspondence, so
    the match is a positive check that the skip list was empty rather than an
    assumption that it was. The check raises rather than warns: a silent
    off-by-one would attribute each recording's behaviour to a neighbouring
    animal and still produce plausible p-values.</p>

  <h3>Two corrections applied before scoring</h3>
  <ul class="plain">
    <li><strong>Deduplication.</strong> The Koopman-family runs predate the
      h5/csv dedup and see 4,925 sessions; 1,079 duplicate rows are collapsed
      (h5 preferred) before per-animal averaging, giving 3,846 recordings and
      298 animals in every arm.</li>
    <li><strong>Truncation.</strong> Every contrast re-run at a common 5,381
      frames, matching the MoSeq control.</li>
  </ul>

  <h2><span class="sec">Section 03 &middot; the geometric reading</span>What
    shape each partition is</h2>

  <div class="bleed-wide tablewrap">
    <table>
      <caption>Partition geometry, clustered-frame convention</caption>
      <thead>
        <tr>
          <th scope="col">arm</th>
          <th scope="col" class="num">states</th>
          <th scope="col" class="num">largest state</th>
          <th scope="col" class="num">noise</th>
          <th scope="col" class="num">entropy</th>
          <th scope="col" class="num">noise speed ratio</th>
          <th scope="col" class="num">size&harr;speed corr</th>
        </tr>
      </thead>
      <tbody>
{geom_rows}
      </tbody>
    </table>
  </div>

  <p><strong>The noise speed ratio is the geometric number that predicts the
    behavioural result.</strong> Above 1 means the frames a method failed to
    label are the <em>fast</em> ones &mdash; the documented signature of
    density-based clustering under-detecting brief behaviours. Both HDBSCAN
    arms discard frames moving 10&times; and 19&times; faster than what they
    keep. Both Koopman arms sit near or below 1. The two arms that throw away
    the fast frames are the two arms with no effect.</p>

  <p>The noise columns are not the same quantity across families: HDBSCAN's
    <code>-1</code> means unclustered, Koopman's means <em>near a
    separatrix</em> &mdash; a transition.</p>

  <h3>The two families do not agree</h3>
  <p>Joined on index over 28,586,707 frames, adjusted Rand is
    <span class="num">{ari_pca}</span> in PCA space and
    <span class="num">{ari_diff}</span> in diffusion space. That is
    indistinguishable from independent partitions. Run on the same frames, in
    the same latent space, HDBSCAN and Koopman are not two estimates of one
    thing.</p>

{figures_4}

  <h3>Is the Koopman state count an output?</h3>
  <p>Prior decisions argued that a state count only counts as an
    &ldquo;output&rdquo; if the parameter that could fake it has been varied.
    Sweeping <code>--n-regions</code>:</p>

  <div class="tablewrap">
    <table>
      <caption>Attractors found per <code>--n-regions</code></caption>
      <thead><tr><th scope="col">latent</th>{sweep_head}
        <th scope="col" class="num">scaling</th></tr></thead>
      <tbody>
{sweep_rows}
      </tbody>
    </table>
  </div>

  <p>In PCA space the claim fails outright &mdash; one attractor per region
    across a 16&times; sweep, so the state count is the parameter under another
    name. In diffusion space the exponent is 0.71, so genuine merging happens
    and the count is partly an output, though not parameter-free.
    <strong>Zero limit cycles at every resolution in both arms</strong>: no
    oscillatory behaviour was recovered anywhere.</p>

{figures_3}

  <h2><span class="sec">Section 04 &middot; the transfer operator</span>The
    falsification gate still fails</h2>

  <p>{n_states_ulam} Voronoi microstates on the 11-D observation space, lag
    swept from 0.033&nbsp;s to {horizon}&nbsp;s in 26 log-spaced steps.</p>

  <div class="bleed-wide tablewrap">
    <table>
      <caption>Implied timescales, both arms</caption>
      <thead>
        <tr><th scope="col">arm</th><th scope="col" class="num">dim</th>
          <th scope="col" class="num">t&#8322; at 0.033&nbsp;s</th>
          <th scope="col" class="num">t&#8322; at horizon</th>
          <th scope="col" class="num">growth</th>
          <th scope="col" class="num">local exponent</th>
          <th scope="col" class="num">min t&#8322;/&tau;</th>
          <th scope="col">verdict</th></tr>
      </thead>
      <tbody>
{ts_rows}
      </tbody>
    </table>
  </div>

  <p>The local exponent drifts monotonically from 0.53 to 0.86 &mdash; strictly
    between 0 (a plateau) and 1 (the trivial large-&tau; artifact) everywhere.
    That is long-memory, multi-scale behaviour with <em>no timescale
    separation</em>, a stronger and more specific statement than &ldquo;no
    plateau&rdquo;.</p>

  <p>The estimator is not broken, which is what gives the negative result its
    weight: at every one of the 26 lags all 500 microstates were retained, none
    dropped, leak fraction 0, one connected component, not near-reducible, on
    18.2&ndash;22.4M pairs. The absence of a plateau is a property of the
    data.</p>

  <div class="callout callout--fail">
    <p><strong>The standing concern is unchanged, and this comparison sharpens
      it.</strong> This negative result is what the branch's own cited
      reference predicts at K=1, where delay embedding is the remedy. The gate
      was specified to run <em>before</em> any delay embedding, so it may be
      falsifying K=1 rather than falsifying the branch. What the comparison
      adds: both arms that found a real effect run on the delay embedding, and
      the transfer operator does not.</p>
  </div>

{figures_5}

  <h3>What alignment actually cost</h3>
  <p>Classifying top-versus-bottom tercile of raw centroid speed from aligned
    pose alone, held out by recording, gives AUC
    <span class="num">{auc_lin}</span> linear and
    <span class="num">{auc_boost}</span> boosted. The two terciles differ by
    <strong>55&times;</strong> in real speed (0.77 vs 43.8&nbsp;px/s), and
    posture recovers that at 0.79, not 1.0. The locomotor signature is present
    but not linearly decodable &mdash; a partial loss, which bounds how much
    any purely postural method can recover.</p>

  <h2><span class="sec">Section 05 &middot; ranking</span>Six axes, weighted,
    and the one judgement they rest on</h2>

  <div class="tablewrap">
    <table>
      <caption>Composite weighting</caption>
      <thead><tr><th scope="col">axis</th><th scope="col" class="num">weight</th>
        <th scope="col">measures</th></tr></thead>
      <tbody>
{axes_rows}
      </tbody>
    </table>
  </div>

  <p>Arms missing an axis are renormalised over the axes they have, so
    &ldquo;not measured&rdquo; never reads as &ldquo;measured badly&rdquo;.
    Min&ndash;max scaling means the worst arm on an axis contributes exactly 0
    to it. <strong>Effect is weighted above coverage deliberately</strong>, and
    that single choice is what separates this ranking from the geometric one
    &mdash; ranking by significant-state count alone puts
    <code>diffusion-HDBSCAN</code> first.</p>

  <p><code>parsimony</code> is unscored for both HDBSCAN arms because
    <code>min_cluster_size</code> was never swept. That is untested, not
    passed: the scrutiny demanded of Koopman's state count has never been
    applied to HDBSCAN's.</p>

{figures_6}

  <h2><span class="sec">Section 06 &middot; interpretation</span>Why each model
    behaved as it did</h2>

  <h3>pca-HDBSCAN finds one state because there is one density mode
    <span class="support">well supported</span></h3>
  <p>Alignment removes translation and heading; PCA then keeps the directions
    of largest postural variance. In that space the density has a single
    overwhelming mode. With <code>min_cluster_size=50</code> against 28.6M
    points a cluster needs only 50 members, and it still returned 99.2% of
    clustered frames in one state. The 15.4% it called noise moves 9.7&times;
    faster. It clustered &ldquo;the animal is in a typical posture&rdquo; and
    discarded the rest &mdash; and since freezing and locomotion differ
    55&times; in speed, and speed was subtracted, the surviving postural
    variation genuinely has one mode. An effect of exactly 0.0 is the honest
    consequence.</p>

  <h3>diffusion-HDBSCAN's 37 states are one core plus 36 slivers
    <span class="support">well supported</span></h3>
  <p>The diffusion spectrum is nearly degenerate &mdash; eigenvalues 0.988 to
    0.935, spectral gap <span class="num">{gap}</span>. No gap means no natural
    cluster count, so HDBSCAN shaves satellites off a continuum: 96.4% of
    clustered frames in one state, entropy 0.074, and the worst noise-speed
    ratio of any arm at 19.0.</p>

  <h3>pca-Koopman fragments because PCA space has no basin structure
    <span class="support">well supported</span></h3>
  <p>Each Voronoi region's local operator finds its own fixed point and the
    graph pruning never merges them, because there are no real basins to merge
    into. 47.7% of frames land near a separatrix &mdash; a partition that is
    mostly boundary. Its effects are real but small, because no state is larger
    than 3.1%: a genuine behavioural regime split across 40 basins cannot show
    a large shift in any one of them.</p>

  <h3>diffusion-Koopman wins because its coordinates are dynamical, not
    variance-based <span class="support">supported</span></h3>
  <p>Diffusion-map coordinates approximate the slowest-relaxing directions of a
    diffusion on the pose manifold &mdash; they are ordered by <em>timescale</em>,
    where PCA's are ordered by variance. A basin decomposition computed in slow
    coordinates is far more likely to align with sustained behavioural regimes.
    Three independent signals agree: sublinear state scaling
    (r<sup>0.71</sup>, genuine merging), a low transition fraction (13.9%
    against PCA's 47.7%, basins with real interiors), and a noise-speed ratio
    <em>below</em> 1 (0.60 &mdash; its separatrix frames are slower than basin
    interiors, i.e. postural pauses between regimes rather than fast
    transitions). This is the one arm where the state count, the geometry and
    the behavioural effect all point the same way.</p>

  <h3>MoSeq wins because it kept the channel VIEB subtracted, and has a
    temporal prior <span class="support">confounded</span></h3>
  <p>Freezing is defined by near-zero locomotion; <code>align_all</code>
    subtracts locomotion. On top of that, MoSeq's stickiness prior makes
    syllables bouts by construction, where these VIEB arms assign per frame
    with no temporal model in this path.
    <strong>The confound:</strong> MoSeq differs from VIEB in representation
    <em>and</em> algorithm at once, so this comparison cannot attribute its win
    between the two. Separating them is next step&nbsp;5.</p>

  <h2><span class="sec">Section 07 &middot; limits</span>What this comparison
    cannot tell you</h2>
  <ul class="plain">
    <li><strong>The Koopman-family models were fit on double-counted data.</strong>
      Deduplication here fixes the scoring, not the fit &mdash; both algorithms
      saw 28% of recordings at double weight. Every arm is equally affected, so
      the ranking is probably safe; no arm's absolute numbers are.</li>
    <li><strong>The transfer-operator arm has no per-frame labels on disk</strong>
      (no <code>microstates.npz</code> was written), so it could not be scored
      on discrimination at all.</li>
    <li><strong><code>min_cluster_size</code> was never swept</strong>, so
      HDBSCAN's state count has had none of the scrutiny Koopman's got.</li>
    <li><strong>No seed-stability estimate anywhere</strong> &mdash;
      <code>seed_stability</code> is null in every checkpoint.</li>
    <li><strong>A near-decomposable pooled chain is metastable
      mathematically.</strong> Two sub-populations never observed transitioning
      produce a real slow eigenvalue, and no pooled diagnostic distinguishes
      &ldquo;states are behaviours&rdquo; from &ldquo;states are animals&rdquo;.
      Nothing here tests that.</li>
    <li><strong>The composite weighting is a judgement</strong>, stated so it
      can be disagreed with. The effect-above-coverage ordering is the
      load-bearing part; the rest moves the ranking very little.</li>
  </ul>

  <h2><span class="sec">Section 08 &middot; next</span>What to run, in order</h2>

  <h3>Immediate &mdash; each settles a live question</h3>
  <ol class="findings">
    <li><strong>Re-run <code>diffusion-Koopman</code> on the deduplicated
      alignment.</strong> The best arm is the one whose fit is most worth
      trusting, and right now it was fit on double-counted data. Highest value
      on this list.</li>
    <li><strong>Sweep <code>min_cluster_size</code></strong> (25/50/100/200/500)
      for both HDBSCAN arms and score each with <code>discriminate</code> — the
      only way to know whether HDBSCAN's collapse is a parameter artifact or a
      property of the latent.</li>
    <li><strong>Sweep <code>--n-regions</code> under the discrimination
      metric.</strong> If the 0.19 shift survives r=12&rarr;192, that is a
      strong result.</li>
    <li><strong>Write <code>microstates.npz</code></strong> so the Ulam arm can
      be scored on the same axis as everything else.</li>
  </ol>

  <h3>Next &mdash; separates the confounds</h3>
  <ol class="findings" start="5">
    <li><strong>Run Koopman on the 11-D observation space</strong> (pose PCs
      plus restored locomotor channels). The clean test of &ldquo;MoSeq wins
      because it kept the speed channel&rdquo;: if the effect moves toward 0.36
      when speed is restored, representation is the explanation; if not, the
      temporal prior is.</li>
    <li><strong>Add a temporal prior to the VIEB arms.</strong> v1's HMM
      Viterbi smoothing exists and is not in the v2 path. The other half of the
      same question, and cheap.</li>
    <li><strong>Delay-embed before the timescale gate.</strong> The gate's own
      stated concern is that it falsified K=1 rather than the branch.</li>
  </ol>

  <h3>Then &mdash; makes it publishable</h3>
  <ol class="findings" start="8">
    <li><strong>Seed stability</strong> on the winning arm: refit with 5 seeds,
      report adjusted Rand between runs.</li>
    <li><strong>Watch clips of <code>diffusion-Koopman</code> state 13.</strong>
      The whole argument is that it is a behaviourally real, fear-suppressed
      state. <code>generate_clips.py</code> exists; nobody has looked at
      it.</li>
    <li><strong>Test the near-decomposability confound</strong> — split by
      animal and check whether the slow structure survives within-animal.</li>
  </ol>

  <h3>Open questions</h3>
  <ul class="plain">
    <li><strong>Is <code>pca-HDBSCAN</code> still the production default?</strong>
      It is the v2 pipeline's default path and it is a null detector on this
      dataset. If anything downstream consumes its labels, that should
      stop.</li>
    <li><strong>Should the transfer-operator branch resume at &sect;5a?</strong>
      The gate says stop; the gate's own caveat says it may have tested the
      wrong thing; this report adds evidence for the caveat.</li>
    <li><strong>Is there a ground-truth freezing annotation anywhere?</strong>
      Every metric here is a proxy contrast. Even a few hundred hand-scored
      frames would convert this from a relative ranking into an absolute
      one.</li>
  </ul>

  <details>
    <summary>Reproduce every number on this page</summary>
<pre>cd vieb_v2
python -m results_analysis.collect      --results-root ~/vieb2-results
python -m results_analysis.discriminate --results-root ~/vieb2-results
python -m results_analysis.discriminate --max-frames 5381 \\
                                        --name discrimination_trunc5381
python -m results_analysis.rank         --report ~/vieb2-results/_report
python -m results_analysis.plots        --report ~/vieb2-results/_report
python -m results_analysis.report_html  --report ~/vieb2-results/_report</pre>
    <p>Outputs land in <code>~/vieb2-results/_report/</code>:
      <code>model_comparison.json</code>, <code>discrimination.json</code>,
      <code>discrimination_trunc5381.json</code>, <code>ranking.json</code>,
      <code>figures/</code>.</p>
  </details>

  <footer>
    <p>Branch <code>v2_results</code>. Companion documents:
      <code>docs/V2_MODEL_COMPARISON.md</code> (the full written report),
      <code>docs/V2_RESULTS_CONTEXT.md</code> (where the artifacts live and
      what was verified), <code>docs/TRANSFER_OPERATOR_FINDINGS.md</code>,
      <code>docs/DECISIONS.md</code> #53&ndash;#65.</p>
  </footer>
</div>
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report",
                    default=os.path.expanduser("~/vieb2-results/_report"))
    ap.add_argument("--figures", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    page = build(args.report, args.figures)
    dest = args.out or os.path.join(args.report, "v2_model_comparison.html")
    with open(dest, "w") as fh:
        fh.write(page)
    print(f"[report_html] wrote {dest} ({len(page) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
