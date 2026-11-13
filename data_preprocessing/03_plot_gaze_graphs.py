"""Plot merged AOI visit transitions, session averages and group averages.

Input: outputs/data_preprocessing/trial_summary.csv (step 2).
Output: outputs/visualize_gaze_graphs/. No raw data is modified.
Visits are AOI episodes, not velocity-defined fixations. See STEP3_GRAPHS.md.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations, permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATTR = {"M": "money", "R": "rating", "D": "distance"}
ATTRS = ("money", "rating", "distance")
AXES = ("X", "Y", "Z")
CONDS = {(1, 1): 1, (2, 1): 2, (1, 2): 3, (2, 2): 4, (2, 3): 5}
COND_LABEL = {
    1: "Balanced / uncorrelated",
    2: "X dominant / uncorrelated",
    3: "Balanced / Y-Z aligned",
    4: "X dominant / Y-Z aligned",
    5: "X dominant / X-Z aligned",
}
COLORS = {k: v for k, v in zip(AXES + ("P",), ("#4477AA", "#EE7733", "#228833", "#AA3377"))}
COLORS.update(dict(zip(ATTRS + ("pricing",), COLORS.values())))
VERTICES = ((0., 1.), (-1., -.65), (1., -.65), (0., -.10))


def js(x):
    return json.dumps(x, separators=(",", ":"), allow_nan=False)


def read_array(x):
    if x is None or x == "":
        return []
    v = json.loads(x)
    if not isinstance(v, list):
        raise ValueError("Expected JSON array")
    return v


def safe(s):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(s)).strip("_") or "unknown"


def subtype(n1, n2, mapping):
    """Canonical real-attribute condition; Y-Z pair order is irrelevant."""
    x, y, z = (ATTR[c] for c in mapping)
    if (n1, n2) not in CONDS:
        return "outside_design"
    dominant = x if n1 == 2 else "balanced"
    pair = sorted((y, z) if n2 == 2 else (x, z)) if n2 != 1 else []
    return dominant + "__" + ("_".join(pair) + "_aligned" if pair else "uncorrelated")


def nodes(task, mode):
    base = AXES if mode == "xyz" else ATTRS
    return base + (("P",) if mode == "xyz" else ("pricing",)) if task == "pricing" else base


def sequence(row):
    """Filter to task-relevant regions, then collapse adjacent equal regions.

    Step-2 code 4 is the response arc; code 0 is generic screen center.
    Use output code 0 for the pricing response arc (not generic screen center).
    Removed intervals do not add to any retained visit duration.
    """
    codes = read_array(row.arr2_pos)
    dur = read_array(row.arr2_duration_ms)
    if len(codes) != len(dur):
        raise ValueError("Unequal position/duration arrays")
    mapping, pos = row.xyz_to_attribute, row.pos_to_attribute
    if sorted(mapping) != sorted("MRD") or sorted(pos) != sorted("MRD"):
        raise ValueError("Invalid attribute permutation")
    seq, ms = [], []
    for code, d in zip(codes, dur):
        code = int(code)
        if d is None or not np.isfinite(float(d)) or float(d) < 0:
            raise ValueError("Invalid visit duration")
        if code in (1, 2, 3):
            target = code
        elif code == 4 and row.response_type == "pricing":
            target = 0
        else:
            continue
        if seq and target == seq[-1]:
            ms[-1] += float(d)
        else:
            seq.append(target)
            ms.append(float(d))
    attributes = [ATTR[pos[i-1]] if i else "pricing" for i in seq]
    xyz = [AXES[mapping.index(pos[i-1])] if i else "P" for i in seq]
    return seq, ms, attributes, xyz


def measures(seq, ms, task, mode):
    ns = nodes(task, mode)
    time = dict.fromkeys(ns, 0.)
    edge = {"|".join(sorted(pair)): 0. for pair in combinations(ns, 2)}
    for n, d in zip(seq, ms):
        time[n] += d
    for a, b in zip(seq, seq[1:]):
        if a != b:
            edge["|".join(sorted((a, b)))] += 1.
    return time, edge


def mean_dict(ds):
    if not ds:
        return {}
    keys = sorted(set().union(*(d.keys() for d in ds)))
    return {k: float(np.mean([d.get(k, 0.) for d in ds])) for k in keys}


def avg_metrics(items):
    return mean_dict([a for a, b in items]), mean_dict([b for a, b in items])


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save_fig(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def positions(task, mode):
    return dict(zip(nodes(task, mode), VERTICES))


def clean_axis(ax):
    ax.set(xlim=(-1.48, 1.48), ylim=(-1.13, 1.40))
    ax.set_aspect("equal")
    ax.axis("off")


def draw_average(ax, task, mode, times, edges, title, time_max, edge_max):
    pos = positions(task, mode)
    for pair, value in edges.items():
        a, b = pair.split("|")
        if a not in pos or b not in pos:
            continue
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        if value > 0:
            ax.plot([x1, x2], [y1, y2], lw=7*value/max(edge_max, 1e-12),
                    color="#8B939F", alpha=.8, solid_capstyle="round", zorder=1)
        else:
            ax.plot([x1, x2], [y1, y2], lw=.45, ls=":", color="#CBD0D5", zorder=1)
        ax.text((x1+x2)/2, (y1+y2)/2, f"{value:.2f}", fontsize=8,
                ha="center", va="center", zorder=4,
                bbox=dict(fc="white", ec="none", pad=1.3, alpha=.93))
    for n, (x, y) in pos.items():
        value = times.get(n, 0.)
        # Scatter marker area is proportional to duration (zero gets a hollow dot).
        ax.scatter(x, y, s=max(25, 2600*value/max(time_max, 1e-12)),
                   facecolor=COLORS[n] if value > 0 else "white",
                   edgecolor=COLORS[n], lw=1.5, zorder=2)
        ax.annotate(n + f"\n{value:.0f} ms", (x, y), xytext=(0, 28 if y >= 0 else -29),
                    textcoords="offset points", ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=9, weight="semibold",
                    zorder=5)
    ax.set_title(title, fontsize=10, pad=12)
    clean_axis(ax)


def draw_trial(row, seq, ms, attrs, xyz, path, dpi):
    """One dot per merged visit; triangle reflects actual display positions."""
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    pos = dict(zip((1, 2, 3, 0), VERTICES))
    if row.response_type != "pricing":
        pos.pop(0)
    for n, (x, y) in pos.items():
        label = (ATTR[row.pos_to_attribute[n-1]] + " / " +
                 AXES[row.xyz_to_attribute.index(row.pos_to_attribute[n-1])]) if n else "pricing arc"
        ax.scatter(x, y, s=1100, facecolor="none", edgecolor="#B4BCC6", lw=1)
        ax.text(x, y+.31, label, fontsize=10, ha="center", weight="semibold")
    indices = defaultdict(list)
    for i, n in enumerate(seq):
        indices[n].append(i)
    xy = [None]*len(seq)
    for n, ids in indices.items():
        x, y = pos[n]
        for j, i in enumerate(ids):
            angle = j*2.399963
            r = .24*math.sqrt((j+.5)/len(ids))
            xy[i] = (x+r*math.cos(angle), y+r*math.sin(angle))
    for i in range(len(seq)-1):
        a, b = xy[i:i+2]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                     connectionstyle="arc3,rad=0.10", shrinkA=6, shrinkB=7,
                     color="#657080", lw=.9, alpha=.72, zorder=1))
    for i, ((x, y), attr, d) in enumerate(zip(xy, attrs, ms)):
        ax.scatter(x, y, s=40+190*d/max(max(ms, default=1), 1),
                   c=COLORS[attr], ec="white", lw=.7, zorder=3)
        ax.text(x, y, str(i+1), ha="center", va="center", color="white", fontsize=7, zorder=4)
    ax.set_title(f"Subject {row.subject_id} | session {int(row.session_order):02d} | "
                 f"trial {int(row.trial):02d} | {row.response_type}", fontsize=12)
    clean_axis(ax)
    if not seq:
        ax.text(0, 0, "No retained gaze visits", ha="center", va="center", color="#666666")
    # A time-ordered strip remains readable when many arrows overlap.
    if seq:
        total = sum(ms)
        text = "  ".join(f"{i+1}:{n}/{d:.0f}" for i, (n, d) in enumerate(zip(xyz, ms)))
        import textwrap
        lines = textwrap.wrap(text, width=95)
        fig.subplots_adjust(bottom=min(.43, .14+.024*len(lines)))
        fig.text(.5, .035, "Visit order (XYZ / ms) | retained time " + f"{total:.0f} ms\n" +
                 "\n".join(lines), ha="center", va="bottom", fontsize=8)
    save_fig(fig, path, dpi)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trial-summary", type=Path, default=ROOT/"outputs"/"data_preprocessing"/"trial_summary.csv")
    p.add_argument("--output-dir", type=Path, default=ROOT/"outputs"/"visualize_gaze_graphs")
    p.add_argument("--subject", type=str)
    p.add_argument("--max-sessions", type=int, default=0)
    p.add_argument("--skip-trial-plots", action="store_true")
    p.add_argument("--skip-session-plots", action="store_true")
    p.add_argument("--skip-group-plots", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Regenerate existing individual trial PNGs")
    p.add_argument("--dpi", type=int, default=130)
    args = p.parse_args()
    out = args.output_dir.resolve()
    if not out.is_relative_to((ROOT/"outputs").resolve()):
        p.error("Output must be under repository outputs/")
    df = pd.read_csv(args.trial_summary, dtype={"subject_id": str}, keep_default_na=False)
    required = {"subject_id","session_order","trial","response_type","node1","node2",
                "xyz_to_attribute","pos_to_attribute","arr2_pos","arr2_duration_ms"}
    missing = required-set(df)
    if missing:
        p.error("Missing trial_summary columns: " + ", ".join(sorted(missing)))
    if args.subject:
        df = df[df.subject_id == args.subject]
    df = df.sort_values(["subject_id","session_order","trial"])
    if args.max_sessions:
        keep = df[["subject_id","session_order"]].drop_duplicates().head(args.max_sessions)
        df = df.merge(keep, on=["subject_id","session_order"], how="inner", validate="many_to_one")
    if df.empty:
        p.error("No matching trials")
    if df.duplicated(["subject_id","session_order","trial"]).any():
        p.error("Duplicate subject/session/trial identifiers")
    trial_rows, sessions, session_metrics = [], [], {}
    for (subject, order), sdf in df.groupby(["subject_id","session_order"], sort=False):
        first = sdf.iloc[0]
        task, n1, n2 = first.response_type, int(first.node1), int(first.node2)
        if task not in ("choice","pricing"):
            raise ValueError(f"Unexpected response_type: {task}")
        for col in ("response_type","node1","node2","xyz_to_attribute","pos_to_attribute"):
            if sdf[col].nunique() != 1:
                raise ValueError(f"Inconsistent {col} within subject={subject}, session={order}")
        cid = CONDS.get((n1,n2), 0)
        sub = subtype(n1,n2,first.xyz_to_attribute)
        metrics = {"xyz": [], "attribute": []}
        base = out/"subjects"/f"subject_{safe(subject)}"/f"session_{int(order):02d}_{task}"
        n_valid = 0
        for row in sdf.itertuples(index=False):
            seq, ms, attrs, xyz = sequence(row)
            valid = bool(seq and sum(ms)>0)
            n_valid += int(valid)
            trial_rows.append(dict(subject_id=subject, session_order=order, trial=row.trial,
                response_type=task, condition_id=cid or "", node1=n1, node2=n2,
                attribute_condition=sub, position_seq_json=js(seq), duration_ms_json=js(ms),
                xyz_seq_json=js(xyz), attribute_seq_json=js(attrs),
                n_visits=len(seq), valid_retained_gaze=int(valid)))
            # Missing/empty gaze is not zero dwell time. Keep row, exclude from averages.
            if valid:
                metrics["xyz"].append(measures(xyz,ms,task,"xyz"))
                metrics["attribute"].append(measures(attrs,ms,task,"attribute"))
            path = base/"trials"/f"trial_{int(row.trial):02d}.png"
            if not args.skip_trial_plots and (args.overwrite or not path.exists()):
                draw_trial(row,seq,ms,attrs,xyz,path,args.dpi)
        rec = dict(subject_id=subject, session_order=order, response_type=task,
                   condition_id=cid, attribute_condition=sub, node1=n1,node2=n2,
                   xyz_to_attribute=first.xyz_to_attribute, n_trials=len(sdf),
                   n_valid_trials=n_valid, n_empty_or_zero_trials=len(sdf)-n_valid)
        for mode in ("xyz","attribute"):
            t,e = avg_metrics(metrics[mode])
            session_metrics[(subject,order,mode)] = (t,e)
            sessions.append(dict(**rec, mode=mode, mean_time_ms_json=js(t),mean_edge_count_json=js(e)))
        print(f"{subject} session {order} ({task}): {n_valid}/{len(sdf)} valid trials", flush=True)
    save_csv(trial_rows,out/"trial_sequences.csv")
    save_csv(sessions,out/"session_graph_summary.csv")

    # Average valid trials within sessions, sessions within each participant/condition,
    # then participants equally. Invalid design sessions remain in individual plots/CSV.
    records = pd.DataFrame(sessions)
    good = records[(records.condition_id>0)&(records.n_valid_trials>0)]
    groups = []
    for mode in ("xyz","attribute"):
        subset = good[good["mode"]==mode]
        families = [("abstract", "condition_id")]
        if mode == "attribute":
            families.append(("concrete","attribute_condition"))
        for family,key in families:
            for (task,c),g in subset.groupby(["response_type",key],sort=True):
                participant_means = []
                for _,sg in g.groupby("subject_id",sort=False):
                    participant_means.append(avg_metrics([
                        session_metrics[(r.subject_id,r.session_order,mode)]
                        for r in sg.itertuples(index=False)
                    ]))
                t,e = avg_metrics(participant_means)
                label = COND_LABEL[int(c)] if family=="abstract" else str(c).replace("__"," / ").replace("_"," ")
                groups.append(dict(mode=mode,family=family,response_type=task,condition=c,
                    condition_label=label,n_subjects=g.subject_id.nunique(),n_sessions=len(g),
                    n_valid_trials=int(g.n_valid_trials.sum()),mean_time_ms_json=js(t),
                    mean_edge_count_json=js(e),
                    attribute_subtype_counts=js(dict(Counter(g.attribute_condition)))))
    save_csv(groups,out/"group_graph_summary.csv")

    # Common absolute visual scaling across all average graphs, not per-panel normalization.
    all_t = [v for rec in sessions+groups for v in json.loads(rec["mean_time_ms_json"]).values()]
    all_e = [v for rec in sessions+groups for v in json.loads(rec["mean_edge_count_json"]).values()]
    # Separate level scales keep group comparisons readable despite long individual sessions.
    st, se = max([1.]+all_t), max([1.]+all_e)
    gt = max([1.]+[v for rec in groups for v in json.loads(rec["mean_time_ms_json"]).values()])
    ge = max([1.]+[v for rec in groups for v in json.loads(rec["mean_edge_count_json"]).values()])
    if not args.skip_session_plots:
        for (subject,order),g in records.groupby(["subject_id","session_order"],sort=False):
            r=g.iloc[0]; task=r.response_type
            fig,axes=plt.subplots(1,2,figsize=(10,5.0))
            for ax,mode in zip(axes,("xyz","attribute")):
                t,e=session_metrics[(subject,order,mode)]
                if t:
                    draw_average(ax,task,mode,t,e,mode.upper(),st,se)
                else:
                    clean_axis(ax);ax.text(0,0,"No valid gaze trials",ha="center")
            fig.suptitle(f"Subject {subject} | session {int(order):02d} | {task} | "
                         f"C{r.condition_id if r.condition_id else 'NA'} | "
                         f"{r.n_valid_trials}/{r.n_trials} trials",fontsize=11)
            fig.text(.5,.015,"Circle area: mean total dwell / trial (ms) | edge width: mean transitions / trial",ha="center",fontsize=8)
            save_fig(fig,out/"subjects"/f"subject_{safe(subject)}"/f"session_{int(order):02d}_{task}"/"average.png",args.dpi)
    if not args.skip_group_plots:
        for r in groups:
            fig,ax=plt.subplots(figsize=(6,5.3))
            title=f"{r['response_type']} | {r['condition_label']}\n"
            title+=f"{r['n_subjects']} participants / {r['n_sessions']} sessions"
            draw_average(ax,r["response_type"],r["mode"],json.loads(r["mean_time_ms_json"]),
                         json.loads(r["mean_edge_count_json"]),title,gt,ge)
            fig.text(.5,.01,"Dwell ms / trial | transitions / trial | equal participant weight",ha="center",fontsize=8)
            name=(f"C{r['condition']}" if r["family"]=="abstract" else safe(r["condition"]))
            save_fig(fig,out/"group_average"/r["mode"]/r["family"]/r["response_type"]/f"{name}.png",args.dpi)
        # Two comparison sheets, each 2 tasks x 5 abstract conditions.
        lookup={(r["mode"],r["response_type"],r["condition"]):r for r in groups if r["family"]=="abstract"}
        for mode in ("xyz","attribute"):
            fig,axes=plt.subplots(2,5,figsize=(22,9.5))
            for i,task in enumerate(("choice","pricing")):
                for j,c in enumerate(range(1,6)):
                    ax=axes[i,j]; r=lookup.get((mode,task,c))
                    if r:
                        draw_average(ax,task,mode,json.loads(r["mean_time_ms_json"]),
                                     json.loads(r["mean_edge_count_json"]),
                                     f"{task} C{c}\nN={r['n_subjects']}",gt,ge)
                    else:
                        clean_axis(ax);ax.set_title(f"{task} C{c}\nNo valid data")
            fig.suptitle(f"Group gaze graphs | {mode.upper()} | shared scale | equal participant weight",fontsize=15)
            save_fig(fig,out/"group_average"/f"{mode}_10_conditions.png",args.dpi)
    print(f"Done: {len(trial_rows)} trials; {len(records)//2} sessions; {len(groups)} group cells -> {out}")

if __name__ == "__main__":
    main()
