"""Create one-row-per-session metadata and stimulus summaries from MATLAB files.

Input:  <repo>/data/raw_use/infos_{choice|price}_{node1}_{node2}_{subject}.mat
Output: <repo>/outputs/data_preprocessing/session_summary.csv

Requires: numpy, scipy. MATLAB MCOS table/GazeData contents are NOT decoded.
The session order is inferred from MAT-file header timestamps, not block IDs.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"^infos_(choice|price)_(\d+)_(\d+)_(.+)\.mat$", re.I)
XYZ = {1: "MRD", 2: "MDR", 3: "RMD", 4: "RDM", 5: "DMR", 6: "DRM"}
POSITION = {1: "MRD", 2: "MDR", 3: "RDM", 4: "RMD", 5: "DMR", 6: "DRM"}
ATTRIBUTE = {"M": "money", "R": "rating", "D": "distance"}
CONDITIONS = {
    (1, 1): (1, "balanced", "uncorrelated"),
    (2, 1): (2, "X_dominant", "uncorrelated"),
    (1, 2): (3, "balanced", "YZ_aligned"),
    (2, 2): (4, "X_dominant", "YZ_aligned"),
    (2, 3): (5, "X_dominant", "XZ_aligned"),
}


def scalar(value, default=""):
    """Convert MATLAB scalar values to ordinary Python CSV-compatible values."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size != 1:
            return default
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    return value if isinstance(value, (int, float, str, bool)) else default


def matlab_header_time(header):
    if isinstance(header, bytes):
        header = header.decode("ascii", errors="replace")
    match = re.search(r"Created on:\s*(.+)$", header)
    if not match:
        return ""
    try:
        return datetime.strptime(match.group(1).strip(), "%a %b %d %H:%M:%S %Y").isoformat(sep=" ")
    except ValueError:
        return ""


def count_gaze_containers(infos):
    """Count trial entries with a stored gazeData field; NOT valid gaze samples."""
    datamat = getattr(infos, "dataMat", None)
    trials = getattr(datamat, "trial", None)
    if trials is None:
        return 0
    trials = np.asarray(trials, dtype=object).reshape(-1)
    return sum(getattr(trial, "gazeData", None) is not None for trial in trials)


def summarize(path, input_dir):
    m = NAME.fullmatch(path.name)
    if m is None:
        raise ValueError(f"Invalid file name: {path.name}")
    task, file_n1, file_n2, file_subject = m.groups()
    data = loadmat(path, variable_names=["infos"], squeeze_me=True, struct_as_record=False)
    infos = data["infos"]
    n1 = int(scalar(getattr(infos, "node1", None), -1))
    n2 = int(scalar(getattr(infos, "node2", None), -1))
    map_code = int(scalar(getattr(infos, "xyz_map", None), -1))
    pos_code = int(scalar(getattr(infos, "cond_pos", None), -1))
    xyz = XYZ.get(map_code, "")
    pos = POSITION.get(pos_code, "")
    cid, importance, correlation = CONDITIONS.get((n1, n2), ("", "unknown", "unknown"))
    issues = []
    subject = str(scalar(getattr(infos, "subj_id", None), file_subject))
    if subject != file_subject:
        issues.append("subject_filename_mismatch")
    if (n1, n2) != (int(file_n1), int(file_n2)):
        issues.append("condition_filename_mismatch")
    if not cid:
        issues.append("unexpected_importance_correlation_combination")
    if not xyz:
        issues.append("unknown_xyz_map")
    if not pos:
        issues.append("unknown_cond_pos")
    row = {
        "subject_id": subject,
        "session_index_inferred": "",  # populated after sorting MAT-file timestamps
        "session_timestamp_local_unspecified": matlab_header_time(data.get("__header__", b"")),
        "source_file": path.relative_to(input_dir).as_posix(),
        "file_size_mb": round(path.stat().st_size / (1024**2), 3),
        "response_type": "choice" if task.lower() == "choice" else "pricing",
        "response_scale": "-1_or_+1" if task.lower() == "choice" else "-100_to_+100",
        "condition_id": cid,
        "node1_importance_code": n1,
        "importance": importance,
        "node2_correlation_code": n2,
        "correlation": correlation,
        "xyz_map": map_code,
        "X_attribute": ATTRIBUTE.get(xyz[0], "") if xyz else "",
        "Y_attribute": ATTRIBUTE.get(xyz[1], "") if xyz else "",
        "Z_attribute": ATTRIBUTE.get(xyz[2], "") if xyz else "",
        "dominant_attribute": ATTRIBUTE.get(xyz[0], "") if xyz and n1 == 2 else "",
        "aligned_attributes": (
            f"{ATTRIBUTE[xyz[1]]}+{ATTRIBUTE[xyz[2]]}" if xyz and n2 == 2
            else f"{ATTRIBUTE[xyz[0]]}+{ATTRIBUTE[xyz[2]]}" if xyz and n2 == 3
            else ""
        ),
        "opposing_attribute": (
            ATTRIBUTE[xyz[0]] if xyz and n2 == 2
            else ATTRIBUTE[xyz[1]] if xyz and n2 == 3
            else ""
        ),
        "cond_pos": pos_code,
        "top_attribute": ATTRIBUTE[pos[0]] if pos else "",
        "bottom_left_attribute": ATTRIBUTE[pos[1]] if pos else "",
        "bottom_right_attribute": ATTRIBUTE[pos[2]] if pos else "",
        "change_w": scalar(getattr(infos, "change_w", None)),
        "w1_ini_money_over_rating": scalar(getattr(infos, "w1_ini", None)),
        "w3_ini_distance_over_rating": scalar(getattr(infos, "w3_ini", None)),
        "min_w": scalar(getattr(infos, "min_w", None)),
        "adv_eye_code": scalar(getattr(infos, "adv_eye", None)),
        "selected_eye": {1: "right", 2: "left"}.get(scalar(getattr(infos, "adv_eye", None)), ""),
        "screen_center_x_px": scalar(getattr(infos, "xCenter", None)),
        "screen_center_y_px": scalar(getattr(infos, "yCenter", None)),
        "has_matlab_behavior_table": int(getattr(infos, "table", None) is not None),
        "behavior_values_decoded": 0,
        "n_gaze_trial_containers": count_gaze_containers(infos),
        "gaze_samples_decoded": 0,
        "validation_warnings": "",
    }
    raw_d = getattr(infos, "D", None)
    raw_ab = getattr(infos, "AB", None)
    d = np.asarray(raw_d, dtype=float) if raw_d is not None else np.empty((0, 3))
    ab = np.asarray(raw_ab, dtype=float) if raw_ab is not None else np.empty((0, 6))
    row["n_trials_D"] = d.shape[0] if d.ndim == 2 else 0
    row["n_trials_AB"] = ab.shape[0] if ab.ndim == 2 else 0
    if d.ndim != 2 or d.shape[1] != 3:
        issues.append("D_not_n_by_3")
    if ab.ndim != 2 or ab.shape[1] != 6:
        issues.append("AB_not_n_by_6")
    if row["n_trials_D"] != row["n_trials_AB"]:
        issues.append("D_AB_trial_count_mismatch")
    metrics = (
        "n_finite_D_trials", "p_XY_same_sign", "p_XZ_same_sign", "p_YZ_same_sign",
        "p_all_three_same_sign", "mean_abs_X", "mean_abs_Y", "mean_abs_Z",
        "mean_X", "mean_Y", "mean_Z", "mean_abs_X_minus_max_YZ",
        "p_X_largest_abs", "p_balanced_within_0p2", "p_abs_sum_D_le_0p5",
        "mean_abs_sum_D", "max_abs_sum_D",
    )
    row.update({name: "" for name in metrics})
    if d.ndim == 2 and d.shape[1] == 3 and len(d):
        good = d[np.isfinite(d).all(axis=1)]
        row["n_finite_D_trials"] = len(good)
        if len(good) < len(d):
            issues.append("nonfinite_D_trials")
        if len(good):
            x, y, z = good.T
            ax, ay, az = np.abs(good).T
            sum_d = good.sum(axis=1)
            row.update({
                "p_XY_same_sign": float(np.mean(x * y > 0)),
                "p_XZ_same_sign": float(np.mean(x * z > 0)),
                "p_YZ_same_sign": float(np.mean(y * z > 0)),
                "p_all_three_same_sign": float(np.mean(((x > 0) & (y > 0) & (z > 0)) | ((x < 0) & (y < 0) & (z < 0)))),
                "mean_abs_X": float(ax.mean()),
                "mean_abs_Y": float(ay.mean()),
                "mean_abs_Z": float(az.mean()),
                "mean_X": float(x.mean()),
                "mean_Y": float(y.mean()),
                "mean_Z": float(z.mean()),
                "mean_abs_X_minus_max_YZ": float((ax - np.maximum(ay, az)).mean()),
                "p_X_largest_abs": float(np.mean((ax > ay) & (ax > az))),
                "p_balanced_within_0p2": float(np.mean((np.max(np.abs(good), axis=1) - np.min(np.abs(good), axis=1)) < 0.2)),
                "p_abs_sum_D_le_0p5": float(np.mean(np.abs(sum_d) <= 0.5 + 1e-9)),
                "mean_abs_sum_D": float(np.abs(sum_d).mean()),
                "max_abs_sum_D": float(np.abs(sum_d).max()),
            })
    row["validation_warnings"] = ";".join(issues)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw_use")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "data_preprocessing" / "session_summary.csv")
    args = parser.parse_args()
    inp = args.input.resolve()
    out = args.output.resolve()
    if not inp.is_dir():
        parser.error(f"Input directory does not exist: {inp}")
    # All generated files must remain under the repository's outputs/ directory.
    if not out.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Output must be inside the repository's outputs/ directory")
    paths = sorted(p for p in inp.rglob("infos_*.mat") if p.is_file())
    if not paths:
        parser.error(f"No infos_*.mat files found under {inp}")
    rows = []
    for path in paths:
        try:
            rows.append(summarize(path, inp))
        except Exception as exc:
            print(f"[ERROR] {path.name}: {exc}")
            raise SystemExit(1) from exc
    rows.sort(key=lambda r: (r["subject_id"], r["session_timestamp_local_unspecified"] or "9999", r["source_file"]))
    counts = Counter()
    for row in rows:
        counts[row["subject_id"]] += 1
        row["session_index_inferred"] = counts[row["subject_id"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} sessions from {len(counts)} participants -> {out}")
    print("MATLAB MCOS table and GazeData values were not decoded; gaze counts represent containers only.")
    for row in rows:
        if row["validation_warnings"]:
            print(f"[WARNING] {row['source_file']}: {row['validation_warnings']}")


if __name__ == "__main__":
    main()
