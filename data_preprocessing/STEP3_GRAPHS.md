# Gaze transition graphs (after trial preprocessing)

Run from D:\Projects\restaurant-dm:

    git pull
    python data_preprocessing\03_plot_gaze_graphs.py

Input: outputs/data_preprocessing/trial_summary.csv from step 2. No MATLAB
re-export is necessary. Requires numpy, pandas, matplotlib (already used in step 2).

Quick test, with a separate output directory to avoid replacing full summary CSVs:

    python data_preprocessing\03_plot_gaze_graphs.py --subject 102 --max-sessions 1 --output-dir outputs/visualize_gaze_graphs_test

Only session and group graphs (no individual-trial PNGs):

    python data_preprocessing\03_plot_gaze_graphs.py --skip-trial-plots

Existing trial PNGs are reused unless --overwrite is passed. Summary CSVs and
average graphs are regenerated on each run; use separate output directories for
filtered runs. All output directories must be under outputs/.

## Visit definition and coding

Read arr2_pos and arr2_duration_ms from step 2. Retain attribute AOIs 1, 2, 3,
plus response arc AOI 4 in pricing. In this new sequence only, response arc is
coded 0. The original step-2 generic center AOI 0 and other AOI -1 are omitted;
choice has no central node. Filter then merge adjacent equal retained regions,
summing their durations but not excluded interval durations. For example,
[1, 2, 3, 4, 1] in a pricing trial becomes [1, 2, 3, 0, 1].
Physical positions, actual attributes and XYZ identities refer to the same
visits and share the same duration vector. A visit is an AOI episode, not a
velocity-defined physiological fixation.

Each trial PNG uses actual screen positions, labels them with both attribute
and XYZ, and draws one numbered dot per visit and directed arrows in visit
order. Dot size encodes duration relative to that trial. A numbered time-order
strip includes durations in ms.

## Averages

Circle area encodes mean total dwell time per trial, not mean duration per
individual visit. Undirected edge width encodes mean number of transitions
between that pair per trial (both directions combined), not transitions per
second. Zero edges are dotted. Session plots use a shared session scale; group
plots use a shared group scale so panels are comparable. Values are printed.

Averages omit trials with no retained visits or zero retained duration; the
CSV records their counts. The group average is hierarchical: average valid
trials within session, sessions within participant and condition, then average
participants equally. Invalid node1/node2 pairs retain trial/session outputs
but are excluded from the five-condition group comparison.

## Condition grouping

XYZ: five abstract design conditions for each task.
Real attributes (money, rating, distance): retain the same five abstract
conditions for a directly comparable ten-panel overview, plus a second grouping
by actual dominant attribute and actual aligned pair. Pairs are canonicalized
so rating+distance and distance+rating are the same condition. The real
attribute grouping has at most 16 distinct valid configurations per task:
1 balanced/uncorrelated, 3 dominant/uncorrelated, 3 balanced/aligned,
3 dominant versus the other two aligned, 6 dominant aligned with one other
attribute. Unobserved concrete configurations do not get a plot. The abstract
comparison sheets show missing cells explicitly.

## Output

    outputs/visualize_gaze_graphs/
        trial_sequences.csv
        session_graph_summary.csv
        group_graph_summary.csv
        subjects/subject_ID/session_01_pricing/
            average.png
            trials/trial_01.png
        group_average/
            xyz_10_conditions.png
            attribute_10_conditions.png
            xyz/abstract/choice/C1.png
            xyz/abstract/pricing/C1.png
            attribute/abstract/choice/C1.png
            attribute/concrete/choice/<condition>.png

Session average PNGs show XYZ and actual attribute panels side by side.
Group summary contains mode, family (abstract or concrete), condition,
participant/session/trial counts, mean dwell and transitions, and subtype counts.
Pricing endpoint flags from step 2 do not automatically exclude trials.
