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

Circle size encodes mean total dwell time per trial, not mean duration per
individual visit. The shared power scale (area approximately proportional to
mean dwell to the power 1.7, with a small visible floor) improves separation.
Undirected edge width encodes mean number of transitions between that pair per
trial (both directions combined), not transitions per second. Zero edges are
dotted. Fixed-position node labels give the mean and 95% Student-t CI; exact
edge values appear in a separate strip. Session CIs use valid trials; group CIs
use participant-level means. A single independent unit has no estimable CI (NA).
CIs are not clipped. They are descriptive and not corrected for multiplicity.
Session and group plots have separate shared scales.

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


## All-condition overviews and paired baseline contrasts

Each absolute group overview now has a 2-by-6 all_conditions sheet: choice and
pricing x C1-C5 plus a pooled column. Within each participant, average sessions
within condition, then weight their observed valid C1-C5 conditions equally;
participants are subsequently weighted equally. This pooled descriptive value
can reflect differing observed conditions across participants. The original
2-by-5 sheets are also retained. Important (dominant attribute) and correlated
(aligned pair) are annotated. Separate concrete-attribute overview sheets show
all observed configurations for each task.

The group_baseline_difference directory repeats the abstract and concrete
graphs for paired within-participant changes. For a given task, participant,
condition and representation, first compute their condition mean and their
own C1 (balanced, uncorrelated) mean, then subtract C1 node by node and edge by
edge. Participants without a valid same-task C1 are excluded from that contrast;
never substitute the group baseline. The group mean and 95% Student-t CI are
computed from the paired differences, not from independently aggregated
condition means. The C1 contrast is a self-comparison (exactly zero). The pooled
contrast is the within-participant equal mean of observed C2-C5 conditions
minus C1. Concrete groups use the same participant's actual-attribute C1;
matching is by attribute identity, not screen location. Abstract groups use
XYZ identity.

Circle size and edge width encode magnitude of change; muted blue indicates
increase, rust indicates decrease, and hollow/dotted indicates zero. Signed
node and edge numbers retain the actual units (ms/trial and transitions/trial).
All contrast panels share a common magnitude scale separate from absolute graphs.

Additional outputs:
- group_average/xyz_all_conditions.png and attribute_all_conditions.png
- group_average/attribute_concrete_{choice,pricing}_all_conditions.png
- group_baseline_difference/ (same graph hierarchy and overview names)
- group_baseline_difference_summary.csv: paired mean/CI for each cell,
  matched participant count n_subjects, target eligible count n_eligible_subjects,
  and session/trial counts restricted to matched target sessions.
- participant_baseline_differences.csv: each participant's paired node and edge
  changes before group averaging. No automatic exclusion of endpoint candidates.

Run only group plots:

    python data_preprocessing\03_plot_gaze_graphs.py --skip-trial-plots --skip-session-plots

The updated renderer uses scipy.stats for Student-t intervals (SciPy is already
a dependency of steps 1 and 2).
