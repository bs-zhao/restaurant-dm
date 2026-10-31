# Step 2: trial gaze summary, plots and pricing endpoint QC

Input: data/raw_use. All generated files go to outputs/data_preprocessing only.
The ordering of sessions comes from step 1's session_order.

## Windows CMD commands (D:\Projects\restaurant-dm)

    git pull
    python -m pip install numpy scipy pandas matplotlib
    python data_preprocessing\01_session_summary.py
    matlab -batch "addpath('D:/Projects/restaurant-dm/data_preprocessing'); export_matlab_sessions"
    python data_preprocessing\02_trial_summary.py

MATLAB is required to decode infos.table and the Tobii GazeData MCOS objects.
Add the Tobii MATLAB SDK to your MATLAB path if its classes are not loading.

Test one session:

    python data_preprocessing\02_trial_summary.py --subject 102 --max-sessions 1

Generate detailed PNGs for every trial (in addition to 40-trial overviews):

    python data_preprocessing\02_trial_summary.py --details all

By default, detailed PNGs are created only for pricing-endpoint candidates.
Options: --details flagged/all/none, --no-overviews, --plot-columns 8.

## Output

- outputs/data_preprocessing/matlab_exports/infos_*_behavior.csv: decoded trial behavior.
- outputs/data_preprocessing/matlab_exports/infos_*_gaze.csv: Tobii samples, both eyes,
  selected eye, original microsecond timestamp, fallback-eye indicator.
- outputs/data_preprocessing/trial_summary.csv: one row per trial, original behavior,
  conditions, stimuli D_X/D_Y/D_Z, AOI sequence vectors in JSON, durations, start/end
  times, physical attribute and XYZ labels, transitions, summaries and QC.
- outputs/data_preprocessing/pricing_endpoint_qc.csv: terminal arc endpoint times,
  duration, side, sample count, visits, candidate flag, raw timestamps, response.
- outputs/data_preprocessing/session_endpoint_qc.csv: one row for every processed
  session (including choice sessions), with n_endpoint_detection_triggered
  counting trials that ENTERED the >90 pricing endpoint screen,
  n_endpoint_candidates counting trials flagged by the gaze detector, the
  trial indices of each group in JSON arrays, and counts of reasons for
  skipping detection. It also reports n_triggered_with_no_gaze.
  This is separate from the step-1 session_summary.csv and is regenerated
  whenever step 2 runs. For --subject/--max-sessions, only processed sessions
  are included.
- outputs/data_preprocessing/gaze_qc/subject_<ID>/session_<ORDER>_<TASK>/overview.png:
  one 40-trial overview per session.
- outputs/data_preprocessing/gaze_qc/subject_<ID>/session_<ORDER>_<TASK>/trials/trial_<N>.png:
  detailed plots for flagged or all trials.
- Any previously generated flat PNGs at the gaze_qc root are preserved under
  the appropriate session's legacy/ folder when step 2 starts.

## AOIs and timing

AOI codes: -1 other/invalid, 0 center, 1 top attribute, 2 lower-left attribute,
3 lower-right attribute, 4 pricing response arc. cond_pos maps to
Money/Rating/Distance; xyz_map maps those attributes to X/Y/Z.
The response arc takes priority over attribute AOIs in pricing.
Default AOIs match the actual stimulus rectangles with 35-px padding.
Use --aoi-mode legacy to reproduce the original s1_extract positional bands
apart from separate arc classification.

arr2 has all AOI visits; pure1 removes -1; pure2 retains only attribute
codes 1/2/3. After filtering, neighboring same-attribute visits are merged
with duration summed; elapsed end-start may include excluded gaps.

These visits are consecutive AOI dwell episodes, NOT formal velocity/dispersion
fixations. Duration is derived from the Tobii timestamps and capped sample gaps
(75ms default). Millisecond onset and offset are relative to the FIRST Tobii
sample, NOT to Psychtoolbox GetSecs because their clock synchronization is unknown.

## Pricing endpoint anomaly

PTB_v4/main_price.m defines the response arc centered at xCenter,yCenter+90px,
with click annulus strictly 100<r<200px. Preference +/-100 lies on radial
lines at +/-72 degrees from the vertical. Gaze within 60px of either endpoint
ray is marked near-endpoint, even slightly outside the clickable annulus.

The detector runs ONLY when the trial is pricing AND the absolute reported
preference, normalized to a 100-point maximum, is STRICTLY GREATER than 90.
The original PTB_v4/main_price.m stores signed preference -100..+100, so its
default normalization does not change numeric values. At 90 or less, or
when preference is missing, endpoint screening is skipped. Raw gaze and
attribute AOI sequences are still extracted normally.

New QC columns: price_response_100, price_abs_100,
endpoint_detection_eligible, endpoint_detection_status. The candidate flag
is 0 for skipped trials; endpoint_pattern is not_screened.
Optional switches: --endpoint-min-abs-price-100 90 and --price-scale-max 100.

Within the final 1200ms of raw gaze (only eligible trials), identify
near-endpoint visits. A candidate requires >=5 gaze samples and >=80ms per qualifying visit, its last
qualifying visit ending within 250ms of the last gaze sample, and either one
>=300ms sustained visit or two qualifying visits near the same endpoint.
All visits, with raw and relative start/end times and side, are preserved
as endpoint_visits_json regardless of flag status.

Thresholds are configurable, for example:

    python data_preprocessing\02_trial_summary.py --endpoint-window-ms 1500 --endpoint-radius-px 75 --endpoint-sustained-ms 400

WARNING: PTB_v4 does NOT record unsuccessful mouse clicks. Prolonged or
repeated terminal gaze near +/-100 can motivate manual inspection but cannot
prove repeated clicks. Candidates are NOT excluded automatically.

## Runtime validation

The Python visualization and endpoint classifier must be checked on real
MATLAB-exported gaze; SciPy alone cannot inspect Tobii MCOS object contents.
