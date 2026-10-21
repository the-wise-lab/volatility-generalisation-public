# 1. Data preparation

`scripts/prepare_data.py` reads the behavioural data, applies the exclusions,
and writes one file per condition to `data/prepared/`. All later steps read
these files.

```bash
uv run python scripts/prepare_data.py
```

By default it reads `data/behavioural_data.csv.zip` and writes to
`data/prepared/`. The data format, the conditions and the outcome coding are
described in [`../data/README.md`](../data/README.md).

Useful options:

- `--main-data` and `--output-dir`: use a different input file or output
  folder.
- `--config`: the settings file (default `scripts/config.toml`), which holds
  the manual exclusion list and the exclusion thresholds.
- `--limit-subjects N`: keep only the first `N` participants of each task
  version, for a quick test.

## What it does

1. Keeps the study's participants (IDs that start `new_RLT_`).
2. Removes the duplicate record of block 2's last trial (see
   [the incomplete-block rule](../data/README.md#the-incomplete-block-rule)).
3. Splits the data into the 8 conditions.
4. Checks every exclusion criterion for every participant in every condition.
   A participant is excluded from a condition if they meet any of these:
   - they are on the manual exclusion list (`[data] excluded_subjects`);
   - low choice variability: the standard deviation of the rolling mean
     choice is below 0.2;
   - extreme mean choice: the mean choice is outside 1.25 to 1.75;
   - low engagement in another condition: either of the two choice criteria
     above is met in another condition (a participant who did not engage is
     removed from all conditions);
   - fewer than 120 trials in this condition.
5. Converts the outcomes to reward coding (1 = safe, 0 = explosion).
6. Writes `<condition>.npz` for each condition, and the two exclusion tables
   `qc_exclusion_audit.csv` and `qc_exclusion_counts.csv` (columns described
   in [`../data/README.md`](../data/README.md#data-and-results-files)).

The script stops with an error if the data do not match the task design, for
example if a block has more than 120 trials or a participant has a gap in
their trial numbers.
