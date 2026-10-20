# Datasets

The experiments use six public benchmark series. They are not redistributed here; place the CSV files
in this folder under the following names:

| File | Dataset | Variables | Resolution |
|---|---|---|---|
| `electricity.csv` | Electricity consumption of 321 clients (UCI) | 321 | hourly |
| `traffic.csv` | Road occupancy of 862 sensors (California PeMS) | 862 | hourly |
| `ETTm1.csv`, `ETTm2.csv` | Electricity transformer temperature | 7 | 15 minutes |
| `ETTh1.csv` | Electricity transformer temperature (cross-domain benchmark only) | 7 | hourly |
| `weather.csv` | Weather station measurements | 21 | 10 minutes |

All files are in the common long-horizon forecasting format: a `date` column, the input variables, and
the target column `OT`. The ETT files are published at <https://github.com/zhouhaoyi/ETDataset>. The
same preprocessed versions of all six files are distributed with the Autoformer and Time-Series-Library
repositories (<https://github.com/thuml/Autoformer>, <https://github.com/thuml/Time-Series-Library>);
follow the download links in their README files.

`dataloader.py` holds the dataset and scaling utilities. Missing-value sentinels (`-9999`, present in
Weather) are interpolated by `run_continual.py`.

The regimes are calendar seasons (within-dataset benchmark) or whole sources (cross-domain benchmark);
their boundaries are defined in `dataloader.py` (`SEASONAL_REGIME_REGISTRY`) and in
`configs/crossdomain/*.yaml`.
