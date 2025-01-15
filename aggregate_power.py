import argparse

import matplotlib
from joblib import Parallel, delayed

matplotlib.use("Agg")
import os
import warnings
from pathlib import Path

import EEG_pyline.basic.arrange_data as arrange
import EEG_pyline.signal_processing.pre_process as pre_process
import EEG_pyline.signal_processing.spectral_analysis as spectr
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from mne_bids import (
    BIDSPath,
    get_entity_vals,
    make_report,
    print_dir_tree,
    read_raw_bids,
)

sns.set_palette("muted")
sns.set_style("whitegrid")

psd_params = dict(
    method="welch",
    fminmax=[1, 30],
    window="hamming",
    window_duration=2.5,
    window_overlap=0.5,
    zero_padding=3,
)

BANDS = dict(Delta=[1, 3.9], Theta=[4, 7.9], Alpha=[8, 12], Beta=[12.1, 30])

today = pd.Timestamp("today").strftime("%Y-%m-%d")

brain_regions = {
    "Left frontal": ["AF3", "F3", "FC1"],
    "Right frontal": ["AF4", "F4", "FC2"],
    "Left temporal": ["F7", "FC5", "T7"],
    "Right temporal": ["F8", "FC6", "T8"],
    "Left posterior": ["CP5", "P3", "P7"],
    "Right posterior": ["CP6", "P4", "P8"],
    "Fz": ["Fz"],
    "Cz": ["Cz"],
    "Pz": ["Pz"],
}


def read_failed_bids(file_path):
    """Read the failed_bids.txt file and return a set of excluded subjects."""
    excluded_subjects = set()
    with open(file_path, "r") as f:
        for line in f:
            excluded_subjects.add(line.strip())
    return excluded_subjects


def process_file(f, excluded_subjects):
    name = f.stem
    bids_parts = [t for t in f.stem.split("_")]
    ses = [b for b in bids_parts if "ses" in b][0]
    sub = [b for b in bids_parts if "sub" in b][0]
    task = [b for b in bids_parts if "task" in b][0].split("-")[1]
    subject_id = f"{sub}_{ses}"

    # Check if this subject-session combination should be excluded
    if f"{sub}_{ses}" in excluded_subjects:
        print(f"Skipping excluded subject-session: {sub}_{ses}")
        return None

    epochs = mne.read_epochs(fname=f, verbose=False)

    [psds, freqs] = spectr.calculate_psd(
        epochs, name, **psd_params, verbose=True, plot=False
    )

    df_psd_ch_bands = pd.DataFrame()
    for band, band_freqs in BANDS.items():
        psd_ch_band_temp = spectr.bandpower_per_channel(
            psds, freqs, band_freqs, band, name, epochs
        )
        df_psd_ch_band_temp = arrange.array_to_df(name, epochs, psd_ch_band_temp)
        df_psd_ch_band_temp["Condition"] = task
        df_psd_ch_band_temp["Timepoint"] = ses
        df_psd_ch_band_temp["Band"] = band
        df_psd_ch_band_temp["sub"] = sub.split("-")[1]
        df_psd_ch_band_temp["ses"] = ses.split("-")[1]
        df_psd_ch_band_temp["subject_id"] = subject_id

        df_psd_ch_bands = pd.concat([df_psd_ch_bands, df_psd_ch_band_temp])

    return df_psd_ch_bands


def main(bids_root, output_dir, failed_bids_file, n_jobs=-1):
    bids_root = Path(bids_root)
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root folder not found: {bids_root}")

    failed_bids_file = Path(failed_bids_file)
    if failed_bids_file and not failed_bids_file.exists():
        raise FileNotFoundError(f"Failed BIDS file not found: {failed_bids_file}")

    results_folder = Path(output_dir)
    results_folder.mkdir(parents=True, exist_ok=True)

    # Read the failed_bids.txt file if provided
    excluded_subjects = (
        read_failed_bids(failed_bids_file) if failed_bids_file else set()
    )

    target_tasks = ["restEO", "restEC"]

    matching_files = []
    for task in target_tasks:
        matching_files.extend(bids_root.rglob(f"*{task}*_epo.fif"))

    # Parallelize the processing of files
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_file)(f, excluded_subjects) for f in matching_files
    )

    # Combine results
    df_psd_ch_bands = pd.concat([r for r in results if r is not None])

    df_psd_reg_bands = arrange.df_channels_to_regions(
        df_psd_ch_bands.reset_index(), brain_regions
    )
    df_psd_reg_bands = df_psd_reg_bands.set_index(df_psd_ch_bands.index)
    df_psd_reg_bands.insert(0, "subject_id", df_psd_ch_bands["subject_id"])
    df_psd_reg_bands.insert(1, "sub", df_psd_ch_bands["sub"])
    df_psd_reg_bands.insert(2, "ses", df_psd_ch_bands["ses"])
    df_psd_reg_bands.insert(3, "Condition", df_psd_ch_bands["Condition"])
    df_psd_reg_bands.insert(4, "Timepoint", df_psd_ch_bands["Timepoint"])
    df_psd_reg_bands.insert(5, "Band", df_psd_ch_bands["Band"])

    channel_file = results_folder / f"channels_bandpowers_{today}.csv"
    region_file = results_folder / f"regions_bandpowers_{today}.csv"
    df_psd_ch_bands.to_csv(channel_file)
    df_psd_reg_bands.to_csv(region_file)

    print(f"Saved channel bandpowers to: {channel_file}")
    print(f"Saved region bandpowers to: {region_file}")
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate EEG power data from BIDS directory."
    )
    parser.add_argument(
        "-b",
        "--bids_root",
        type=str,
        required=True,
        help="Path to the BIDS root directory",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for saving CSV files",
    )
    parser.add_argument(
        "-f",
        "--failed_bids_file",
        type=str,
        help="Path to the failed_bids.txt file (optional)",
    )
    parser.add_argument(
        "-j",
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of jobs for parallel processing. Default is -1 (use all available cores).",
    )
    args = parser.parse_args()

    main(args.bids_root, args.output_dir, args.failed_bids_file, args.n_jobs)
