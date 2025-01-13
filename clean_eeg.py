import datetime
import logging
import multiprocessing
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Set the backend to non-interactive before importing pyplot
import matplotlib.pyplot as plt

plt.ioff()  # Turn off interactive mode

import os

from joblib import Parallel, delayed

os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import EEG_pyline.signal_processing.pre_process as pre_process
import mne
from mne_bids import BIDSPath


def setup_logging(output_root):
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file = Path(output_root) / f"main_processing_log_{today}.txt"
    logging.basicConfig(
        filename=str(log_file),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)


def setup_file_logging(clean_path, name):
    log_file = clean_path / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


def process_file(args):
    name, f, clean_folder, config = args
    problems = []

    clean_path = Path(f.copy().update(root=clean_folder).fpath.parent)
    clean_path.mkdir(parents=True, exist_ok=True)
    logger = setup_file_logging(clean_path, name)

    try:
        logger.info("Working on %s", f)
        logger.info("config: %s", config)

        raw = mne.io.read_raw_bdf(
            f,
            infer_types=True,
            eog=config["eog_channels"],
            stim_channel=config["stimulus_channel"],
        )

        duration = raw.times[-1]
        if duration < 180:
            msg = f"Data is too short: {duration} s... skipping"
            problems.append((f, msg))
            logger.warning(msg)
            return problems

        logger.info(f"Cleaning EEG for {name}")

        raw = (
            raw.set_montage(mne.channels.make_standard_montage(config["montage"]))
            .load_data()
            .set_eeg_reference(ref_channels=config["reference"], verbose=False)
        )

        events = mne.find_events(
            raw,
            stim_channel=config["stimulus_channel"],
            consecutive=False,
            output="offset",
        )

        tminmax = process_events(events, raw, name, logger)

        if tminmax is not None:
            cropped_raw = raw.crop(tmin=tminmax[0], tmax=tminmax[1])
            logger.info(
                f"Signal cropped for {name}. Duration: {tminmax[1] - tminmax[0]} s"
            )
        else:
            cropped_raw = raw
            logger.info(f"Signal NOT cropped for {name}")

        cropped_raw = cropped_raw.drop_channels(config["stimulus_channel"])

        filt = pre_process.filter_raw_data(
            cropped_raw,
            config["filter_design"],
            line_remove=None,
            eog_channels=config["eog_channels"],
            plot_filt=False,
            savefig=False,
            verbose=False,
        )

        # fig = mne.viz.plot_raw_psd(filt, fmin=0, fmax=30, n_fft=4048, show=False)
        fig = filt.compute_psd(fmin=0, fmax=30, n_fft=4048).plot(show=False)
        fig.savefig(clean_path / f"{name}_filtered_psd.png")
        plt.close(fig)

        epochs = pre_process.artefact_rejection(
            filt, name, epo_duration=config["epochs_duration"], verbose=False
        )

        epochs.save(clean_path / f"{name}_epo.fif", overwrite=True)

        fig = epochs.compute_psd(fmin=0, fmax=30).plot(show=False)
        fig.savefig(clean_path / f"{name}_epo_cleaned_psd.png")
        plt.close(fig)

        logger.info(f"Finished processing {name}")

        plt.close("all")

    except Exception as e:
        msg = f"Error processing {name}: {str(e)}"
        problems.append((f, msg))
        logger.error(msg)

    return problems


def process_events(events, raw, name, logger):
    if len(events) >= 3:
        tminmax = [events[0][0] / raw.info["sfreq"], events[-1][0] / raw.info["sfreq"]]
        if len(events) > 3:
            logger.warning(f"More than 3 event points found for {name}")
    elif len(events) in [1, 2]:
        logger.warning(f"Only {len(events)} event point(s) found for {name}")
        if events[0][0] > 100000:
            tminmax = [0, events[0][0] / raw.info["sfreq"]]
        else:
            tminmax = [events[0][0] / raw.info["sfreq"], None]
    else:
        tminmax = None
        logger.warning(f"NO event points found for {name}")

    if tminmax is not None and (230 <= tminmax[1] - tminmax[0] <= 250) != True:
        logger.warning(f"Raw signal length is not between 230-250s for {name}")

    return tminmax


def main(bids_root, output_root, cpus, test=False):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    setup_logging(output_root)

    logging.info("Starting EEG pipeline")
    logging.info(f"Using {cpus} CPUs")
    logging.info(f"Input: {bids_root}")
    logging.info(f"Output: {output_root}")

    config = dict(
        cpus=cpus,
        eog_channels=["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"],
        stimulus_channel="Status",
        montage="biosemi32",
        reference="average",
        filter_design=dict(
            l_freq=0.5,
            h_freq=30,
            filter_length="auto",
            method="fir",
            l_trans_bandwidth="auto",
            h_trans_bandwidth="auto",
            phase="zero",
            fir_window="hamming",
            fir_design="firwin",
        ),
        epochs_duration=5,
        tasks=["restEO", "restEC"],
    )
    # prettry print the config to the log
    logging.info(f"Config: {config}")

    bids_root = Path(bids_root)
    bids_files = []

    for task in config["tasks"]:
        bids_path = BIDSPath(
            root=bids_root, task=task, extension=".bdf", datatype="eeg"
        )
        bids_files.extend(bids_path.match())

    if test:
        bids_files = bids_files[:2]

    names = [bids.fpath.stem for bids in bids_files]

    with Parallel(n_jobs=config["cpus"]) as parallel:
        results = parallel(
            delayed(process_file)((name, f, output_root, config))
            for name, f in zip(names, bids_files)
        )

    all_problems = [prob for sublist in results for prob in sublist]
    if all_problems:
        logging.error("Problems encountered during processing:")
        for f, msg in all_problems:
            logging.error(f"{f}: {msg}")

    logging.info("Finished EEG pipeline")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python script.py <bids_root> <output_root> [cpus] [--test]")
        sys.exit(1)

    bids_root = sys.argv[1]
    output_root = sys.argv[2]
    # if there was a third argument, use it as the number of CPUs
    if len(sys.argv) == 4:
        cpus = int(sys.argv[3])
    else:
        cpus = multiprocessing.cpu_count()

    # if --test in the arguments, only process the first two files
    test = "--test" in sys.argv

    main(bids_root, output_root, cpus, test)
