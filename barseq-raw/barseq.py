#!/usr/bin/env python3

"""
Main pipeline for barseq software

"""

import os
from copy import deepcopy
import sys
import logging
from pathlib import Path

# Module import
from utils import write_output, read_barcodes_new, format_filename, make_barseq_directories
from process_reads_fast import count_barcodes

import argparse



__author__ = "Vikash Pandey"
__email__ = "vikash.pandey@umu.se"

# Get logger
logger = logging.getLogger("barseq-obilab")


class Cd:
    """ Context manager for moving directories. """
    def __init__(self, new_path):
        self.new_path = new_path

    def __enter__(self):
        self.old_path = os.getcwd()
        os.chdir(self.new_path)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.old_path)


class Run:
    """ Class that stores settings for barseq processes. """
    def __init__(self, args):
        self.experiment = args.result
        self.sequences = Path(args.input)
        self.barcodes = Path(args.barcodes)
        # self.barseq_sample_collection = list()
        self.sample_dict = dict()
        self.path = f"{self.experiment}/"
        self.log = f"{self.path}log.txt"


class SampleRecord:
    """ Class for storing sample properties. """
    def __init__(self, sample: str, filename: str, barcode_dict):
        self.sample = sample
        self.filename = filename
        self.barcode_dict = deepcopy(barcode_dict)
        self.LEFT_SEQUENCE = ""
        self.RIGHT_SEQUENCE = ""




def main(args) -> None:
    """
    This is the main pipe line to analyze barseq counts.
    """
    # creating folder to put log file and barcode counts
    runner = Run(args) # here we create folder name which is equal to experiment name
    make_barseq_directories(runner) # if there is already folder then this will return error massage
    # Add file handler
    fh = logging.FileHandler(runner.log, mode="w")  # creating a log file
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M"))

    logger.addHandler(fh)
    logger.info("***** Starting barseq *****")

    # read barcode from fasta files
    logger.info(f"Reading in barcodes from {runner.barcodes.name}")

    # read barcode
    # barcodes = read_barcodes(runner.barcodes) # this is the old script

    barcodes = read_barcodes_new(runner.barcodes) # this is the old script

    # Process each sequencing file
    seq_files_list = sorted(os.listdir(runner.sequences))
    for seq_file in seq_files_list:
        if not seq_file.endswith(".DS_Store"):
            sample = format_filename(seq_file)
            logger.info(f"Counting Barcodes in {sample}")
            runner.sample_dict[sample+'_F'] = deepcopy(barcodes)
            runner.sample_dict[sample+'_R'] = deepcopy(barcodes)
            # Change cwd
            with Cd(runner.sequences):

                count_barcodes(seq_file, runner.sample_dict,[sample+'_F',sample+'_R'])



    # Write to output

    logger.info(f"Writing results to {runner.path}")
    write_output(runner.sample_dict, barcodes, runner)

    # Confirm completion of barseq
    logger.info("***** barseq is complete! *****")


if __name__ == "__main__":

    """ # -------- START HERE --------  # """

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Directory with Illumina reads in fastq format")
    parser.add_argument("-b", "--barcodes", help="CSV file with barcode and correspondent gene names (Barcode,Gene newline ATGAAGACTGTTGCCGTA,WT)")
    parser.add_argument("-r", "--result", help="Name of experiment, it is used for creating results folder")
    args_list=parser.parse_args()
    # args = sys.argv[1:]
    # barseq -i <directory of sequencing reads> -b <barcode file> -r <result folder>
    main(args_list)

