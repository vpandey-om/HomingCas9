#!/usr/bin/env python3

"""
Count barcode frequency in fastq/fasta files given by user.

"""

import screed
import logging
import regex as re


__author__ = "Vikash Pandey"
__email__ = "vikash.pandey@umu.se"

# Get logger
logger = logging.getLogger("barseq-obilab")


def count_barcodes(seq_file, barcode_dict,samples) -> None:
    """
    Count barcode frequency in sequence file.
    Returns a DataFrame object

    :param seq_file: file with reads
    :param barcode_dict: barcode dictionary of sample
    :return:
    """

    # the sequence of the tag amplification primer
    ba_primer = 'GTAATTCGTGCGCGTCAG';
    # sequence from cassette primer R2 to primer binding site for arg97
    # this is the same for all constructs
    r2_to_amp97 = 'CCGCCTACTGCGACTATAGAGATATCAACCACTTTGTACAAGAAAGCTGGGTGGTACCCATCGAAATTGAAGG';

    ba_primer_end = ba_primer[-5:]; # get end of 5 base pair of BA primer
    ba_primer_end_rc = reverse_complement(ba_primer_end) # get reverse complement

    r2_start= r2_to_amp97[:5]; #  get start of 5 base pair of cassette primer R2
    r2_start_rc= reverse_complement(r2_start)
    flank_regex_fwd = re.compile(ba_primer_end+"(\w{8,16})"+r2_start)
    flank_regex_rev = re.compile(r2_start_rc+"(\w{8,16})"+ba_primer_end_rc)
    _other_reads = list()
    with screed.open(seq_file) as reads:
        n_reads1 = 0

        n_reads2 = 0
        for read in reads:
            barcode_dict_fwd,n_reads1=applycountFast(read,barcode_dict[samples[0]],flank_regex_fwd,n_reads1,_other_reads,flag="fwd")
            barcode_dict_rev,n_reads2=applycountFast(read,barcode_dict[samples[1]],flank_regex_rev,n_reads2,_other_reads,flag="rev")
    calMatchReads(seq_file,barcode_dict_fwd,n_reads1,flag="forward")
    calMatchReads(seq_file,barcode_dict_rev,n_reads2,flag="reverse")
    # Compile regex patterns
    #flank_regex = re.compile("(GCTCATGCACTTGATTCC){e<=1}([ATGC]{18})(GACTTGACCTGGATGTCT){e<=1}")

     # import pdb;pdb.set_trace()
    # barcode_regex = dict()

    # import pdb;pdb.set_trace()
    # barcode_regex = dict()
    # for b in barcode_dict[samples[0]]:
    #     #barcode_regex[b] = re.compile("(%s){e<=1}" % b)
    #     barcode_regex[b] = re.compile("(%s)" % b)
    # # Open sequence file
    # with screed.open(seq_file) as reads:
    #     n_reads1 = 0
    #     n_reads2 = 0
    #     for read in reads:
    #         # barcode_dict_fwd,n_reads1=applyCount(read,barcode_dict[samples[0]], barcode_regex,flank_regex_fwd, _other_reads,n_reads1,flag="fwd")
    #         # barcode_dict_rev,n_reads2=applyCount(read,barcode_dict[samples[1]], barcode_regex,flank_regex_rev, _other_reads,n_reads2,flag="rev")
    #         barcode_dict_fwd,n_reads1=applycountFast(read,barcode_dict[samples[0]], barcode_regex,flank_regex_fwd, _other_reads,n_reads1,flag="fwd")
    #         barcode_dict_rev,n_reads2=applycountFast(read,barcode_dict[samples[1]], barcode_regex,flank_regex_rev, _other_reads,n_reads2,flag="rev")
    #
    # calMatchReads(seq_file,barcode_dict_fwd,n_reads1,flag="forward")
    # calMatchReads(seq_file,barcode_dict_rev,n_reads2,flag="reverse")

def reverse_complement(dna):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A','N':'N'}
    return ''.join([complement[base] for base in dna[::-1]])


def applycountFast(read,barcode_dict,flank_regex,n_reads,_other_reads,flag="rev"):

    try:
        putative_barcode = re.search(flank_regex, read.sequence)[1]
        if flag=="rev":
            putative_barcode=reverse_complement(putative_barcode)
        if putative_barcode in barcode_dict.keys():
            barcode_dict[putative_barcode]["count"] += 1
        else:
            barcode_dict["_other"]["count"] += 1
            _other_reads.append(read)

    except TypeError:
        barcode_dict["_other"]["count"] += 1
        _other_reads.append(read)
    n_reads += 1
    return barcode_dict,n_reads



def applyCount(read,barcode_dict, barcode_regex,flank_regex, _other_reads,n_reads,flag="rev"):

    try:
        putative_barcode = re.search(flank_regex, read.sequence)[1]
        if flag=="rev":
            putative_barcode=reverse_complement(putative_barcode)

        for known_barcode in barcode_regex:
            if re.search(barcode_regex[known_barcode], putative_barcode):
                barcode_dict[known_barcode]["count"] += 1
                break
                # Putative barcode present, does not match known barcodes
            else:
                barcode_dict["_other"]["count"] += 1
                _other_reads.append(read)
            # No putative barcode present
    except TypeError:
        barcode_dict["_other"]["count"] += 1
        _other_reads.append(read)

    n_reads += 1

    return barcode_dict,n_reads

def calMatchReads(seq_file,barcode_dict,n_reads,flag):
    # Calculate matched reads
    matched_reads = sum([x['count'] for x in barcode_dict.values() if x["gene"] != "_other"])
    _other_reads = barcode_dict['_other']['count']

    logger.info(f"For {flag} {seq_file}, {matched_reads} of "
                f"{n_reads} ({round((matched_reads/n_reads) * 100, 2)}%) matched known barcodes.")
    logger.info(f"Reads without barcode match: {_other_reads} ({round((_other_reads/n_reads)*100, 2)}%) for {seq_file}")
    return

if __name__ == '__main__':
    pass
