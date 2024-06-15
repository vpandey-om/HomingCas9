# HomingCas9 

This repository contains the analysis of barcode sequencing (barseq) data aimed at studying oocyst phenotypes in a rodent malaria parasite, Plasmodium berghei [Preprint]((https://www.biorxiv.org/content/10.1101/2024.06.02.597011v1)).

## Overview
We used barcoded PlasmoGEM knockout vectors or modified homing vectors that carry gRNA to mutagenize a P.berghei background line  that express Cas9 protein and produce only fertile male gametocytes (male/fd1-). Our pilot screen covered 21 targetable genes, allowing us to study effect of homing in revealing oocyst phenotypes in scale. Our study revealed the effect of homing in revealing post-fertilization phenotypes of P.berghei genes.  

In our study, we transfected into male  only lines with either homing vector or PlasmoGEM vector which were used to infect two groups of mice. Subsequently, transfected parasites were selected using pyrimethamine and then used to infect a new mouse together with a line providing female gametocytes for mosquito feeding.

On day 0, the male mutants transfected  with either  PlasmoGEMvectororhoming vector were separately fed to mosquitoes in duplicate: "Mosquito Feed 1" (MF1, indicated by blue dots) and "Mosquito Feed 2" (MF2, indicated by red dots). For each mosquito feed, we collected one day 0 input sample. Therefore, on day 0, we generated a total of four samples.

On day 13 or 14, we collected output samples from each mosquito feed in duplicate. Consequently, on day 13 or 14, we generated a total of eight samples. 

![Sample description](https://github.com/vpandey-om/Fertility_screen/blob/master/output/sample.png)

Within this repository, you will find scripts and datasets detailing our analysis of raw barseq data and the computation of oocyst conversion rate phenotype metrics. Additionally, we have included the scripts used to generate figures featured in our research paper.


## Prerequisites

Users need to install before using the Snakemake workflow.

- Python (>=3.7)
- Snakemake (7.32.4)

## Installation

Install Snakemake using pip.
~~~
pip install snakemake
~~~
## Concepts and Descriptions
In this section, we will explain the terminology and concepts that were employed in the calculation of oocyst conversion rate and calculation of off target effect by gRNA.

### Relative abundance
To calculate the abundance of mutants in each sample, we averaged the forward and reverse reads. Subsequently, we computed the abundance for each pool. For example, for a pool of 3 mutants/genes, the count matrix can be generated as shown in the following table.

| Genes | Sample  1 | Sample  2 |
|----------|----------|----------|
| Gene1 | 10.5 | 11.5 |
| Gene2 | 30 | 10.5 |
| Gene3|  20.5| 5.5 |



We determine the relative abundance by dividing the abundance of a specific feature by the total abundance in the sample. The resulting relative abundance table is as
follows:

| Genes | Sample  1 | Sample  2 |
|----------|----------|----------|
| Gene1 | 10.5/60 | 11.5/31 |
| Gene2 | 30/60 | 10/31 |
| Gene3|  20.5/60| 5.5/31 |

### Oocyst conversion rate

#### Step 1: Compute mean and variance for each mutant
On day 0 and day 13, we utilized relative abundance to calculate the mean and variance for each set of PCR duplicates. Subsequently, we applied a logarithmic transformation to the mean values (log2(mean)) and computed the relative variability represented by the coefficient of variation squared (CV^2).

#### Step 2: Inverse-variance weighting   
Inverse-variance weighting is a statistical technique that combines multiple random variables to reduce the variance of the weighted average. It is particularly useful in our analysis where we calculate the change in barcode abundance between day0 and day13. This change is computed as the difference between the logarithms of the mean abundance at day13 and day0. Additionally, we consider the variance of the data, which is propagated as the sum of relative variances at day0 and day13. This allows us to effectively assess changes in barcode abundance while accounting for the variability in the data.

#### Step 3: Normalized by spike-in controls
In our study, we included spike-in controls to normalize the change in barcode abundance. This normalized change in barcode abundance, which we refer as "oocyst conversion rate".

For each pool, we computed the oocyst conversion rate of mutants. Mutants were categorized as `Reduced` if their oocyst conversion rate, plus two times the standard deviation, fell below a certain cutoff value; otherwise, they were categorized as `Not Reduced`. Since spike-in controls were included in all pools, we employed inverse-variance weighting to obtain a consolidated measure of oocyst conversion rate and the associated error for this variable. This approach allowed us to effectively combine data from multiple pools while considering variations introduced by the spike-in controls.

## Usage

### Convert Fastq to count matrix
To convert paired forward and reverse reads from BARseq experiments into a count matrix, we employ the following command.
~~~
snakemake --use-conda --conda-frontend conda raw_barseq  --cores 1 -F --config input_file=barseq-raw/testdata/sequence barcode_file=barcode_to_gene_210920.csv output_dir=barseq-raw/testRes -p  
~~~
The process requires two key inputs: a directory (e.g., `barseq-raw/testdata/sequence`) where all the fastq files for the samples are stored and a CSV file (e.g., `barcode_to_gene_210920.csv`) containing barcode information for each gene or mutant. The resulting output is directed to another directory (e.g., `barseq-raw/testRes`), where both the mutant count matrix file and a log file are generated.

To identify and remove mutants with zero counts in all samples, use the following command. This will generate the file `removed_zeros_barcode_counts_table.csv`
in the `barseq-raw/testRes` directory.
~~~
snakemake --use-conda --conda-frontend conda remove_zeros  --cores 1 -F --config output_dir=barseq-raw/testRes -p
~~~

### Count matrix to conversion rate
To convert the count matrix to the oocyst conversion rate for (homing-) screen, use this script:
~~~
python codes/rawToConversion.py
~~~

To convert the count matrix to the oocyst conversion rate for (homing+) screen, use this script:

~~~
python codes/rawToConversion_homing.py codes/rawToConversionRate.py
~~~

### Plot figures
To generate figures, use this script:
~~~
python codes/plotFigures.py
~~~

