# Usage

This Python script is used to compute the main results presented in the paper. It is organized into sections so that it can be executed as a notebook. The first section loads the training dataset from the data_base_channel repertory, and the second section loads all the necessary functions. The remaining sections are used for data computation. Each section produces a .h5 result file, which is then used to generate the graphs shown in the paper. The main parameter for computation is the dataset size or SNR range, depending on the section.

# To use the python code:

- Download data_base_channel.zip via https://github.com/b-com/source_cell_free_cc/releases/download/v1.0.0-data_base_channel/data_base_channel.zip,
- extract archive near CC_based_exclusion.py.

# To access to the blender scene, use: 
https://github.com/b-com/source_cell_free_cc/releases/download/v1.0.0/factory.zip


# Requirements

The python packages used in the simulation are summurized in requirements.txt

# Computation Sections

1. Non-Linear Dimension Reduction (NLDR) Hyperparameter Tuning
This section evaluates different NLDR methods to determine their optimal hyperparameters. For each method, the trustworthiness (TW) and continuity (CT) metrics are computed for chart dimensions in the range [2,10] and number of neighbors in [4,15]. The optimal chart dimension and number of neighbors for each method are then selected based on the highest trustworthiness.

2. Exclusion Region Spectral Efficiency
Using the hyperparameters determined in section 1, this section computes the average spectral efficiency in the exclusion region with respect to SNR. The noise is calibrated using the conventional Zero-Forcing (ZF) precoder.

3. Secrecy Rate Computation
Based on the hyperparameters from section 1, this section computes the secrecy rate as a function of SNR, calibrated with the conventional ZF precoder.

4. Secrecy Rate with Training Dataset Subsampling
This section evaluates the secrecy rate for different subsampling fractions of the training dataset, again using the hyperparameters from section 1 and calibrating noise with the conventional ZF precoder.



