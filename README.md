# Unimatch

## Overview

This project is designed for the segmentation of kerogen images using a semi-supervised learning approach. The code leverages PyTorch and the segmentation_models_pytorch library to implement a deep learning model that is trained on both labeled and unlabeled data.

## Project Structure
dataset/kerogens.py: Contains the KerogensDataset class for handling the dataset.
util/util.py: Utility functions.
util/train_helper.py: Helper functions for training.
util/eval_helper.py: Helper functions for evaluation.
Main script: The primary script that includes functions for setting the seed, loading data, running training epochs, and managing the training process.

## Setting Up

1. Clone the Repository

git clone <repository_url>
cd <repository_directory>

2. Prepare Dataset

Ensure that your dataset is organized as required by the KerogensDataset class.
Update the paths in the configuration accordingly.


## Running the Code

1. Set the Seed:

set_seed(42)

2. Get Arguments

Implement the get_args() function to parse necessary arguments such as data directories, save paths, etc.

3. Train the model
   trainer(args, config)


Contact
For any issues or questions, please open an issue on the repository or contact me at suraj.prasad@iitb.ac.in

