# NERHRT
Source code for " Number-Enhanced Representation with Hierarchical Recursive Tree Decoding for Math Word Problem Solving".

Please note that the currently available code has not undergone a thorough refinement process. We intend to upload a more polished and robust version of the code after the paper's publication.

## Data
There are four datasets used in this work, Math23K, MAWPS, SVAMP and MathQA. Preprocessed data can be directly accessed in the `./data` but need to decompression.

## Experiment logs & model weights
All the training logs for the experiments in the paper and the corresponding model checkpoints can be accesses in this [link](https://drive.google.com/drive/folders/1uWwnyGx1Bi8Rv-E12kfNlDQPrwMmaKE9?usp=drive_link).

## Commands & reproducing results

### Getting started
* python == 3.8.x
* LTP == 4.1.5.post2 for the Chinese dataset
* transformers == 4.7.x
* CoreNLP == 4.4.0 for the English dataset

Do the following:
All the commands are listed in the scripts directory. 
Before rerunning the training process, please make sure the pretrained data are downloaded and untared into datasets directory; before doing the test, please make sure the model checkpoints are downloaded and untared into cache models directory.

### Train & replicate
Followers can rerun the training process by the `run.py` scripts with train in their name, or reproduce the experiment results by scripts with `test.py` in their name.

