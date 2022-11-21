
# BDH-reproducibility-challenge

Original project code: https://github.com/Xueping/tesan


## Dependencies
Since the project is complex and contains many dependancies it is easies to use the provided environments.yml file to create a conda enviroment with all the needed packages

(Tested on Apple Silicon Mac only)


## Datasets

The are two datasets required for this project, MIMIC III and CMS.

The MIMIC III dataset requires credentials to access from:
https://physionet.org/content/mimiciii/1.4/

At the bottom of the page download and place the following files in the dataset/mimic3 project's directory:
ADMISSIONS.csv <br/>
DIAGNOSES_ICD.csv<br/>
DRGCODES.csv<br/>
PRESCRIPTIONS.csv<br/>
PROCEDURES_ICD.csv<br/>

The CMS dataset is publicly avialable at:
https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs/DE_Syn_PUF

For each DE1.0 Sample (1-20) download the coresponding 2008-2010 Outpatient Claims (ZIP). Once all files are extracted move all the csv files to dataset/CMS in the project's directory

## Functionality 

Run src/dataset/data_preparation.py to preprocess the data.

The training and evaluation processes occur in src/embedding/concept/train_tesan.py and src/embedding/mortality/mortality_train.py depending on which of the experiments from the paper you are trying to run.


## Running The Code

Both the train_tesan.py and mortality_train.py each have many arguments depending on whcich model and experiment you are trying to train from the paper as well as room for the user to experiment with different hyperparameters themselves. 

All of them can be seen by adding -h or --help arguments when running the scripts

The commands bellow are some basic commands to create and evalute some of the models show in the tables/graphs from the paper.

(All examples use one epoch for to save time and still take time, in the paper 30 epochs was used)

python src/embedding/concept/train_tesan.py --data_source mimic3  --max_epoch 1 --train_batch_size 64 --visit_threshold=1 --num_samples 10  --skip_window 6

python src/embedding/concept/train_tesan.py --data_source cms  --max_epoch 1 --train_batch_size 128 --visit_threshold=1 --num_samples 5 --skip_window 6

python src/embedding/concept/train_tesan.py --model interval --data_source mimic3  --max_epoch 1 --train_batch_size 64 --visit_threshold=1 --num_samples 10  --skip_window 6

 python src/embedding/mortality/mortality_train.py --num_steps 100 --visit_threshold 1

 ## Notes On Changes

 The majority of the code is from the original paper's github repo (linked above). The biggest changes we performed was migrating the code from Tensoflow v1 to Tensorflow v2. Additionally we stripped out some portions of the code that did not perform anything as well as streamlined the code and the possible arguments.

 ## Whats Missing?

 The main thing that is missing from this code base is the code needed to train and create the various baseline models. The code also needs a testing mode to be able to test the models on a seperate test set instead of relying on training metrics.
