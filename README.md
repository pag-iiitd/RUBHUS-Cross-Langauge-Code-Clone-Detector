
# Rubhus-Cross-Langauge-Code-Clone-Detector

Paper Link - https://www.computer.org/csdl/journal/ts/5555/01/10242168/1QdYTNsq4Xm

This repository contains the source code for the paper - "Improving Cross-Language Code CloneDetection via Code Representation Learning and Graph Neural Networks"

## 📜 Code Organisation 
Current organisation contains files pertaining to models (`rubhusModel.py, baselineModel.py`), trainers (`trainerBaseline.py , trainerRubhus.py`) and some helper function file.  

    Repository
    ├── helper functions
    ├── models
    └── trainers
   
After setting up the repository, it would contain dataset files as well.

## ⚙ Setting Up 

### 1. Clone the repo

       git clone https://github.com/Akash-Sharma-1/Rubhus-Cross-Langauge-Clone-Detector.git

### 2. Installing Dependencies

       pip install -r requirements.txt

Note - Pytorch and Pytorch-Geometric (+ associated dependencies) versions must be installed in accordance the compatablity of Cuda version and operating system 

### 3. Setting up Datasets
The datasets which were used for experiments couldn't be uploaded to the repository due to file size limits. These files are to be downloaded and can be used independently for testing/running the models.

#### 3.1 Extraction of Dataset Files
- Java-Python Dataset - [Link](https://daniel.perez.sh/research/2019/cross-language-clones/)  
- C-Java Dataset - [Link](https://www.kaggle.com/datasets/arjoonn/codechef-competitive-programming)

#### 3.2 Setting up Dataset Files
- Unzip the downloaded files and extract the datasets files.
- Place these extracted files in the root directory of this repository

#### 3.3 Configuration of file paths
- **Dataset paths** - After extraction of the dataset, clone pair files and non-clone pair text files must be stored in the root directory in a folder named 'CloneDetectionSrc'. 
- **Processed Data folder** - A folder named 'cloneDetectionData' must be created  in the root directory where all the processed data files will be stored for training the model
- **Trained Models folder** - A folder named 'cloneDetectionModels' must be created  in the root directory where all the formed model files will be stored.


## 💫 Usage 

### 1. Configuration of Hyperparameters

- Hyperparameters are defined inside the trainer files and can modified as per convenience. 

The hyperparameter variables explanation table is as follows : 
 
|  Var Name |  Hyperparameter | Default Value  |
|--|--|--|
| dim  | Embedding size (dimension) for the model | 64 |
| epochs | #Epochs for the training  | 25 |
| batch_size | Size of the data batch | 32 |
| lamda | Regulariser  | 0.001 |
| use_unsup_loss | Usage of unsupervised loss in model training  | True |
| lr | Learning Rate (initial)  | 0.001 |
| optimizer | Optimizer of loss  | Adam |
| scheduler | Learning Rate Scheduler | ReduceLROnPlateau |

### 2. Training RUBHUS Model
       python3 trainerRubhus.py

### 3. Training Baseline Model
       python3 trainerBaseline.py


## ⭐ About the original setup 
- In our experiments we have trained Rubhus and Baseline Models for Java Python Dataset and for C-Java Dataset separately. 
- The hyperparameters used in the original experiments as well as in this source code are reported in the paper.
- We have used GTx 2080Ti GPU to run our experiments. The time analysis of the tool also has been reported in the paper.

## 📑 Citing the project 

If you are using this for academic work, we would be thankful if you could cite the following paper.
`BIBTEX`

```
@{,
 author = {Nikita Mehrotra*, Akash Sharma*, Rahul Purandare},
 title = {Improving Cross-Language Code CloneDetection via Code Representation Learning and Graph Neural Networks},
 ....
}
```

