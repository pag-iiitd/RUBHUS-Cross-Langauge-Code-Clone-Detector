
# Rubhus-Cross-Langauge-Clone-Detector

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
- Java-Python Dataset - [Link](https://drive.google.com/file/d/1pOkkNpc9lmMXME8mCUYJRjl_-5GJzB6f/view?usp=sharing)  
- C-Java Dataset - [Link](https://drive.google.com/file/d/1pOkkNpc9lmMXME8mCUYJRjl_-5GJzB6f/view?usp=sharing)

#### 3.2 Setting up Dataset Files
- Unzip the downloaded files and extract the datasets files.
- Place these extracted files in the root directory of this repository

#### 3.3 Configuration of file paths
- .

## 💫 Usage 

### 1. Configuration of Hyperparameters

- Hyperparameters are defined inside the trainer files and can modified as per convenience. 

The hyperparameter variables explanation table is as follows : 
 
|  Var Name |  Hyperparameter | Default Value  |
|--|--|--|
| dim  |  |  |
| epochs |  |  |
| batch_size |  |  |
| lamda |  |  |
| separate_encoder |  |  |
| optimizer |  |  |
| scheduler |  |  |

### 2. Training RUBHUS Model
       python3 trainerRubhus.py

### 3. Training Baseline Model
       python3 trainerBaseline.py
      
### 4. Results 
- .


## ⭐ About the original setup 
- In our experiments we have trained Rubhus and Baseline Models for x and y epochs for Java Python Dataset and x2 and y2 epochs for C-Java Dataset. 
- The hyperparameters used in the original experiments as well as in this source code are reported in the paper.
- We have used GTx 2080Ti GPU to run our experiments. The time analysis of the tool also has been reported in the paper.

## 📑 Citing the project 

If you are using this for academic work, we would be thankful if you could cite the following paper.
`BIBTEX`

```
@{,
 author = {},
 title = {Improving Cross-Language Code CloneDetection via Code Representation Learning and Graph Neural Networks},
 ....
}
```

## ✍ Contact 
