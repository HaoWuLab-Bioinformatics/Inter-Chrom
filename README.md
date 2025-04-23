# Inter-Chrom
A novel deep learning framework with DNABERT’s word embedding for identifying Chromatin interactions along with motif importance investigation

## The framework of Inter-Chrom
![image](figure1.jpg)

## Overview
The folder "**data**" contains the datasets of three cell lines for Inter-Chrom to train and test.  
The folder "**getData**" contains get_token.py and motif_find.py.  
get_token.py: to divide the sequence by word, convert it into a token, which is represented by ID, and finally filter it into a subsequence composed of words with a frequency of the top 500 and a subsequence with an ID size of the top 500.  
motif_find.py: to match specific motifs in DNA sequences.  
The folder "**model**" contains 4 files.  
model.py: main framework of Inter-Chrom  
train.py: the process of training  
get_dataset.py: to obtain the samples for training and testing with chromosome-splitting strategy
DNABERT_embedding_matrix.npy: including the word vector corresponding to the token's ID  
The folder "**motif**" contains the position weight matrix (PWM) of motifs and the p-value threshold score from the HOCOMOCO Human v11 database.  

## Dependency
See "**requirements.txt"** for all detailed libraries.  
Other developers can use the following command to install the dependencies contained in "**requirements.txt"**:  
`pip install -r requirements.txt`  
