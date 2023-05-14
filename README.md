# Avengers Ensemble : Authorship Obfuscation
This repository contains implementation for the Avengers Ensemble which is an ensemble architecture for the text obfuscation using MUTANT-X. 

### Overview
This project aims to implement the Mutant-X authorship obfuscation techniques as proposed in the research paper, "Mutant-X: Efficient Authorship Obfuscation Techniques". The Mutant-X approach involves using a pipeline of Support Vector Machine (SVM) classifiers to identify the authorship of a given text and then modifying the text to obfuscate the authorship while maintaining the original meaning. This project improves on the original paper's approach by incorporating the Ensemble Vote classifier and utilizing the writeprint features to train the SVM classifiers.

### Features
- Pipeline of 10 SVM classifiers for ensemble attribution classifier
- Ensemble Vote classifier to drop low accuracy classifiers from the ensemble
- Writeprint features used to train the SVM classifiers, including:
  - Average character per word
  - Frequency of letters
  - Most common letter bigrams and trigrams
  - Digits percentage
  - Characters percentage
  - Frequency of digits
  - Frequency of word length
- Word2vec model from gensim to compute neighbors of words
- Mutant-X classifier with SVC trained on amazon review dataset as internal attribution classifier
- Local dictionary to save computed neighbors and reduce computing time
- Achieved an average METEOR score of 0.5 on the amazon review dataset, an improvement from the implemented paper.

### Usage
To use the Mutant-X authorship obfuscation techniques implemented in this project, follow these steps:
1. Clone this GitHub repository to your local machine.
2. Install the required dependencies listed in the requirements.txt file.
3. Prepare the input text file that you want to obfuscate.
4. Run the main.py script with the input text file as a command-line argument.
5. The obfuscated text will be saved in an output file with the same name as the input file, but with "_obfuscated" appended to the filename.

### Conclusion
The Mutant-X authorship obfuscation techniques implemented in this project offer an efficient and effective way to obfuscate the authorship of a given text. By utilizing the Ensemble Vote classifier and the writeprint features, this project improves on the original paper's approach and achieves better results on the amazon review dataset. The local dictionary also helps to reduce computing time, making the obfuscation process faster and more efficient.