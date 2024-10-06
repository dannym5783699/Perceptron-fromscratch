# Assignment2_NeuralNetworks
## Overview
This Project explores the performance and use of 4 different classification models.
- Widrow Hoff Learning
- Linear Support Vector Machine (SVM)
- Logistic Regression
- Weston Watkins

The first three use binary classification and the Binary Models tests runs these three models with the same learning 
rate and data set to extrapolate the performance difference between the three. The Weston Watkins SVM is a multiclass 
learning set that can be run using the Test Weston Watkins script. 
## Project Navigation
- `assignment2_neuralnetworks`: Contains the functional code for the project
    - `models` : Contains all 4 Models represented in python code
- `cv_results` : Contains all collected data from test Runs
- `docs` : Contains the assignment and analysis write-up of the project
- `figures` : Contains the generated graphs from the project
- `tests` : Contains all the test scripts to run the 4 Classification Models

## Build Poetry Environment
- Use the following lines in your terminal to set up the Poetry Environment With all dependent libraries.
```bash
poetry shell
poetry install
```
## How to Run Tests
### Binary Models
```bash
RunBM
```
### Weston-Watkins
```bash
RunWW
```
### Widrow Hoff
```bash
RunWH
```