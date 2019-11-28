# CMSC818B_GPregression
This repo includes implementation of problems 4(a) and 4(b) defined [here](hw2.pdf). Both problems are modelled as a Gaussian Process. The training input and output data is used fit a
mapping from the input to the output data which is then used to predict the output mean and standard
deviation values for the test input data.

## Dependencies 
To run the code in this repo the following dependencies are required:
- Scikit-learn
- numpy
- matplotlib

## Execution
To run the code for problem 4(a) type in terminal:
```
python problem4a_sol.py
```
You can also use the flag `--kernel=rbf` for squared exponential kernel.

To run the code for problem 4(a) type in terminal:
```
python problem4b_sol.py
```
The results and implementation details for the two problems can be found in this [document](HW2_Abhinav_Modi.pdf).
