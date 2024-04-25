# Alphabet Soup Funding Predictor
## Background
The nonprofit foundation Alphabet Soup requires a tool to aid in selecting applicants with the best chance of success in their ventures. Leveraging machine learning and neural networks, the goal is to create a binary classifier predicting the success of applicants funded by Alphabet Soup. The dataset provided contains over 34,000 organizations that have received funding, with various metadata columns such as application type, affiliation, classification, and more.

## Instructions
### Step 1: Preprocess the Data
* Read in the charity_data.csv to a Pandas DataFrame.

<img width="1382" alt="Screenshot 2024-04-25 at 11 13 16 AM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/2492aa5c-ec1e-464e-80c7-8b90aa4dd0a8">


* Identify the target and feature variables for the model and drop the EIN and NAME columns.
  * Target Variable: IS_SUCCESSFUL
  * Feature Variable: Other columns Values
 
<img width="1348" alt="Screenshot 2024-04-25 at 11 13 37 AM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/afd03634-2f77-412d-bdb7-cf6e95afecef">


* Determine the number of unique values for each column.

<img width="1348" alt="Screenshot 2024-04-25 at 11 13 52 AM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/e7b33447-7df1-4027-adfa-bf528a2ee363">


* Bin "rare" categorical variables together if needed.

<img width="1388" alt="Screenshot 2024-04-25 at 11 20 36 AM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/ca14e99f-78ba-4241-be83-01e6ec1ab783">
<img width="1388" alt="Screenshot 2024-04-25 at 11 20 56 AM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/50d155bf-7715-4197-8609-cb78c1159166">
<img width="1388" alt="Screenshot 2024-04-25 at 11 21 04 AM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/c623ae7e-e710-451f-854a-df01c582ba67">

* Encode categorical variables using pd.get_dummies().



* Split the data into training and testing datasets.
* Scale the features datasets using StandardScaler.
### Step 2: Compile, Train, and Evaluate the Model
* Create a neural network model using TensorFlow and Keras.
* Design hidden layers with appropriate activation functions.
* Compile and train the model, saving weights every five epochs.
* Evaluate the model's loss and accuracy using test data.
* Save the results to an HDF5 file named AlphabetSoupCharity.h5.
### Step 3: Optimize the Model
* Experiment with adjusting input data, adding neurons or layers, changing activation functions, and modifying training epochs.
* Create a new Google Colab file named AlphabetSoupCharity_Optimization.ipynb.
* Import dependencies and preprocess the dataset.
* Design an optimized neural network model.
* Save the optimized results to AlphabetSoupCharity_Optimization.h5.
### Step 4: Write a Report
* Provide an overview of the analysis.
* Address data preprocessing and model compilation, training, and evaluation.
* Summarize the model's performance and suggest improvements.






