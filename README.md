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

<img width="1388" alt="Screenshot 2024-04-25 at 12 54 15 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/e798cf15-be8d-4c50-a12f-c66779c3f99d">

* Split the data into training and testing datasets.

<img width="1388" alt="Screenshot 2024-04-25 at 12 54 29 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/df5a3219-caf1-4bd7-87f4-6a555e04c3cc">

* Scale the features datasets using StandardScaler.

<img width="1388" alt="Screenshot 2024-04-25 at 12 54 36 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/58aa74dd-8f64-490e-8974-219dcfb02746">

### Step 2: Compile, Train, and Evaluate the Model
* Create a neural network model using TensorFlow and Keras.
* Design hidden layers with appropriate activation functions.

<img width="1388" alt="Screenshot 2024-04-25 at 4 41 49 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/440e3f16-704a-41db-b61f-976aca95fab0">

* Compile and train the model.

<img width="1388" alt="Screenshot 2024-04-25 at 4 41 49 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/391a8810-192b-4f6c-8d95-294dda91b559">

* Evaluate the model's loss and accuracy using test data.

<img width="1388" alt="Screenshot 2024-04-25 at 4 44 30 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/74e677c2-60db-4385-87f3-94570f738dce">

* Save the results to an HDF5 file named AlphabetSoupCharity.h5.

<img width="1388" alt="Screenshot 2024-04-25 at 4 44 57 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/09ee4b4c-3761-483d-a530-7ff703b5d524">

### Step 3: Optimize the Model
#### Experiment with adjusting input data, adding neurons or layers, changing activation functions, and modifying training epochs.
* Import dependencies and preprocess the dataset.
* Design an optimized neural network model.
* Save the optimized results to AlphabetSoupCharity_Optimization.h5.

#### Attempt #1
<img width="1388" alt="Screenshot 2024-04-25 at 4 47 51 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/2f35f37f-170c-467f-86c3-db2b48f71e83">
<img width="1388" alt="Screenshot 2024-04-25 at 4 48 18 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/5b4a4405-3ea3-4737-8baa-e6b9d0cce112">
<img width="1388" alt="Screenshot 2024-04-25 at 4 49 09 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/95739354-2b29-4b7c-8311-48972d731593">
<img width="1388" alt="Screenshot 2024-04-25 at 4 49 22 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/d9431459-04e5-40ae-b4df-28cb09dfe1ac">

#### Attempt #2
<img width="1388" alt="Screenshot 2024-04-25 at 4 50 43 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/40f29466-d1fe-4907-b99c-14a41afa1577">
<img width="1388" alt="Screenshot 2024-04-25 at 4 50 54 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/66efb464-8483-4a88-9036-4789758cd86f">
<img width="1388" alt="Screenshot 2024-04-25 at 4 51 10 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/eb4c94fa-9c9e-4f33-8692-d77d2cc72144">

#### Attempt #3
<img width="1388" alt="Screenshot 2024-04-25 at 4 52 09 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/3ca19908-1531-4b91-baf3-a069ca994387">
<img width="1388" alt="Screenshot 2024-04-25 at 4 52 21 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/f9d2f6a8-4e1a-4606-a03b-ddae09c3ec3d">
<img width="1388" alt="Screenshot 2024-04-25 at 4 52 29 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/96b035f8-d62d-4e9d-9e1f-b24084d092b6">
<img width="1388" alt="Screenshot 2024-04-25 at 4 52 44 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/615ec458-9822-4fd3-ba95-fdc321951796">
<img width="1388" alt="Screenshot 2024-04-25 at 4 53 00 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/b6a4bfe3-0cdf-4436-8961-d37c5ccecb85">

#### Saving File
<img width="1388" alt="Screenshot 2024-04-25 at 4 54 19 PM" src="https://github.com/mattflanagan2/deep-learning-challenge/assets/146908072/47241f85-937e-46dd-9993-037ee9680bf9">


### Step 4: Report on the Neural Network Model
#### Overview of the analysis.
 The purpose of this analysis is to develop a predictive model leveraging machine learning techniques to assist Alphabet Soup, a nonprofit foundation, in selecting funding applicants with the highest likelihood of success. By utilizing neural networks and preprocessing a comprehensive dataset containing organizational metadata, the goal is to create a binary classifier that can effectively predict whether funded organizations will utilize the money effectively. Ultimately, this analysis aims to enhance Alphabet Soup's decision-making process, enabling more efficient allocation of resources and maximizing the impact of their funding initiatives.

#### Results
##### Data Preprocessing
 * Target Variable: IS_SUCCESSFUL
 * Feature Variable: All other columns
 * Columns that should be removed are the name and EIN columns as they are not needed for the evaluation of funding outcomes

##### Compiling, Training, and Evaluating the Model
* How many neurons, layers, and activation functions did you select for your neural network model, and why?
 I utilized three layers, an input, hidden layer, and output layer. I utilized 43 as the input value as there are 43 different columns. The first layer had 80 neurons, the second had 30, and the final had 1. 
* Were you able to achieve the target model performance?
  I was unable to achieve target model performance of over 75% in all three of my attempts. For my model, I opted for a structured approach, employing three layers: an input layer, a hidden layer, and an output layer. Given the dataset's complexity with 43 distinct columns, I set the input layer to accommodate this breadth of features. In the hidden layers, I strategically distributed neurons, selecting 80 neurons for the first layer, 30 for the second, and a single neuron for the output layer, aiming to balance model complexity and predictive power.  I aimed to ensure that the model could effectively learn from the diverse set of input variables. By strategically allocating neurons across layers, I sought to create a network architecture capable of capturing both high-level and nuanced patterns in the data, thus enhancing the model's predictive capacity.
  
* What steps did you take in your attempts to increase model performance?
  During my experimentation, I explored configurations with 4-5 layers and adjusted the number of neurons across these layers. This approach was motivated by the intuition that such depth and breadth in the network architecture could potentially enhance the model's accuracy. I also tried creating a bin for the ask_amt column to reduce the amount of data points to analyze.
  
* Summarize the model's performance and suggest improvements.
 All three attempts were unable to achieve target performace, I feel there is overfitting involved, or trying to analyze too much data resulting in noisy data. I feel that the best way forward would be to continue with binning the ask_amt column or adjusting how the model reads that column in some way. 





