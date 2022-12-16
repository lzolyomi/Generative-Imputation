This repository is a reduced version of the code I wrote for my final thesis at the Technical University of Eindhoven.

# Introduction
This repo contains the implementation of Generative Adversarial Imputation Networks (GAIN) in Tensorflow 2.  
[Link to GAIN paper.](http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf) GAIN is a Deep Generative Network based imputation method, designed for tabular data sets. 
The repo contains the source code and a test notebook. 
- `model.py`: Contains `GAIN` class, the main class to interact with the imputation model.
- `runtime.py`: `run()` function used to initialize and train the GAIN model in the fastest way.
- `callbacks.py`: callbacks for logging metrics, early stopping and model save contained in the `Callbacks` class.
- `missingness.py`: `MissingMechanism` class used to introduce missingness in the (training) data.
- `utils.py`: utility functions used in the `GAIN` class.
- `evaluator.py`: `Evaluator` class used to give insights on a training session.

--- 

# Run GAIN
The easiest way to initialize a new `GAIN` object and train it is with the `run(args: dict, data: np.array)` function. This function accepts the arguments in a dictionary (see all arguments below) and the training data with no missing values (as a numpy array). 
The function then introduces missing values to the data, trains the GAIN model and returns the (artificially missing and imputed) training data and the fitted GAIN model.

### Arguments
Pass all arguments as one dictionary to the `run(args: dict, data: np.array)` function or the `GAIN` class directly. **Bold parameters** are required, others are optional, the model will run if they are not specified.

- **batch_size**: batch size for training the networks. GAIN divides the data onto batches of batch_size number of samples prior training.

- **hint_rate**: Hyperparameter for GAIN. Ranges [0,1]. Higher number means higher proportion of missingness indicator mask is revealed to Discriminator. If lower than 0.5 training is unstable.

- **alpha**: Hyperparameter for GAIN. Controls the weight of MSE Reconstrucion loss for generator. Higher value means higher weight on the MSE reconstruction loss, this will be minimized first during training.

- **beta**: Hyperparameter for GAIN. Controls the weight of Entropy Reconstruction loss for generator. Higher value means higher weight on the Entropy reconstruction loss, this will be minimized first during training.

- **epochs**: Number of training epochs to perform.

- **missing_rate**: (0,1] Proportion of artifical missingness the algorithm introduces for training. Always higher than zero.

- **learning_rate**: Learning rate of the optimizers for both networks.

- random_batches: float (0,1] or array length 2, elements float (0,1). Used to determine the proportion of batches used in each training epoch (batches are selected randomly). If array passed, random float in the specified range is sampled in each epoch. (e.g. `random_batches:(0.02, 0.1)` means that 2-10% of all batches will be used during training). If single float specified, number is used to determine proportion of batches used per epoch. If = 1.0 all batches used (default).

- log: {True, False} Creates a csv file with losses and errors and saves it in workdir/logs/ directory. Defaults to False.

- debug: {True, False} Enables debug prints for runtime and model. Defaults to False.

- rounding: {True, False} Round final imputed data to integers. Defaults to False.

- early_stop: {float, False} Enables early stopping. Float specifies the threshold the new lowest metric (loss) has to pass to be counted as new lowest point. Monitored metric is sum of generator and discriminator loss. Defaults to False.

- validation: {tuple, False} if Tuple passed it has to contain (missing test, ori_test, test_mask) where missing_test has values removed from ori_test according to mask. Assumes no missingness in ori_test! Defaults to False. Only shows results if log enabled.

- normalized_loss: {True, False} If True logs will include normalized RMSE scores too. Only visible if logs enabled. Defaults to False.

- OHE_features: array of pairs, indicating the first and last index of each one hot encoded categorical features. e.g. if the data contains two OHE features, one with columns ranging from col index 5 through 11 and the other with columns ranging from col index 15-33, this argument should be set to : [(5,12), (15, 34)] (notice that the second index always incremented by one). If the indices are not specified correctly, during initialization the `run` function gives warnings (via checks in the `MissingMechanism` class).

- gen_save: {int, False} After every specified epoch saves generator. Defaults to False (no saving) .

- discr_save: {int, False} After every specified epoch saves discriminator. Defaults to False (no saving). These can then be accessed in `GAIN.cb.generator` and `discriminator` respectively.

---

# The GAIN class
The GAIN class is the main object in the codebase. It handles the followings:
- initializes the two neural networks with predetermined parameters (two dense layers with same number of neurons as features in the data set),
- preprocesses the (missing) data by normalizing it, generating the hint matrix and dividing the data, the mask and the hint matrices onto batches,
- handle the training process for each batch in the `.train_step(X, M, H)` method,
- calculating losses and errors during and after training,
- imputing the artificially missing input data set

All of this can be done by calling the `.train(data, mask, ori_data)` method of an initialized GAIN object, where data is the data with missingness, mask is the missingness indicator matrix and ori_data is the original data set without any missing values. 
The trained GAIN object can be used to impute other data sets with missing values (assuming features do match up). For this call the `.impute(data: numpy array)` method with the data as a numpy array.

# The MissingMechanism class
Responsible to introduce missingness prior training to the training data without any missing values. The missingness introduced for categorical variables is in correspondence with OHE features. 
It first sorts continuous and OHE features by their column index, then introduces the missingness per column (or per feature for OHE features).
Also responsible to generate the hint matrix during the preprocessing stage. When calling the train method, GAIN initializes a MissingMechanism object that can be accessed in `GAIN.missing_mechanism`. The `run` function also initializes this object to introduce the missingness for the input data.

# The Callbacks class
Callbacks are used to add various features and monitoring to the training process.
- A Callbacks object is created when we create a GAIN object, and is accessible in `GAIN.cb`. This object also has access to all the arguments of the GAIN object, as the arguments dictionary is passed onto it during initialization.
- The object keeps track of all the training metrics via the `.add_log(name:str, value:float)` method. At the end of training the `create_log()` method generates a csv file with all the logged metrics. The csv file is saved in _/logs/train-run-{hour}-{minute}.csv_.
- If early stopping is enabled the `.early(metric: float)` method is used to monitor training. If the metric stagnates for 10 epochs the training process is stopped.
- Callbacks is also used to determine the random proportion of batches used per epoch (if enabled). 
- Finally, callbacks can also be used to store the generator and discriminator after certain number of iterations using the `.save_model(iter, generator, discriminator)` method. _iter_ is the number of iterations after which saving should happen (e.g. _iter = 5_ would mean that both networks are saved in every 5th epoch).

# The Evaluator class
Initialize it with a path to a csv logfile generated by Callbacks. Calling `.print_summary()` method prints a textual summary of the lowest value for each logged metric. The `.make_plot(col1:array, col2:array)` method accepts two lists containing names of logged metrics as arguments and creates a double scale Plotly lineplot with the chosen values. 

# runtime
In `runtime.py` there is a main function `run(args:dict, data: np.array)` that accepts a dictionary with arguments and the training data as numpy array. 
- First the function checks whether the data has no missing values. If it does, the training will still run but will give a printed warning that some metrics may be invalid (mainly RMSE).
- A `MissingMechanism` object is created to introduce the specified amount of missing values to our data.
- Then a `GAIN` object is initialized and trained using the artificially missing data with the `.train()` method.
- Finally it checks if the training ran properly (no collapse) and returns the artificially missing then imputed data, as well as the trained GAIN object.

## Example run 

At the end of `runtime.py` is a sample code that can be run on dummy data. The dummy data can be downloaded from the [UCI ML repo](https://archive.ics.uci.edu/ml/datasets/letter+recognition) or is also included in the [GitHub repo for the original GAIN](https://github.com/jsyoon0823/GAIN).
To run the example, first modify the *data_path* variable to point to the dummy data, then execute the file. The `run` function will be executed with logging and normalized metrics enabled, providing textual confirmations about these. The training should run automatically with a progress bar displayed in the command line. On the dummy data the training should last 5-10 minutes at most. Once its done, it will display the entire path and filename of the log file (by default it creates a directory 'logs' within the current workdir and saves the logfile there). 
After training, it will ask an input about the _{hour}-{minute}.csv_ ending of the logfile in order to create an `Evaluator` object. Then it will print out the training statistics and make a plot in the /logs/ directory on the training losses.

### For OHE variables
The dummy data only contains continuous variables, for data sets with OHE features, modify the OHE_features argument to contain the indices of OHE features (as in the arguments description). The rest of the training process is unchanged.

# utils
Utility functions, mainly used in the GAIN class. Used for different tasks such as sampling random matrices with different distributions, rounding imputed values and generating separator masks for OHE features during training of the GAIN model.
