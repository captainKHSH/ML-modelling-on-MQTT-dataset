[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/yDpbj8_M)
<p align="center">
<img src="https://github.com/Systems-and-Toolchains-Fall-2023/course-project-option-2-captainKHSH/assets/74871590/867eb86c-ff80-4062-84f3-acc69975bb33" width="500">
<br/>
</p>

![GitHub repo size](https://img.shields.io/github/repo-size/Systems-and-Toolchains-Fall-2023/course-project-option-2-captainKHSH)
![GitLab last commit](https://img.shields.io/gitlab/last-commit/Systems-and-Toolchains-Fall-2023/course-project-option-2-captainKHSH)

# MQTT Dataset

This dataset contains information related to MQTT (Message Queuing Telemetry Transport) protocol.
MQTT, or Message Queue Telemetry Transport, dataset exists to bridge the gap between datasets on network traffic (KDDCUP99, NIMS, NLS-KDD) and datasets that focus on IoT but are missing raw traffic and packet capturing streams. To connect eight smart devices, each with a unique IP address, across two rooms in the proposed IoT home, a MQTT broker enables a thermostat, light, humidity, CO2, smoke, and motion sensor, fan, and door lock to communicate with each other. No firewall exists in this network, which makes it susceptible to attacks when a malicious node connects to it.

The three columns with “tcp” stand for Transmission Control Protocol, which connects an application with an IP address on the network. The next four columns with “mqtt.conack” refer to connection acknowledgement messages, which are metadata describing connection of devices to the network. Following those, the next eight columns with "conflag" describe the details of a specific session for a connected device. The next 18 columns with "mqtt" contain information related to the communication between devices on the network. Finally, the 34th column "target" denotes whether an event was one of five types of attacks or part of legitimate traffic.

## Collaboration in project
```bash
Kiran Prasad Jamuna Prasad [AndrewID: kjamunap]
```
```bash
Sriram Natarajan [AndrewID: ]
```

## Latest Development Changes
```bash
python -m pip install git+https://github.com/Systems-and-Toolchains-Fall-2023/course-project-option-2-captainKHSH
```

## Install
- [Download and install Anaconda](https://www.anaconda.com/download#downloads)
- [pgAdmin4](https://www.postgresql.org/)
- ```bash
  python -m pip install pyspark
  ```
-In the Anaconda Command Prompt, create a new virtual environment called 'aiml_env' with the necessary packages. Do this by running the following command:
- ``` bash
  conda create -n aiml_env numpy scipy pandas matplotlib seaborn ipykernel ipywidgets jupyter scikit-learn cvxopt xgboost pytorch torchvision torchaudio cpuonly -c pytorch
  ```
- Activate the environment with:
- ``` bash
  conda activate aiml_env
  ```
- Edit Jupyter notebooks in your browser by running:
- ``` bash
  jupyter notebook
  ```
  


## Features

| Feature                 | Description                                                  | Type                              |Nullable|
|-------------------------|--------------------------------------------------------------|-----------------------------------|---------|
| tcp.flags               | Flags associated with TCP (Transmission Control Protocol) communication. |text|not null|
| tcp.time_delta          | Time difference between TCP packets.                        |double precision||
| tcp.len                 | Length of the TCP packet.                                    |integer||
| mqtt.conack.flags       | MQTT connection acknowledgment flags.                        |text||
| mqtt.conack.flags.reserved | Reserved flags for MQTT connection acknowledgment.         |double precision||
| mqtt.conack.flags.sp   | MQTT connection acknowledgment flag - SP.                    |double precision||
| mqtt.conack.val         | Value associated with MQTT connection acknowledgment.        |double precision||
| mqtt.conflag.cleansess  | MQTT connection flag indicating clean session.               |double precision||
| mqtt.conflag.passwd    | MQTT connection flag indicating password usage.             |double precision||
| mqtt.conflag.qos       | MQTT connection flag indicating QoS (Quality of Service).   |double precision||
| mqtt.conflag.reserved  | Reserved flag for MQTT connection.                           |double precision||
| mqtt.conflag.retain    | MQTT connection flag indicating message retention.           |double precision||
| mqtt.conflag.uname     | MQTT connection flag indicating username usage.              |double precision||
| mqtt.conflag.willflag  | MQTT connection flag indicating will flag.                   |double precision||
| mqtt.conflags           | Combined MQTT connection flags.                              |text||
| mqtt.dupflag            | MQTT duplicate message flag.                                 |double precision||
| mqtt.hdrflags           | MQTT header flags.                                           |text||
| mqtt.kalive             | MQTT keep-alive interval.                                    |double precision||
| mqtt.len                | Length of the MQTT packet.                                   |double precision||
| mqtt.msg                | MQTT message.                                                |text||
| mqtt.msgid              | MQTT message identifier.                                     |double precision||
| mqtt.msgtype            | MQTT message type.                                           |double precision||
| mqtt.proto_len          | Length of the MQTT protocol.                                 |double precision||
| mqtt.protoname          | MQTT protocol name.                                          |text||
| mqtt.qos                | MQTT quality of service.                                     |double precision||
| mqtt.retain             | MQTT retain flag.                                            |double precision||
| mqtt.sub.qos            | MQTT subscribe quality of service.                            |double precision||
| mqtt.suback.qos         | MQTT subscribe acknowledgment quality of service.            |double precision||
| mqtt.ver                | MQTT protocol version.                                       |double precision||
| mqtt.willmsg            | MQTT will message.                                           |double precision||
| mqtt.willmsg_len        | Length of the MQTT will message.                              |double precision||
| mqtt.willtopic          | MQTT will topic.                                             |double precision||
| mqtt.willtopic_len      | Length of the MQTT will topic.                                |double precision||
| target                  | Target variable indicating a specific target.                |char string|not null|
|dataset                  |Diffrentiate between "train" or "test"                        |char string|not null|
|id                       |"matt_pkey" PRIMARY KEY, btree                                |integer|not null|

Please refer to the dataset documentation for more details on each feature and its representation.


## Constraints

Here are the constraints for columns in the dataset:

| Feature                 | Constraint                                                |
|-------------------------|-----------------------------------------------------------|
|tcp.len|Must be non-negative (>= 0).|
| mqtt.len                | Must be non-negative (>= 0).|
| tcp.time_delta               | Must be non-negative (>= 0).|
|mqtt.protoname         | Length must not exceed a specified maximum length.        |
|dataset|Check if the data is "train" or "test"|

Please refer to the dataset documentation for more details on the constraints and the dataset structure.


## Usage

To use this dataset, follow the instructions below:
1. [In this project, you will use MQTT Dataset available on Kaggle](https://www.kaggle.com/datasets/cnrieiit/mqttset)
2. Download the dataset.
3. Load both train and test dataset into one Postgres Database table.
5. Apply the constraints as needed for your analysis or machine learning tasks.

## Build and Populate Necessary Tables
1. Database and Schema Creation
Created a PostgreSQL database. Established a schema named "mqtt" within the database.
2. Data Ingestion
Ingested augmented train and test datasets from the Final CSV folder. Merged datasets into one table in the "mqtt" schema. Added a field to distinguish between train and test datasets.
3. Constraints
Implemented necessary constraints (e.g., primary key, foreign key) based on dataset requirements.

## Spark Analytics Functions
Developed Python functions using PySpark to answer the provided questions.Ingested data from the PostgreSQL table for analysis.
Specific Questions Analyzed
1. Average Length of MQTT Messages: Calculated the average length of MQTT messages in the training dataset.
2. Average Length of TCP Messages by Target Value: Computed the average length of TCP messages for each target value programmatically.
3. Most Frequent X TCP Flags: Built a Python function using PySpark to list the most frequent X TCP flags, handling tie scenarios.
4. Popular Target on Google News: Determined the most popular target on Google News using the provided query.
5. Decryption of Target Values: Decrypted target values to proper English equivalents if required.

## Machine Learning Modeling
1. Feature Engineering: Applied proper feature engineering principles. Conducted data cleaning and engineering.Check for Null and NA values in the columns.There are no null/NA values in the coloumns of our dataset. Thus, we do not need to do imputation for null/Na values. Checking duplicate rows. Classifying our variables
To classify our variables lets first check the unique values in the columns.
We can see that there are columns with only one unique value. Let's drop these columns.
Now, there are binary columns which has values as strings, values other than 0 and 1. Therefore, lets encode these columns with binary values 0 and 1. Also we will cast these columns to datatype double.
Lets investigate more on the column mqtt_msg.
As we can see this column has roughly half unique values with very large individual dataset numeric value, this column will cause the pipeline to fail. hence dropping this column.
Now, lets classify our variables.
Getting summary table for our dataset.
Lets see outliers and check if we need to handle them.
As there are no rows with more than 3 outliers, we will not drop any rows on the basis of outliers.
doing Correlation Matrix. List of correlated columns that needs to be removed.
correlated_col = ['mqtt_proto_len_encoded_binary','mqtt_protoname_encoded_binary','mqtt_conflag_uname','mqtt_qos','mqtt_len','mqtt_ver']
Now, let's handle further feature engineering steps including removing correlated columns, one hot encoding, vectorizing the features and outcomes in our pipeline transformer setup. Load the training and test dataframe using the pipeline.
2. Model Building: Created machine learning models,Using logistic regression as the first classification model. First we are fitting the model.Creating predicitions based on the above model.  Finding the Test and Train accuracy. Creating the Confusion matrix and finding the distribution.Do Cross-validation and analyse the hyperparameters and change accordingly. Then choosing Random Forrest Classifier is our second classification model. Finding the Test and Train accuracy. Creating the Confusion matrix and finding the distribution.Do Cross-validation and analyse the hyperparameters and change accordingly.
3. Parameter Tuning and Test Accuracy:Identified tunable parameters for each model. Tuned parameters using appropriate metrics. Ran the best-tuned models on the test dataset and recorded the test accuracy.
4. Pytorch ML modelling: Creating the pipeline and splitting our dataset to validation_data and test_data.
Creating tensors of train, test and validate datasets. Initializing instance of our model,The architecture of this MLP is designed for a specific input dimension (input_dim) and output dimension (output_dim).Adjust the input and output dimensions based on the requirements of your specific problem.
Initializing instance of our model.it employs a multilayer perceptron architecture with predefined hyperparameters such as a learning rate of 0.1, a batch size of 64, and 10 training epochs. The CrossEntropyLoss function is utilized as the criterion for evaluating the model's performance. Data loading is facilitated through PyTorch's DataLoader for both training and validation datasets. The Adam optimizer is chosen to update the model parameters during training. The code iterates through each epoch, conducting forward and backward passes, calculating losses, and optimizing the model's weights. Training and validation metrics, including losses and accuracies, are logged for analysis. Importantly, the model is saved as 'current_best_model' if the validation accuracy in the current epoch surpasses the best accuracy recorded so far. This strategy ensures that the best-performing model is preserved. The training loop prints relevant metrics for each epoch, offering insights into the model's progression and allowing users to monitor its performance over the specified number of epochs. Adjustments to hyperparameters, loss functions, and model architecture can be made to suit the specific requirements of the classification task at hand.
5. The best-performing model, which was previously saved during the training process, is loaded and evaluated on a test dataset. The mybestmodel instance is initialized as an instance of the myMultilayerPerceptron class, specifying the input dimension based on the shape of the training data. The model's state dictionary is then loaded with the parameters saved in the file named "current_best_model." Subsequently, a DataLoader is employed to load batches of the test dataset, and the model is applied to each batch to obtain prediction scores. The accuracy of the model on the test data is computed by comparing the predicted labels to the ground truth labels, and the average accuracy across all batches is calculated. Finally, the test accuracy is printed to the console, providing an assessment of the model's performance on previously unseen data. This evaluation on a separate test set helps ensure that the model generalizes well beyond the training and validation data, offering insights into its real-world applicability. Adjustments to the batch size or any other relevant parameters can be made based on specific requirements.
## Deploy Your Code to the Cloud
1. Cloud Deployment: Chose [Googel Cloud Consol] for deployment.
2. Creating the cluster on the cloud using Dataproc, assign the  master and worker nodes. Adapted the code for cloud deployment. run the code.
3. If you have issues running on the cloud then restart the kernel or change the region of the cluster.

## Code Walkthrough access from dropbox
https://www.dropbox.com/scl/fo/jtpy67wa8uhrxdi9lq77p/h?rlkey=4vuf880j0spvkmg7cbpvf52e9&dl=0


