# Advertising Spend and ROI Prediction

## Overview

In this guide, we will perform data analysis and data preparation tasks to train a Linear Regression model to predict future ROI (Return On Investment) of variable ad spend budgets across multiple channels including search, video, social media, and email using Snowpark for Python, Streamlit and scikit-learn. By the end of the session, you will have an interactive web application deployed visualizing the ROI of different allocated advertising spend budgets.

If all goes well, you should see the following app in your browser window.

https://user-images.githubusercontent.com/1723932/175127637-9149b9f3-e12a-4acd-a271-4650c47d8e34.mp4

## Prerequisites

* [Snowflake account](https://signup.snowflake.com/)
  * Login to your [Snowflake account](https://app.snowflake.com/) with the admin credentials that were created with the account in one browser tab (a role with ORGADMIN privileges). Keep this tab open during the workshop.
    * Click on the **Billing** on the left side panel
    * Click on [Terms and Billing](https://app.snowflake.com/terms-and-billing)
    * Read and accept terms to continue with the workshop

## Setup

### **Step 1** -- Create Tables, Load Data and Setup Stages

In Snowsight, open [setup.sql](setup.sql) and run all SQL statements to create tables and load data from AWS S3.

### **Step 2** -- Clone Repo

* `git clone https://github.com/Snowflake-Labs/sfguide-ad-spend-roi-snowpark-python-streamlit-scikit-learn.git` OR `git clone git@github.com:Snowflake-Labs/sfguide-ad-spend-roi-snowpark-python-streamlit-scikit-learn.git`

* `cd sfguide-ad-spend-roi-snowpark-python-streamlit-scikit-learn.git`

### **Step 3** -- Create And Activate Python 3.8 Environment

* Note: You can download the miniconda installer from
https://conda.io/miniconda.html. OR, you may use any other Python environment with Python 3.8
  
* `conda create --name snowpark -c https://repo.anaconda.com/pkgs/snowflake python=3.8`

* `conda activate snowpark`

### **Step 4** -- Install Snowpark for Python, Streamlit and other libraries in the environment

* `conda install -c https://repo.anaconda.com/pkgs/snowflake snowflake-snowpark-python pandas notebook scikit-learn cachetools streamlit`

### **Step 5** -- Update [connection.json](connection.json) with your Snowflake account details and credentials

* Note: For the **account** parameter, specify your [account identifier](https://docs.snowflake.com/en/user-guide/admin-account-identifier.html) and do not include the snowflakecomputing.com domain name. Snowflake automatically appends this when creating the connection.

## Data Engineering -- Data Analysis and Data Preparation

* In a terminal window, browse to this folder and run `jupyter notebook` at the command line. (You may also use other tools and IDEs such Visual Studio Code.)
* Open and run through the [Snowpark_For_Python_DE.ipynb](Snowpark_For_Python_DE.ipynb)
  * Note: Make sure in the Jupyter notebook the (Python) kernel is set to ***snowpark***

The notebook does the following...

* Establish secure connection to Snowflake
* Load data from Snowflake tables into Snowpark DataFrames
* Perform Exploratory Data Analysis on Snowpark DataFrames
* Pivot and Join data from multiple tables using Snowpark DataFrames
* Demostrate how to automate data preparation using Snowflake Tasks

## Machine Learning

* In a terminal window, browse to this folder and run `jupyter notebook` at the command line. (You may also use other tools and IDEs such Visual Studio Code.)
* Open and run through the [Snowpark_For_Python_ML.ipynb](Snowpark_For_Python_ML.ipynb)
  * Note: Make sure the Jupyter notebook (Python) kernel is set to ***snowpark***

The notebook does the following...

* Establish secure connection to Snowflake
* Load features and target from Snowflake table into Snowpark DataFrame
* Prepare features for model training
* Create a Python Stored Procedure to deploy model training code on Snowflake
* Create Python Scalar and Vectorized User-Defined Functions (UDF) for inference on new data points
  *NOTE: The Scalar UDF is called from the Streamlit Apps. See [Snowpark_Streamlit_Revenue_Prediction.py](Snowpark_Streamlit_Revenue_Prediction.py) and [Snowpark_Streamlit_Revenue_Prediction_SiS.py](Snowpark_Streamlit_Revenue_Prediction_SiS.py)*

## Streamlit Application

### Running it locally

* In a terminal window, browse to this folder and run the [Streamlit app](Snowpark_Streamlit_Revenue_Prediction.py) by executing `streamlit run Snowpark_Streamlit_Revenue_Prediction.py`

* If all goes well, you should see the following app in your browser window.

https://user-images.githubusercontent.com/1723932/175127637-9149b9f3-e12a-4acd-a271-4650c47d8e34.mp4

### Running it in Snowsight -- Streamlit-in-Snowflake (SiS)

* If you have SiS enabled in your account, follow these steps to run the application in Snowsight instead of locally on your machine.

  1) Click on **Streamlit Apps** on the left navigation menu
  2) Click on **+ Streamlit App** on the top right
  3) Enter **App name**
  4) Select **Warehouse** and **App locaton** (Database and Schema) where you'd like to create the Streamlit applicaton
  5) Click on **Create**
  6) At this point, you will be provided code for an example Streamlit application. Now open [Snowpark_Streamlit_Revenue_Prediction_SiS.py](Snowpark_Streamlit_Revenue_Prediction_SiS.py) and copy-paste the code into the example Streamlit application.
  7) Click on **Run** on the top right

* If all goes well, you should see the following app in Snowsight.

![Streamlin-in-Snowflake](assets/app_sis.png)

### Differences between two Streamlit Apps

The only difference between the two versions of Streamlit applications is how you create and access the Session object.

When running locally, you'd create and access the new Session object it like so:

```python
# Function to create Snowflake Session to connect to Snowflake
def create_session():
    if "snowpark_session" not in st.session_state:
        session = Session.builder.configs(json.load(open("connection.json"))).create()
        st.session_state['snowpark_session'] = session
    else:
        session = st.session_state['snowpark_session']
    return session
```

When running in Snowflake (SiS), you'd access the current Session object like so:

```python
session = snowpark.session._get_active_session()
```
