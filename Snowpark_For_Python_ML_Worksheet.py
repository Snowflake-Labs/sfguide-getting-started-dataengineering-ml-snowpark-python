##### Prerequisite
# Run through https://github.com/Snowflake-Labs/sfguide-getting-started-dataengineering-ml-snowpark-python/blob/main/Snowpark_For_Python_DE_Worksheet.py
#####

##### Usage
# Load this file in Snowsight as a Python Worksheet. To learn more see https://docs.snowflake.com/en/user-guide/ui-snowsight-worksheets-gs
#####

##### Worksheet Overview
# Perform data analysis and data preparation tasks to train a Linear Regression model to predict future ROI (Return On Investment) of variable ad spend budgets across multiple channels including search, video, social media, and email using Snowpark for Python and scikit-learn.
# In Particular
# * Load features and target from Snowflake table into Snowpark DataFrame
# * Prepare features for model training
# * Create a Python Stored Procedure to deploy model training code on Snowflake
# * Create Python Scalar and Vectorized User-Defined Functions (UDF) for inference on new data points
#####

##### Introduction
# What is Snowpark?
# It allows data engineers and developers to query and transform data as well as write data applications in languages other than SQL using a set of APIs and DataFrame-style programming constructs in Python, Java, and Scala. These applications run on and take advantage of the same distributed computation on Snowflake's elastic engine as the SQL workloads. Learn more about Snowpark -- https://www.snowflake.com/snowpark/
#
# QuickStart Guide: https://quickstarts.snowflake.com/guide/getting_started_with_dataengineering_ml_using_snowpark_python/index.html
# YouTube: [TBD]
#####

##### Snowflake Anaconda Channel
# For convenience, Snowpark for Python and 1000s of other popular open source third-party Python packages that are built and provided by Anaconda are made available to use out of the box in Snowflake. There is no additional cost for the use of the Anaconda packages apart from Snowflake’s standard consumption-based pricing. To view the list of packages see https://repo.anaconda.com/pkgs/snowflake.

# Import Snowpark for Python
import os
from joblib import dump
import snowflake.snowpark as snowpark
from snowflake.snowpark.types import Variant
from snowflake.snowpark.functions import udf,sum,col,array_construct,month,year,call_udf,lit
from snowflake.ml.modeling.compose import ColumnTransformer
from snowflake.ml.modeling.pipeline import Pipeline
from snowflake.ml.modeling.preprocessing import PolynomialFeatures, StandardScaler
from snowflake.ml.modeling.linear_model import LinearRegression
from snowflake.ml.modeling.model_selection import GridSearchCV

def main(session: snowpark.Session):
    # What is a Snowpark DataFrame
    # It represents a lazily-evaluated relational dataset that contains a collection of Row objects with columns defined by a schema (column name and type). Here are some of the ways to load data in a Snowpark DataFrame are:
    # - session.table('table_name')
    # - session.sql("select col1, col2... from tableName")*
    # - session.read.options({"field_delimiter": ",", "skip_header": 1}).schema(user_schema).csv("@mystage/testCSV.csv")*
    # - session.read.parquet("@stageName/path/to/file")*
    # - session.create_dataframe([1,2,3], schema=["col1"])*

    ### Let's load Features and Target into a Snowpark DataFrame
    # At this point we are ready to perform the following actions to save features and target for model training.
    # Here we will
    # - Delete rows with missing values
    # - Exclude columns we don't need for modeling
    # - Save features into a Snowflake table called MARKETING_BUDGETS_FEATURES

    # Load Features and Target
    snow_df_spend_and_revenue_per_month = session.table('spend_and_revenue_per_month')

    # Delete rows with missing values
    snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.dropna()
    
    # Exclude columns we don't need for modeling
    snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.drop(['YEAR','MONTH'])
    
    # Save features into a Snowflake table call MARKETING_BUDGETS_FEATURES
    snow_df_spend_and_revenue_per_month.write.mode('overwrite').save_as_table('MARKETING_BUDGETS_FEATURES')

    print("Features And Target")
    # See the output of this command in "PY Output" tab below
    snow_df_spend_and_revenue_per_month.show()

    print("Training model")
    ### Model Training in Snowflake
    #### Python function to train a Linear Regression model using scikit-learn
    #
    # Let's create a simple modle using the *scikit-learn* like modeling primitives from 
    # the Snowpark ML package which trains the model in Snowflake Warehouse.
    #
    # Snowpark ML is a set of tools including SDKs and underlying infrastructure to build 
    # machine learning models. With Snowpark ML, you can pre-process data, train ML models all 
    # within Snowflake, using a single SDK, and benefit from Snowflake’s proven performance, 
    # scalability, stability and governance at every stage of the Machine Learning workflow.
    #
    # TIP: Learn more about [Snowpark-optimized Warehouses](https://docs.snowflake.com/en/user-guide/warehouses-snowpark-optimized.html).

    CROSS_VALIDATION_FOLDS = 10
    POLYNOMIAL_FEATURES_DEGREE = 2

    # Create train and test snowpark dataframes
    train_df, test_df = session.table("MARKETING_BUDGETS_FEATURES").random_split(weights=[0.8, 0.2], seed=0)

    # BUILD MODEL


    # Preprocess the Numeric columns
    # We apply PolynomialFeatures and StandardScaler preprocessing steps to the numeric columns
    # NOTE: High degrees can cause overfitting.
    numeric_features = ['SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL']
    numeric_transformer = Pipeline(steps=[('poly',PolynomialFeatures(degree = POLYNOMIAL_FEATURES_DEGREE)),('scaler', StandardScaler())])

    # Combine the preprocessed step together using the Column Transformer module
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)])

    # The next step is the integrate the features we just preprocessed with our Machine Learning algorithm to enable us to build a model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LinearRegression())])
    parameteres = {}


    # Use GridSearch to find the best fitting model based on number_of_folds folds
    model = GridSearchCV(
        estimator=pipeline,
        param_grid=parameteres,
        cv=CROSS_VALIDATION_FOLDS,
        label_cols=["REVENUE"],
        output_cols=["PREDICTED_REVENUE"]
    )

    # TRAIN
    model.fit(train_df)

    # EVAL
    train_r2_score = model.score(train_df)
    test_r2_score = model.score(test_df)

    # Print model R2 score on train and test data
    print(
        f"R2 score on Train: {train_r2_score},"
        f"R2 score on Test: {test_r2_score}"
    )

    print("Saving model")
    # Save model
    ###Extract SKLearn object 
    sk_model = model.to_sklearn()

    model_output_dir = '/tmp'
    model_file = os.path.join(model_output_dir, 'model.joblib')
    dump(sk_model, model_file)
    session.file.put(model_file, "@dash_models", overwrite=True)

    ### Create Scalar User-Defined Function (UDF) for inference
    # Now to deploy this model for inference, let's **create and register a Snowpark Python UDF and add the trained model as a dependency**. Once registered, getting new predictions is as simple as calling the function by passing in data.
    # Note: Scalar UDFs operate on a single row / set of data points and are great for online inference in real-time.
    # TIP: Learn more about Snowpark Python User-Defined Functions https://docs.snowflake.com/en/developer-guide/snowpark/python/creating-udfs.html

    session.clear_imports()
    session.clear_packages()

    # Add trained model and Python packages from Snowflake Anaconda channel available on the server-side as UDF dependencies
    session.add_import('@dash_models/model.joblib.gz')
    session.add_packages('pandas','joblib','scikit-learn==1.1.1')

    @udf(name='predict_roi',session=session,replace=True,is_permanent=True,stage_location='@dash_udfs')
    def predict_roi(budget_allocations: list) -> float:
        import sys
        import pandas as pd
        from joblib import load
        import sklearn
    
        IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
        import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]
        
        model_file = import_dir + 'model.joblib.gz'
        model = load(model_file)
                
        features = ['SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL']
        df = pd.DataFrame([budget_allocations], columns=features)
        roi = abs(model.predict(df)[0])
        return roi
    
    ### Call Scalar User-Defined Function (UDF) for inference on new data
    # Once the UDF is registered, getting new predictions is as simple as calling the call_udf() Snowpark Python function and passing in new datapoints.
    # Let's create a SnowPark DataFrame with some sample data and call the UDF to get new predictions.

    test_df = session.create_dataframe([[250000,250000,200000,450000],[500000,500000,500000,500000],[8500,9500,2000,500]], 
                                    schema=['SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL'])
    test_df = test_df.select(
        'SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL', 
        call_udf("predict_roi", 
        array_construct(col("SEARCH_ENGINE"), col("SOCIAL_MEDIA"), col("VIDEO"), col("EMAIL"))).as_("PREDICTED_ROI"))

    # See the output of this in "Results" tab below
    return test_df