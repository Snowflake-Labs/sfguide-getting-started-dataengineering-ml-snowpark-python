#!/usr/bin/env python

# Snowpark for Python
from snowflake.snowpark.session import Session
from snowflake.snowpark.types import IntegerType, StringType, StructType, FloatType, StructField, DateType, Variant
from snowflake.snowpark.functions import udf, sum, col,array_construct,month,year,call_udf,lit
from snowflake.snowpark.version import VERSION
# Misc
import json
import pandas as pd
import logging 
logger = logging.getLogger("snowflake.snowpark.session")
logger.setLevel(logging.ERROR)

def connect_to_snowflake():
  # Create Snowflake Session object
  print("Connecting to Snowflake...")
  connection_parameters = json.load(open('connection.json'))
  session = Session.builder.configs(connection_parameters).create()

  # Current Environment
  print("Current Environment...")
  snowflake_environment = session.sql('select current_user(), current_role(), current_database(), current_schema(), current_version(), current_warehouse()').collect()
  snowpark_version = VERSION
  print('   User                        : {}'.format(snowflake_environment[0][0]))
  print('   Role                        : {}'.format(snowflake_environment[0][1]))
  print('   Database                    : {}'.format(snowflake_environment[0][2]))
  print('   Schema                      : {}'.format(snowflake_environment[0][3]))
  print('   Warehouse                   : {}'.format(snowflake_environment[0][5]))
  print('   Snowflake version           : {}'.format(snowflake_environment[0][4]))
  print('   Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_version[2]))
  return session

def data_pipeline_feature_engineering(session: Session) -> str:
  try:
    # DATA TRANSFORMATIONS
    # Perform the following actions to transform the data

    # Load the campaign spend data
    snow_df_spend = session.table('campaign_spend')

    # Transform the data so we can see total cost per year/month per channel using group_by() and agg() Snowpark DataFrame functions
    snow_df_spend_per_channel = snow_df_spend.group_by(year('DATE'), month('DATE'),'CHANNEL').agg(sum('TOTAL_COST').as_('TOTAL_COST')).\
        with_column_renamed('"YEAR(DATE)"',"YEAR").with_column_renamed('"MONTH(DATE)"',"MONTH").sort('YEAR','MONTH')

    # Transform the data so that each row will represent total cost across all channels per year/month using pivot() and sum() Snowpark DataFrame functions
    snow_df_spend_per_month = snow_df_spend_per_channel.pivot('CHANNEL',['search_engine','social_media','video','email']).sum('TOTAL_COST').sort('YEAR','MONTH')
    snow_df_spend_per_month = snow_df_spend_per_month.select(
        col("YEAR"),
        col("MONTH"),
        col("'search_engine'").as_("SEARCH_ENGINE"),
        col("'social_media'").as_("SOCIAL_MEDIA"),
        col("'video'").as_("VIDEO"),
        col("'email'").as_("EMAIL")
    )

    # Load revenue table and transform the data into revenue per year/month using group_by and agg() functions
    snow_df_revenue = session.table('monthly_revenue')
    snow_df_revenue_per_month = snow_df_revenue.group_by('YEAR','MONTH').agg(sum('REVENUE')).sort('YEAR','MONTH').with_column_renamed('SUM(REVENUE)','REVENUE')

    # Join revenue data with the transformed campaign spend data so that our input features (i.e. cost per channel) and target variable (i.e. revenue) can be loaded into a single table for model training
    snow_df_spend_and_revenue_per_month = snow_df_spend_per_month.join(snow_df_revenue_per_month, ["YEAR","MONTH"])

    # SAVE FEATURES And TARGET
    # Perform the following actions to save features and target for model training

    # Delete rows with missing values
    snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.dropna()

    # Exclude columns we don't need for modeling
    snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.drop(['YEAR','MONTH'])

    # Save features into a Snowflake table call MARKETING_BUDGETS_FEATURES
    snow_df_spend_and_revenue_per_month.write.mode('overwrite').save_as_table('MARKETING_BUDGETS_FEATURES')
    return "SUCCESS"
  except:
    return "FAIL"

def train_revenue_prediction_model(
  session: Session, 
  features_table: str, 
  number_of_folds: int, 
  polynomial_features_degrees: int, 
  train_accuracy_threshold: float, 
  test_accuracy_threshold: float, 
  save_model: bool) -> Variant:
  
  from sklearn.compose import ColumnTransformer
  from sklearn.pipeline import Pipeline
  from sklearn.preprocessing import PolynomialFeatures
  from sklearn.preprocessing import StandardScaler
  from sklearn.linear_model import LinearRegression
  from sklearn.model_selection import train_test_split, GridSearchCV

  import os
  from joblib import dump

  # Load features
  df = session.table(features_table).to_pandas()

  # Preprocess the Numeric columns
  # We apply PolynomialFeatures and StandardScaler preprocessing steps to the numeric columns
  # NOTE: High degrees can cause overfitting.
  numeric_features = ['SEARCH_ENGINE','SOCIAL_MEDIA','VIDEO','EMAIL']
  numeric_transformer = Pipeline(steps=[('poly',PolynomialFeatures(degree = polynomial_features_degrees)),('scaler', StandardScaler())])

  # Combine the preprocessed step together using the Column Transformer module
  preprocessor = ColumnTransformer(
      transformers=[
          ('num', numeric_transformer, numeric_features)])

  # The next step is the integrate the features we just preprocessed with our Machine Learning algorithm to enable us to build a model
  pipeline = Pipeline(steps=[('preprocessor', preprocessor),('classifier', LinearRegression())])
  parameteres = {}

  X = df.drop('REVENUE', axis = 1)
  y = df['REVENUE']

  # Split dataset into training and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)

  # Use GridSearch to find the best fitting model based on number_of_folds folds
  model = GridSearchCV(pipeline, param_grid=parameteres, cv=number_of_folds)

  model.fit(X_train, y_train)
  train_r2_score = model.score(X_train, y_train)
  test_r2_score = model.score(X_test, y_test)

  model_saved = "False"
  if save_model:
    if train_r2_score >= train_accuracy_threshold and test_r2_score >= test_accuracy_threshold:
      # Upload trained model to a stage
      model_output_dir = '/tmp'
      model_file = os.path.join(model_output_dir, 'model.joblib')
      dump(model, model_file)
      session.file.put(model_file,"@dash_models",overwrite=True)
      model_saved = "True"

  # Return model R2 score on train and test data
  return {"R2 score on Train": train_r2_score,
          "R2 threshold on Train": train_accuracy_threshold,
          "R2 score on Test": test_r2_score,
          "R2 threshold on Test": test_accuracy_threshold,
          "Model saved": model_saved}

def create_data_pipeline_ml_tasks(session,resume=False):
  print("Creating root/parent Snowflake Task: data pipeline")
  create_data_pipeline_feature_engineering_task = """
  CREATE OR REPLACE TASK data_pipeline_feature_engineering_task
      WAREHOUSE = 'DASH_L'
      SCHEDULE  = '5 MINUTE'
  AS
      CALL data_pipeline_feature_engineering()
  """
  session.sql(create_data_pipeline_feature_engineering_task).collect()

  print("Creating child/dependent Snowflake Task: model training")
  create_model_training_task = """
  CREATE OR REPLACE TASK model_training_task
      WAREHOUSE = 'DASH_L'
      AFTER data_pipeline_feature_engineering_task
  AS
      CALL train_revenue_prediction_model('MARKETING_BUDGETS_FEATURES',10,2,0.85,0.85,True)
  """
  session.sql(create_model_training_task).collect()

  if resume:
    session.sql("alter task model_training_task resume").collect()
    session.sql("alter task data_pipeline_feature_engineering_task resume").collect()

if __name__ == "__main__":
  session = connect_to_snowflake()
  if session:
    print("Executing data pipeline function to load and transform the data using Snowpark DataFrames...")
    if data_pipeline_feature_engineering(session) == "SUCCESS":
      # Register data pipelining function as a Stored Procedure so it can be run as a task
      print("Registering data pipeline function as a Stored Procedure so it can run as a task on Snowflake...")
      session.sproc.register(
        func=data_pipeline_feature_engineering,
        name="data_pipeline_feature_engineering",
        packages=['snowflake-snowpark-python'],
        is_permanent=True,
        stage_location="@dash_sprocs",
        replace=True)

      # Register model training function as a Stored Procedure
      print("Registering model training function as a Stored Procedure so it can run as a task on Snowflake...")
      session.sproc.register(
        func=train_revenue_prediction_model,
        name="train_revenue_prediction_model",
        packages=['snowflake-snowpark-python','scikit-learn','joblib'],
        is_permanent=True,
        stage_location="@dash_sprocs",
        replace=True)

      print("Executing Stored Procedure to train the model on Snowflake...")
      cross_validaton_folds = 10
      polynomial_features_degrees = 2
      train_accuracy_threshold = 0.85
      test_accuracy_threshold = 0.85
      save_model = True
      ml_training_result = session.call('train_revenue_prediction_model',
                          'MARKETING_BUDGETS_FEATURES',
                          cross_validaton_folds,
                          polynomial_features_degrees,
                          train_accuracy_threshold,
                          test_accuracy_threshold,
                          save_model)

      print(ml_training_result)
      if json.loads(ml_training_result)["Model saved"] == "True":
        print("Creating Snowflake Tasks to run data pipeline and model training Snowpark stored procedures to run on Snowflake on a set schedule...")
        create_data_pipeline_ml_tasks(session,resume=True)

        # Suspend tasks to avoid resource utilization 
        session.sql("alter task data_pipeline_feature_engineering_task suspend").collect()
        session.sql("alter task model_training_task suspend").collect()

      print("DONE")
    else:
      print("An error has occurred in function data_pipeline_feature_engineering().")
  else:
    print("Unable to connect to Snowflake. Please check your credentials and try again.")
