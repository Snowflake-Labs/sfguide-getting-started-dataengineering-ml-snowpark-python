##### Load this file in Snowsight as a Python Worksheet. To learn more see https://docs.snowflake.com/en/user-guide/ui-snowsight-worksheets-gs

##### PREREQUISITE
# Create tables, stages, and load data by running SQL statements in https://github.com/Snowflake-Labs/sfguide-getting-started-dataengineering-ml-snowpark-python/blob/main/setup.sql
#####

# Import Snowpark for Python
import snowflake.snowpark as snowpark
from snowflake.snowpark.types import Variant
from snowflake.snowpark.functions import udf,sum,col,array_construct,month,year,call_udf,lit

def main(session: snowpark.Session): 
    ### Features and Target
    # At this point we are ready to perform the following actions to save features and target for model training.
    # Here we will
        # Delete rows with missing values
        # Exclude columns we don't need for modeling
        # Save features into a Snowflake table called MARKETING_BUDGETS_FEATURES

    # Load data
    snow_df_spend_and_revenue_per_month = session.table('spend_and_revenue_per_month')

    # Delete rows with missing values
    snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.dropna()
    
    # Exclude columns we don't need for modeling
    snow_df_spend_and_revenue_per_month = snow_df_spend_and_revenue_per_month.drop(['YEAR','MONTH'])
    
    # Save features into a Snowflake table call MARKETING_BUDGETS_FEATURES
    snow_df_spend_and_revenue_per_month.write.mode('overwrite').save_as_table('MARKETING_BUDGETS_FEATURES')

    print("Features And Target")
    snow_df_spend_and_revenue_per_month.show()
    
    ### Model Training in Snowflake 
    # Let's create a Python function that uses **scikit-learn and other packages which are already included in** [Snowflake Anaconda channel](https://repo.anaconda.com/pkgs/snowflake/) and therefore available on the server-side when executing the Python function as a Stored Procedure running in Snowflake.
    # This function takes the following as parameters:
        # session: Snowflake Session object.
        # features_table: Name of the table that holds the features and target variable.
        # number_of_folds: Number of cross validation folds used in GridSearchCV.
        # polynomial_features_degress: PolynomialFeatures as a preprocessing step.
        # train_accuracy_threshold: Accuracy thresholds for train dataset. This values is used to determine if the model should be saved.
        # test_accuracy_threshold: Accuracy thresholds for test dataset. This values is used to determine if the model should be saved.
        # save_model: Boolean that determines if the model should be saved provided the accuracy thresholds are met.
    
    #TIP: Learn more about see https://docs.snowflake.com/en/user-guide/warehouses-snowpark-optimized.html

    def train_revenue_prediction_model(
        session: snowpark.Session, 
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
    
        model_saved = False
    
        if save_model:
            if train_r2_score >= train_accuracy_threshold and test_r2_score >= test_accuracy_threshold:
                # Upload trained model to a stage
                model_output_dir = '/tmp'
                model_file = os.path.join(model_output_dir, 'model.joblib')
                dump(model, model_file)
                session.file.put(model_file,"@dash_models",overwrite=True)
                model_saved = True
    
        # Return model R2 score on train and test data
        return {"R2 score on Train": train_r2_score,"R2 threshold on Train": train_accuracy_threshold,"R2 score on Test": test_r2_score,"R2 threshold on Test": test_accuracy_threshold,"Model saved":model_saved}

    ### Create Stored Procedure to deploy model training code on Snowflake
    # Assuming the testing is complete and we're satisfied with the model, let's register the model training Python function as a Snowpark Python Stored Procedure by supplying the packages (snowflake-snowpark-python,scikit-learn, and joblib) it will need and use during execution.
    #TIP: Learn more about Snowpark Python Stored Procedures https://docs.snowflake.com/en/sql-reference/stored-procedures-python.html

    session.sproc.register(
        func=train_revenue_prediction_model,
        name="train_revenue_prediction_model",
        packages=['snowflake-snowpark-python','scikit-learn','joblib'],
        is_permanent=True,
        stage_location="@dash_sprocs",
        replace=True)

    ### Execute Stored Procedure to train model and deploy it on Snowflake
    # Now we're ready to train the model and save it onto a Snowflake stage so let's set save_model = True and run/execute the Stored Procedure using session.call() function.
    cross_validaton_folds = 10
    polynomial_features_degrees = 2
    train_accuracy_threshold = 0.85
    test_accuracy_threshold = 0.85
    save_model = True

    print("Execute Stored Procedure to train model and deploy it on Snowflake")
    print(session.call('train_revenue_prediction_model',
                        'MARKETING_BUDGETS_FEATURES',
                        cross_validaton_folds,
                        polynomial_features_degrees,
                        train_accuracy_threshold,
                        test_accuracy_threshold,
                        save_model))

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

    return test_df