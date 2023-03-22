##### Prerequisite
# Create tables and load data by running SQL statements in https://github.com/Snowflake-Labs/sfguide-getting-started-dataengineering-ml-snowpark-python/blob/main/setup.sql
#####

##### Usage
# Load this file in Snowsight as a Python Worksheet. To learn more see https://docs.snowflake.com/en/user-guide/ui-snowsight-worksheets-gs
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
import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import month,year,col,sum

def main(session: snowpark.Session): 
    # What is a Snowpark DataFrame
    # It represents a lazily-evaluated relational dataset that contains a collection of Row objects with columns defined by a schema (column name and type). Here are some of the ways to load data in a Snowpark DataFrame are:
    # - session.table('table_name')
    # - session.sql("select col1, col2... from tableName")*
    # - session.read.options({"field_delimiter": ",", "skip_header": 1}).schema(user_schema).csv("@mystage/testCSV.csv")*
    # - session.read.parquet("@stageName/path/to/file")*
    # - session.create_dataframe([1,2,3], schema=["col1"])*

    ### Load Aggregated Campaign Spend Data from Snowflake table into Snowpark DataFrame
    # Let's first load the campaign spend data. This table contains ad click data that has been aggregated to show daily spend across digital ad channels including search engines, social media, email and video.
    snow_df_spend = session.table('campaign_spend')

    ### Total Spend per Channel per Month
    # Let's transform the data so we can see total cost per year/month per channel using _group_by()_ and _agg()_ Snowpark DataFrame functions.
    # TIP: For a full list of functions, see https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/_autosummary/snowflake.snowpark.functions.html#module-snowflake.snowpark.functions

    snow_df_spend_per_channel = snow_df_spend.group_by(year('DATE'), month('DATE'),'CHANNEL').agg(sum('TOTAL_COST').as_('TOTAL_COST')).\
    with_column_renamed('"YEAR(DATE)"',"YEAR").with_column_renamed('"MONTH(DATE)"',"MONTH").sort('YEAR','MONTH')

    print("Total Spend per Year and Month For All Channels")
    # See the output of this command in "PY Output" tab below
    snow_df_spend_per_channel.show()

    ### Pivot on Channel
    # Let's further transform the campaign spend data so that each row will represent total cost across all channels per year/month using pivot() and sum() Snowpark DataFrame functions. 
    # This transformation will enable us to join with the revenue table such that we will have our input features and target variable in a single table for model training.
    # TIP: For a full list of functions, see https://docs.snowflake.com/en/developer-guide/snowpark/reference/python/_autosummary/snowflake.snowpark.functions.html#module-snowflake.snowpark.functions

    snow_df_spend_per_month = snow_df_spend_per_channel.pivot('CHANNEL',['search_engine','social_media','video','email']).sum('TOTAL_COST').sort('YEAR','MONTH')
    snow_df_spend_per_month = snow_df_spend_per_month.select(
        col("YEAR"),
        col("MONTH"),
        col("'search_engine'").as_("SEARCH_ENGINE"),
        col("'social_media'").as_("SOCIAL_MEDIA"),
        col("'video'").as_("VIDEO"),
        col("'email'").as_("EMAIL")
    )

    print("Total Spend Across All Channels")
    # See the output of this command in "PY Output" tab below
    snow_df_spend_per_month.show()

    ### Save Transformed Data into Snowflake Table
    # Let's save the transformed data into a Snowflake table *SPEND_PER_MONTH*.

    snow_df_spend_per_month.write.mode('overwrite').save_as_table('SPEND_PER_MONTH')

    ### Total Revenue per Year and Month
    # Now let's load revenue table and transform the data into revenue per year/month using group_by() and agg() functions.

    snow_df_revenue = session.table('monthly_revenue')
    snow_df_revenue_per_month = snow_df_revenue.group_by('YEAR','MONTH').agg(sum('REVENUE')).sort('YEAR','MONTH').with_column_renamed('SUM(REVENUE)','REVENUE')

    print("Total Revenue per Year and Month")
    # See the output of this command in "PY Output" tab below
    snow_df_revenue_per_month.show()

    ### Join Total Spend and Total Revenue per Year and Month Across All Channels
    # Next let's join this revenue data with the transformed campaign spend data so that our input features (i.e. cost per channel) and target variable (i.e. revenue) can be loaded into a single table for model training. 
    snow_df_spend_and_revenue_per_month = snow_df_spend_per_month.join(snow_df_revenue_per_month, ["YEAR","MONTH"])

    print("Total Spend and Revenue per Year and Month Across All Channels")
    # See the output of this command in "PY Output" tab below
    snow_df_spend_and_revenue_per_month.show()

    # Snowpark makes is really convenient to look at the DataFrame query and execution plan using explain() Snowpark DataFrame function.
    # See the output of this command in "PY Output" tab below
    snow_df_spend_and_revenue_per_month.explain()

    ### Save Transformed Data into Snowflake Table
    # Let's save the transformed data into a Snowflake table *SPEND_AND_REVENUE_PER_MONTH*

    snow_df_spend_and_revenue_per_month.write.mode('overwrite').save_as_table('SPEND_AND_REVENUE_PER_MONTH')

    # See the output of this in "Results" tab below
    return snow_df_spend_and_revenue_per_month
