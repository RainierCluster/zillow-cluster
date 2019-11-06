import pandas as pd
import numpy as np 

import env

def get_db_url(db):
    """
    Produces a url from env credentials
    >> Input:
    database
    << Output:
    url
    """
    return f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'

def get_sql_zillow():
    """
    Obtain queries from zillow database for transactions in 2017:
    - Merged all tables on the main properties table
    - Removed properties with not latitude and longitude
    >> Input:
    none
    << Output:
    dataframe
    """
    query = '''
    SELECT * FROM properties_2017
    LEFT JOIN airconditioningtype
	    USING(airconditioningtypeid)
    LEFT JOIN architecturalstyletype
	    USING(architecturalstyletypeid)
    LEFT JOIN buildingclasstype
	    USING(buildingclasstypeid)
    LEFT JOIN heatingorsystemtype
	    USING(heatingorsystemtypeid)
    JOIN predictions_2017
	    USING(parcelid)
    LEFT JOIN propertylandusetype
	    USING(propertylandusetypeid)
    LEFT JOIN storytype
	    USING(storytypeid)
    LEFT JOIN typeconstructiontype
	    USING(typeconstructiontypeid)
    WHERE latitude is not null and longitude is not null
    ''' 
    df = pd.read_sql(query, get_db_url("zillow"))
    return df

def wrangle_zillow(df):
    """
    Filter properties with the following conditions:
    - Transactions from 2017 only
    - Kept only the latest transactions
    - Filtered by single-unit properties
    >> Input:
    none
    << Output:
    wrangled dataframe
    """    
    # keep only 2017 values
    df = df [df.transactiondate.str.startswith("2017")]
    # keep only the most recent transaction date
    df = df.sort_values("transactiondate", ascending=False).drop_duplicates("parcelid")
    # remove all the duplicate id columns brought in from sql joins
    df.drop(columns = ["typeconstructiontypeid","storytypeid", "propertylandusetypeid", "heatingorsystemtypeid", "buildingclasstypeid","architecturalstyletypeid","airconditioningtypeid","id"], inplace=True)
    # keep single family homes and remove unit counts greater than 1
    df = df [df.propertylandusedesc == "Single Family Residential"]
    df = df [(df.unitcnt != 2) & (df.unitcnt != 3)]
    return df