import pandas as pd


def describe_df(df) -> pd.DataFrame:
    pass


def typify_variables():
    pass


def get_features_num_regression():
    pass


def plot_features_num_regression():
    pass


def get_features_cat_regression():
    pass


def plot_features_cat_regression():
    pass


# ######################
# OTHER USEFUL FUNCTIONS
# ######################

def get_cardinality(df:pd.DataFrame, threshold_categorical=10, threshold_continuous=30) -> pd.DataFrame:
    '''
    Calculates and returns cardinality statistics for each column in a pandas DataFrame, 
    classifying the columns based on their cardinality.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame for which cardinality statistics will be computed.
    threshold_categorical : int, optional (default=10)
        The threshold used to classify columns as 'Categoric' or 'Numeric - Discrete'. 
        Columns with a number of unique values less than this threshold are classified as 'Categoric'.
    threshold_continuous : int, optional (default=30)
        The threshold percentage used to classify columns as 'Numeric - Continuous'. 
        Columns where the percentage of unique values exceeds this threshold are classified as 'Numeric - Continuous'.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'Card': The number of unique values in each column.
        - '%_Card': The percentage of unique values relative to the total number of rows in each column.
        - 'NaN_Values': The number of missing (NaN) values in each column.
        - 'Type': The data type of each column.
        - 'Class': The classification of each column based on its cardinality.
    '''
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    
    print('pandas.DataFrame shape: ', df.shape)
    
    df_out = pd.DataFrame([df.nunique(), df.nunique()/len(df) * 100, df.isna().sum(), df.dtypes])
    df_out = df_out.T.rename(columns = {0: 'Card', 1: '%_Card', 2: 'NaN_Values', 3: 'Type'})
    
    df_out.loc[df_out['Card'] < threshold_categorical, 'Class'] = 'Categoric'    
    df_out.loc[df_out['Card'] == 2, 'Class'] = 'Binary'
    df_out.loc[df_out['Card'] >= threshold_categorical, 'Class'] ='Numeric - Discrete'
    df_out.loc[df_out['%_Card'] > threshold_continuous, 'Class'] = 'Numeric - Continuous'
    
    return df_out
