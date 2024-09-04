import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import pearsonr


def describe_df(df:pd.DataFrame) -> pd.DataFrame:
    """
    Generates a summary DataFrame that provides detailed information about 
    each column in the input DataFrame. The summary includes data types, 
    percentage of missing values, number of unique values, and the cardinality 
    of each column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be described.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame with the following rows:
        - 'DATA_TYPE': Data type of each column.
        - 'MISSING (%)': Percentage of missing values in each column.
        - 'UNIQUE_VALUES': Number of unique values in each column.
        - 'CARDIN (%)': Cardinality percentage of each column (unique values / total rows).

        The returned DataFrame uses the column names from the input DataFrame.

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame.

    Notes
    -----
    - The function assumes that all columns in the input DataFrame are either 
      numeric, boolean, or categorical (objects).
    - The output DataFrame is transposed and rounded to two decimal places for 
      better readability.
    - The cardinality is a measure of the uniqueness of the data in a column 
      relative to the number of rows.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'A': [1, 2, 2, 3],
    >>>     'B': ['a', 'b', 'a', 'b'],
    >>>     'C': [True, False, True, True]
    >>> })
    >>> df_out = describe_dataframe(df)
    >>> print(df_out)
    
           DATA_TYPE  MISSING (%)  UNIQUE_VALUES  CARDIN (%)
    COL_N                                                  
    A          int64          0.0              3        75.0
    B         object          0.0              2        50.0
    C           bool          0.0              2        50.0
    """
    
    # Verificar que el argumento es un DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
        # print("El argumento proporcionado no es un DataFrame.")
        # return None
    
    data = {
        'COL_N': df.columns,
        'DATA_TYPE': df.dtypes.values,
        'MISSING (%)': df.isnull().mean().values * 100,
        'UNIQUE_VALUES': df.nunique().values,
        'CARDIN (%)': (df.nunique() / len(df)).values * 100
    }
    
    # Crear el DataFrame final
    df_out = pd.DataFrame(data)
    
    # Establecer la columna 'COL_N' como índice
    df_out.set_index('COL_N', inplace=True)
    
    return df_out.round(2).T


# No tiene control de errores
# NO calcula bien los missings y los unique values
def describe_df_JUANMA(df):
    """
    Describe recibe un pandas dataframe para informar de los tipos de datos, los missing los unique values y la cardinalidad.

    Argumentos:
    df (dataframe): Recibe un objeto de tipo Dataframe de pandas.

    Retorna:
    dataframe: Retorna un dataframe conteniendo como filas los tipos de datos, missing, unique values y cardinalidad
    y como columnas las del dataframe original.
    """

    # Obtenemos el nombre de las columnas
    indices = ['DATA_TYPE','MISSINGS (%)','UNIQUE_VALUES','CARDIN (%)']
    columns = df.columns.values
    # Como es más comodo primero generamos la matriz transpuesta y luego transponemos
    # por eso igualo columns a indices y index a columnas, porque será finalmente así.
    df_retorno = pd.DataFrame(columns = indices, index = columns)
    
    # OBTENEMOS LOS TIPOS
    tipos = np.array(df.dtypes)
    # los guardo en la matriz de retorno
    df_retorno['DATA_TYPE']=tipos

    # OBTENEMOS LOS MISSINGS
    for col in df:
        missing = df[col].isnull().sum()+df[col].isna().sum()
        df_retorno.loc[col,'MISSINGS (%)']=round(100*missing/len(df),2)

    # OBTENEMOS LOS UNIQUE_VALUES Y LA CARDINALIDAD
    for col in df:
        unique = len(df[col].unique())
        df_retorno.loc[col,'UNIQUE_VALUES']=unique
        df_retorno.loc[col,'CARDIN (%)']=round(100*unique/len(df),2)

    # PONEMOS EL NOMBRE COL_N
    df_retorno.index.name = "COL_N"

    # DEVOLVEMOS LA MATRIZ
    return df_retorno.T



def typify_variables(df:pd.DataFrame, *, umbral_categoria=10, umbral_continua=30) -> pd.DataFrame:
    """
    Suggests the type of each column in the input DataFrame based on cardinality and thresholds.

    This function takes a pandas DataFrame and provides a suggestion for each column's type 
    based on its cardinality relative to the provided thresholds.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be analyzed.

    umbral_categoria : int
        The threshold for distinguishing between categorical and numerical variables.
        If a column's cardinality is greater than or equal to this threshold, further checks are performed.

    umbral_continua : float
        The threshold for distinguishing between continuous and discrete numerical variables.
        If a column's cardinality percentage is greater than or equal to this threshold, the column is suggested 
        as "Numerica Continua". Otherwise, it is suggested as "Numerica Discreta".

    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns:
        - 'nombre_variable': The name of the column from the input DataFrame.
        - 'tipo_sugerido': Suggested type of the column, which can be "Binaria", "Categórica", 
          "Numerica Continua", or "Numerica Discreta".

    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame or the thresholds are not of the correct type.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 2, None],
    ...     'B': ['x', 'y', 'x', 'x'],
    ...     'C': [10.0, 20.0, 20.0, 30.0]
    ... })
    >>> tipifica_variables(df, umbral_categoria=3, umbral_continua=50.0)
      nombre_variable          tipo_sugerido
    0                A       Binaria
    1                B       Categórica
    2                C  Numerica Continua
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame")
    if not isinstance(umbral_categoria, int) or not isinstance(umbral_continua, (int, float)):
        raise TypeError("Thresholds must be an integer and a float, respectively")
    # SE PUEDE COMPROBAR SI EL DATA FRAME TIENE COLUMNAS O ESTÁ VACÍO

    # Calculate cardinality and percentage of unique values
    cardinality = df.nunique()
    cardinality_percentage = (cardinality / len(df)) * 100

    # Determine the suggested type for each column
    tipo_sugerido = []
    for col in df.columns:
        unique_count = cardinality[col]
        if unique_count == 2:
            tipo_sugerido.append("Binaria")
        elif unique_count < umbral_categoria:
            tipo_sugerido.append("Categórica")
        else:
            if cardinality_percentage[col] >= umbral_continua:
                tipo_sugerido.append("Numérica Continua")
            else:
                tipo_sugerido.append("Numérica Discreta")

    # Create the output DataFrame
    df_out = pd.DataFrame({
        'nombre_variable': df.columns,
        'tipo_sugerido': tipo_sugerido
        })

    return df_out



def typify_variables_JUANMA(df, umbral_categoria, umbral_continua):
    '''
    Describe recibe un pandas dataframe, un entero con el umbral para asignar un tipo
    de variable como "Categórica". Recibe tambien un umbral para continua. En caso de superar
    el umbral_categórica si %cardinalidad > umbral asignara el tipo "Numérica Continua"
    en caso contrario asignará "Numerica Discreta".

    Argumentos:
    df (dataframe): Recibe un objeto de tipo Dataframe de pandas.
    umbral_categoria: Recibe el umbral para asignar como categorica(por debajo). Asigna "Binaria" si la cardinalidad es 2.
    umbral_continua: Recibe el umbral para distinguir entre discreta(por debajo o igual) y continua(por encima)

    Retorna:
    dataframe: Retorna un dataframe conteniendo dos columnas: los nombres y el tipo sugerido
    '''

    # Obtenemos el nombre de las columnas
    indices = df.columns.values 
    columnas = ['TIPO_SUGERIDO']
    # Como es más comodo primero generamos la matriz transpuesta y luego transponemos
    # por eso igualo columns a indices y index a columnas, porque será finalmente así.
    df_retorno = pd.DataFrame(index = indices, columns = columnas)
    
    # OBTENEMOS LA CARDINALIDAD Y SUGERIMOS EL TIPO DE VARIABLE
    for col in df:
        unique = len(df[col].unique())
        if unique == 2:
            tipo_sugerido = "BINARIO"
        elif (unique < umbral_categoria):
            tipo_sugerido = "CATEGORICA"
        elif (unique < umbral_continua):
            tipo_sugerido = "NUMERICA DISCRETA"
        else:
            tipo_sugerido = "NUMERICA CONTINUA"
        df_retorno.loc[col]=tipo_sugerido

    df_retorno.index.name='COL_N'
    
    # DEVOLVEMOS LA MATRIZ
    return df_retorno



def prueba():
    print('Sí funciono.')



def get_features_num_regression(df:pd.DataFrame, target_col, umbral_corr, pvalue=None) -> list:
    """
    Obtiene las columnas numéricas de un DataFrame cuya correlación con la columna objetivo 
    supera un umbral especificado. Además, permite filtrar las columnas en función 
    de la significancia estadística de la correlación, mediante un valor-p opcional.

    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame que contiene los datos a analizar.

    target_col : str
        Nombre de la columna objetivo que se desea predecir; debe ser 
        una variable numérica continua o discreta con alta cardinalidad.

    umbral_corr : float
        Umbral de correlación absoluta para considerar una relación 
        significativa entre las columnas (debe estar comprendido entre 0 y 1).

    pvalue : float (opcional)
        Valor-p que determina el nivel de significancia para 
        filtrar las columnas. Si se proporciona, solo se incluirán 
        las columnas cuya correlación supere el umbral y cuyo 
        valor-p sea mayor o igual a 1 - pvalue. Debe estar comprendido entre 0 y 1.

    Retorna:
    --------
    lista_num : list
        Lista de nombres de columnas numéricas que cumplen con los criterios establecidos.
        Si no hay columnas que cumplan los requisitos, se devuelve una lista vacía.
        Si algún argumento no es válido, se devuelve None.

    Excepciones:
    -----------
    La función imprime mensajes de error en los siguientes casos:
    - Si `df` no es un DataFrame.
    - Si `target_col` no es una columna del DataFrame.
    - Si `target_col` no es una variable numérica continua o es discreta con baja cardinalidad.
    - Si `umbral_corr` no es un número entre 0 y 1.
    - Si `pvalue` no es None y no es un número entre 0 y 1.

    En cualquiera de estos casos, la función retorna `None`.
    """
    # Comprobaciones iniciales de los argumentos
    if not isinstance(df, pd.DataFrame):
        print(f"Error: No se ha introducido un DataFrame válido.")
        return None
    
    if target_col not in df.columns:
        print(f"Error: {target_col} no es una columna del DataFrame.")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: {target_col} no es una columna numérica.")
        return None

    # Verificación adicional para alta cardinalidad en caso de ser discreta
    if df[target_col].dtype == 'int' and df[target_col].nunique() < 10:
        print(f"Error: {target_col} es una columna discreta con baja cardinalidad.")
        return None

    if not isinstance(umbral_corr, (int, float)) or umbral_corr < 0 or umbral_corr > 1:
        print(f"Error: {umbral_corr} no es un valor válido para 'umbral_corr'. Debe estar entre 0 y 1.")
        return None

    if pvalue is not None and (not isinstance(pvalue, (int, float)) or pvalue < 0 or pvalue > 1):
        print(f"Error: {pvalue} no es un valor válido para 'pvalue'. Debe estar entre 0 y 1.")
        return None

    lista_num = []
    
    for columna in df.columns:
        if pd.api.types.is_numeric_dtype(df[columna]) and columna != target_col:
            resultado_test = pearsonr(df[columna], df[target_col], alternative='less')
            correlacion = resultado_test[0]
            p_valor = resultado_test[1]
            
            if abs(correlacion) > umbral_corr:
                if pvalue is None or p_valor >= 1 - pvalue:
                    lista_num.append(columna)
    
    return lista_num



# puede tener un test de significancia entre numéricas y numéricas
# FALTA COMPROBAR CARDINALIDAD
# No retorna None tras las excepciones
# No hace falta lista_num.remove(target_col) si se filtra en el primer if
def get_features_num_regression_LUIS(df, target_col, umbral_corr, pvalue=None):
    """
    Obtiene columnas numéricas del DataFrame cuya correlación con la columna objetivo 
    supera un umbral especificado. Además, permite filtrar las columnas en función 
    de la significancia estadística de la correlación mediante un valor p.

    Parámetros:

    df (pd.DataFrame): DataFrame que contiene los datos a analizar.

    target_col (str): Nombre de la columna objetivo que se desea predecir; debe ser 
                      una variable numérica continua o discreta con alta cardinalidad.

    umbral_corr (float): Umbral de correlación absoluto para considerar una relación 
                         significativa entre las columnas (debe estar entre 0 y 1).

    pvalue (float, opcional): Valor p que determina el nivel de significancia para 
                              filtrar las columnas. Si se proporciona, solo se incluirán 
                              las columnas cuya correlación supere el umbral y cuyo 
                              valor p sea mayor o igual a 1 - pvalue. Debe estar entre 0 y 1.

    Retorna:

    lista_num: Lista de nombres de columnas numéricas que cumplen con los criterios establecidos.
               Si no hay columnas que cumplan los requisitos, se devuelve una lista vacía.

    Excepciones:

    Imprime mensajes de error si alguno de los argumentos no es válido o si hay problemas
    con los tipos de datos.
    """

    if not isinstance(df, pd.DataFrame):
        print(f"{df} no es un argumento válido. Chequea que sea un DataFrame.")

    
    if target_col not in df.columns:
        print(f"{target_col} no es una columna del DataFrame.")

    if umbral_corr > 1 or umbral_corr < 0 or not isinstance(umbral_corr, (int,float)):
        print(f"{umbral_corr} no es un valor válido para 'umbral_corr', debe de estar comprendido entre 0 y 1.")

    if pvalue != None and  (pvalue > 1 or pvalue < 0 or not isinstance(umbral_corr, (int,float))):
        print(f"{pvalue} no es un valor válido para 'pvalue', debe deestar comprendido entre 0 y 1.")


    lista_num = []
    for columna in df.columns:
        if pd.api.types.is_numeric_dtype(df[columna]):
            resultado_test = pearsonr(df[columna], df[target_col], alternative= "less")
            if pvalue == None:
                if abs(resultado_test[0]) > umbral_corr:
                    lista_num.append(columna)
            else:
               if abs(resultado_test[0]) > umbral_corr:
                   if resultado_test[1] >= 1-pvalue:
                       lista_num.append(columna)
                    
    lista_num.remove(target_col)
    
    return lista_num



def plot_features_num_regression():
    '''puede tener un test para la correlación'''
    pass


def get_features_cat_regression():
    '''puede tener dos tests distintos (dependiendo de la cardinalidad se calcularía uno u otro)'''
    pass


def plot_features_cat_regression():
    '''puede usar internamente get_features_cat_regression'''
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
