def describe_df(df):
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
    tipos = np.array(df_titanic.dtypes)
    # los guardo en la matriz de retorno
    df_retorno['DATA_TYPE']=tipos

    # OBTENEMOS LOS MISSINGS
    for col in df_titanic:
        missing = df_titanic[col].isnull().sum()+df_titanic[col].isna().sum()
        df_retorno.loc[col,'MISSINGS (%)']=round(100*missing/len(df_titanic),2)

    # OBTENEMOS LOS UNIQUE_VALUES Y LA CARDINALIDAD
    for col in df_titanic:
        unique = len(df_titanic[col].unique())
        df_retorno.loc[col,'UNIQUE_VALUES']=unique
        df_retorno.loc[col,'CARDIN (%)']=round(100*unique/len(df_titanic),2)

    # PONEMOS EL NOMBRE COL_N
    df_retorno.index.name = "COL_N"

    # DEVOLVEMOS LA MATRIZ
    return df_retorno.T



def tipifica_variables(df,umbral_categoria,umbral_continua):
    """
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
    """

    # Obtenemos el nombre de las columnas
    indices = df.columns.values 
    columnas = ['TIPO_SUGERIDO']
    # Como es más comodo primero generamos la matriz transpuesta y luego transponemos
    # por eso igualo columns a indices y index a columnas, porque será finalmente así.
    df_retorno = pd.DataFrame(index = indices, columns = columnas)
    
    # OBTENEMOS LA CARDINALIDAD Y SUGERIMOS EL TIPO DE VARIABLE
    for col in df_titanic:
        unique = len(df_titanic[col].unique())
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
