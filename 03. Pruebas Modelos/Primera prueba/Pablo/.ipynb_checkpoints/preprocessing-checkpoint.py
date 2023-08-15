def col_adjustments(df, cols_to_clean = None, new_order = None, dic = None, lower = False):   
    # None por defecto para que no de error si no queremos usar alguno de estos parametros
    # le pasamos un dataframe, una lista de columnas a limpiar, una lista con el orden en que queremos que esten, un diccionario donde las claves son las columnas que queremos cambiar de tipo y los valores los nuevos tipos que queremos, y si queremos que convierta todo a minúsculas se lo indicamos
    """Ajustes en las columnas de un DataFrame para conseguir el formato deseado"""
    
    if cols_to_clean != None:
        for col in cols_to_clean:
            if lower == True:
                df[col] = df[col].str.findall(r'[«\"\'¡¿\(]?\w+[\)\"\'.,;:…?»!]*|[\w,]+').str.join(' ').str.lower()
            df[col] = df[col].str.findall(r'[«\"\'¡¿\(]?\w+[\)\"\'.,;:…?»!]*|[\w,]+').str.join(' ')

            
            # toma solo los caracteres que nos interesan: palabras, signos de puntuación, paréntesis, etc. (quizás incluso podríamos descartar algunos de estos), y pasamos a minusculas para simplificar
            
    if new_order != None:
        df = df.reindex(columns = new_order)
        # reordenacion de las columnas para que cuadren todos los dataframes y poder concatenarlos verticalmente
        # al añadir nombres de columnas nuevas, se crean automaticamente columnas de NaNs
    
    if dic != None:
        for col, typ in dic.items():
            df[col] = df[col].astype(typ)
            # para convertir 'Target' a numerica (más eficiente que como string)
            # sería mas simple hacerlo sin funcion puesto que solo hay una columna que interese cambiar de tipo, pero creo que más elegante y formal crear una funcion que pueda devolvernos un dataframe en el formato que queremos aunque hubiera mas columnas a limpiar, distintos ordenes o tipos requeridos en cada caso
        
    return df




def slicer(df, string_to_find, col = 'Cuerpo'):
    
    indices = df[col].str.find(string_to_find)
    df[col] = df.apply(lambda x: x[col][:indices[x.name]], axis = 1)
    
    return df



