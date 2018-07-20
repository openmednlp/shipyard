import pandas as pd
import warnings


def load_xls(
        file_path: str,
        valid_sequences: list,
        sheet='Sheet1',
        usecols='I,S,AC,AD,AG,AL,AN') -> pd.DataFrame:
    """
    Loads data from an Excel file
    :param file_path: path to xls file
    :param valid_sequences: keeps only rows whose Sequenz field contains any of these values
    :param sheet: sheet from where to extract the data
    :param usecols: columns to extract
    :return: returns the DataFrame object
    """
    if type(valid_sequences) == str:
        warnings.warn(
            'Expected list, got string for a list of valid sequences. '
            'Converting to list.'
        )
        valid_sequences = [valid_sequences]

    xls_df = pd.read_excel(
        file_path,
        sheet_name=sheet,
        usecols=usecols
    )
    xls_df = xls_df[
        xls_df['Sequenz'].isin(valid_sequences)
    ]

    search_pattern = '|'.join(
        [
            'tumor',
            'neoplasm',
            'krebs',
            'karzinom',
            'adenom',
            'metasta',
            'malignit'
        ]
    )
    xls_df = xls_df[
        xls_df['R_Diagnose'].str.contains(search_pattern, case=False, regex=True) |
        xls_df['R_Resultat'].str.contains(search_pattern, case=False, regex=True)
    ]

    return xls_df


def assign_key(
        df: pd.DataFrame,
        fields: list,
        key_name='id') -> pd.DataFrame:
    """
    Creates a key string out of composite key elements. All composite keys
    will be concatenated into a single string.
    :param df: input DataFrame object
    :param fields: composite key fields
    :param key_name: the key field name, default is 'id'
    :return: DataFrame object
    """
    composite_key_df = df[fields]
    ids = composite_key_df.apply(lambda x: '_'.join(str(s) for s in x), axis=1)
    if len(composite_key_df) != len(ids.unique()):
        raise ValueError(
            'Could not create a unique composite key.'
            '\nTotal entries: {}'
            '\nUnique composite keys:{}'.format(
                len(composite_key_df),
                len(ids.unique())
            )
        )
    df.loc[:, key_name] = ids
    return df
