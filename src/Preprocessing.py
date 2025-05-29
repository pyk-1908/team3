from sklearn.preprocessing import LabelEncoder

# label encoding non-numeric variables.
def label_encode_non_numeric(df):
    """
    Label encodes all non-numeric columns in the given DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with non-numeric columns label encoded.
    """
    df_encoded = df.copy()
    label_encoders = {}
    for col in df_encoded.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
    return df_encoded, label_encoders


def discretize_columns(df, columns, bins, labels=None, qcut=False):
  pass