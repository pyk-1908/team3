# preprocessing.py
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    """Process raw data according to notebook specifications[1]"""
    # Create temporal feature
    df["Quarter_Since_Start"] = (df["Year"] - df["Year"].min()) * 4 + df["Quarter"]
    
    # Drop specified columns
    df = df.drop(columns=[
        "Year", "Quarter", "Members", "Members_Lag",
        "Rate_Lag", "Treatment", "QuarterInt",
        "ACR_next", "treatment", "CATE_XL"
    ])
    
    # Encode categorical variables
    for col in ["Provider", "Regionality"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    
    return df.drop(columns=["ChurnRate"]), df["ChurnRate"]
