import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_wine_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    X = df.drop('quality', axis=1)
    y = df['quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['quality'] = y.values

    return df_scaled

if __name__ == "__main__":
    input_path = "winequality-red.csv"
    output_path = "preprocessing/dataset preprocessed/winequality_red_preprocessed.csv"

    df_clean = preprocess_wine_data(input_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print(f"✅ Preprocessing selesai. File disimpan di: {output_path}")
