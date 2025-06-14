import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def preprocess_wine_data(file_path):
    """
    Fungsi ini memuat data dari file_path, membersihkan data,
    mendeteksi dan menghapus outlier, serta menstandarkan fitur numerik.

    Args:
        file_path (str): Path ke file CSV.

    Returns:
        pd.DataFrame: DataFrame yang telah diproses dan siap digunakan untuk pelatihan model.
    """
    # 1. Load dataset
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}. Cek struktur folder dan nama file.")

    df = pd.read_csv(file_path)

    # 2. Drop missing values
    df.dropna(inplace=True)

    # 3. Drop duplicates
    df.drop_duplicates(inplace=True)

    # 4. Outlier detection using IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # 5. Feature scaling
    X = df.drop('quality', axis=1)
    y = df['quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Combine scaled features with target
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled['quality'] = y.values

    return df_scaled

if __name__ == "__main__":
    input_path = "winequality-red/winequality-red.csv" 
    output_path = "preprocessing/winequality_red_preprocessed.csv"

    df_clean = preprocess_wine_data(input_path)

    # Buat folder jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print(f"Preprocessing selesai. File disimpan di: {output_path}")
