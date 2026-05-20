import os
import pandas as pd
import numpy as np
import glob
import joblib
from sklearn.preprocessing import StandardScaler

class SepsisPreprocessor:
    def __init__(self, max_time_steps=72):
        self.max_time_steps = max_time_steps
        self.excluded_cols = ['patient_id', 'sepsislabel', 'iculos']

    def ingest_data(self, folder_path):
        csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
        excel_files = glob.glob(os.path.join(folder_path, "*.xlsx"))
        all_files = csv_files + excel_files
        
        if not all_files: return pd.DataFrame()

        patient_dataframes = []
        for file_path in all_files:
            patient_id = os.path.basename(file_path).split('.')[0]
            df = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
            df.columns = [col.lower() for col in df.columns]
            if 'patient_id' not in df.columns: df.insert(0, 'patient_id', patient_id)
            patient_dataframes.append(df)
        return pd.concat(patient_dataframes, ignore_index=True)

    def process_live_inference(self, file_object, filename, hospital_id):
        if filename.endswith('.csv'): df = pd.read_csv(file_object)
        elif filename.endswith('.xlsx'): df = pd.read_excel(file_object)
        else: raise ValueError("Unsupported file format.")

        df.columns = [col.lower() for col in df.columns]
        if 'patient_id' not in df.columns: df.insert(0, 'patient_id', 'live_inference_patient')
        if 'sepsislabel' not in df.columns: df['sepsislabel'] = 0

        df = self.impute(df)
        df = self.engineer_features(df)
        
        scaler_path = os.path.join("data", hospital_id, f"scaler_{hospital_id}.pkl")
        df = self.transform_with_saved_scaler(df, scaler_path)
        
        X_tensor, _, feature_cols = self.create_tensors(df)
        return X_tensor, feature_cols

    def impute(self, df):
        features = [col for col in df.columns if col not in self.excluded_cols]
        df[features] = df.groupby('patient_id')[features].ffill()
        df[features] = df[features].fillna(df[features].median()).fillna(0)
        return df

    def engineer_features(self, df):
        if 'hr' in df.columns and 'sbp' in df.columns:
            df['shock_index'] = np.where(df['sbp'] > 0, df['hr'] / df['sbp'], 0)
        if 'sbp' in df.columns and 'dbp' in df.columns:
            df['pulse_pressure'] = df['sbp'] - df['dbp']
        for v in ['hr', 'map', 'temp']:
            if v in df.columns:
                df[f'{v}_roll_mean_6h'] = df.groupby('patient_id')[v].transform(lambda x: x.rolling(6, min_periods=1).mean())
                df[f'{v}_roll_std_6h'] = df.groupby('patient_id')[v].transform(lambda x: x.rolling(6, min_periods=1).std()).fillna(0)
        return df

    def transform_with_saved_scaler(self, df, load_path):
        if not os.path.exists(load_path): raise FileNotFoundError(f"Scaler missing at {load_path}")
        feature_cols = [col for col in df.columns if col not in self.excluded_cols]
        scaler = joblib.load(load_path)
        df[feature_cols] = scaler.transform(df[feature_cols])
        return df

    def create_tensors(self, df):
        feature_cols = [col for col in df.columns if col not in self.excluded_cols]
        patient_ids = df['patient_id'].unique()
        X = np.zeros((len(patient_ids), self.max_time_steps, len(feature_cols)), dtype=np.float32)
        y = np.zeros((len(patient_ids),), dtype=np.int8)

        grouped = df.groupby('patient_id')
        for i, pid in enumerate(patient_ids):
            group = grouped.get_group(pid)
            p_feat = group[feature_cols].values[-self.max_time_steps:]
            p_lab = group['sepsislabel'].values[-self.max_time_steps:]
            X[i, -len(p_feat):, :] = p_feat
            y[i] = np.max(p_lab) if len(p_lab) > 0 else 0
        return X, y, feature_cols