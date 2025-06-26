import pandas as pd

# Load datasets
edstays = pd.read_excel("edstays.xlsx")
vitalsign = pd.read_excel("vitalsign.xlsx")
medrecon = pd.read_excel("medrecon.xlsx")
triage3 = pd.read_csv("triage 3.csv")
pyxis3 = pd.read_csv("pyxis 3.csv")
diagnosis3 = pd.read_csv("diagnosis 3.csv")

# Rename key timestamp columns explicitly
vitalsign.rename(columns={'charttime': 'charttime_vitalsign'}, inplace=True)
medrecon.rename(columns={'charttime': 'charttime_medecron'}, inplace=True)
pyxis3.rename(columns={'charttime': 'charttime_pyxis'}, inplace=True)

# Define date columns per dataset
date_columns = {
    'edstays': ['intime', 'outtime'],
    'vitalsign': ['charttime_vitalsign'],
    'medrecon': ['charttime_medecron'],
    'pyxis3': ['charttime_pyxis']
}

# Data cleaning function
def clean_dataframe(df, date_cols=None, drop_missing_cols=None):
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    if drop_missing_cols:
        df.dropna(subset=drop_missing_cols, inplace=True)
    df.drop_duplicates(inplace=True)
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            if df[col].dtype == 'object':
                df[col].fillna("Unknown", inplace=True)
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna("Unknown", inplace=True)
    return df

# Clean individual datasets
edstays = clean_dataframe(edstays, date_columns.get('edstays'))
vitalsign = clean_dataframe(vitalsign, date_columns.get('vitalsign'), drop_missing_cols=['charttime_vitalsign'])
medrecon = clean_dataframe(medrecon, date_columns.get('medrecon'), drop_missing_cols=['charttime_medecron'])
triage3 = clean_dataframe(triage3)
pyxis3 = clean_dataframe(pyxis3, date_columns.get('pyxis3'), drop_missing_cols=['charttime_pyxis'])
diagnosis3 = clean_dataframe(diagnosis3)

# Safe merge function
def merge_with_suffix(base_df, new_df, on_keys, suffix_label):
    overlapping_columns = set(base_df.columns).intersection(new_df.columns) - set(on_keys)
    new_df = new_df.rename(columns={col: f"{col}_{suffix_label}" for col in overlapping_columns})
    return pd.merge(base_df, new_df, on=on_keys, how='left')

# Merge all datasets
merged = edstays.copy()
merged = merge_with_suffix(merged, vitalsign, ["subject_id", "stay_id"], "vitalsign")
merged = merge_with_suffix(merged, medrecon, ["subject_id", "stay_id"], "medrecon")
merged = merge_with_suffix(merged, triage3, ["subject_id", "stay_id"], "triage")
merged = merge_with_suffix(merged, pyxis3, ["subject_id", "stay_id"], "pyxis")
merged = merge_with_suffix(merged, diagnosis3, ["subject_id", "stay_id"], "diagnosis")

# Ensure datetime columns after merge
date_cols = ['intime', 'outtime', 'charttime_vitalsign', 'charttime_medecron', 'charttime_pyxis']
for col in date_cols:
    if col in merged.columns:
        merged[col] = pd.to_datetime(merged[col], errors='coerce')

# Drop rows missing these key datetimes
merged.dropna(subset=['charttime_vitalsign', 'charttime_medecron', 'charttime_pyxis'], inplace=True)

# Final fill only for non-datetime columns
for col in merged.columns:
    if merged[col].isnull().any():
        if pd.api.types.is_datetime64_any_dtype(merged[col]):
            continue
        if merged[col].dtype == 'object':
            merged[col].fillna("Unknown", inplace=True)
        elif pd.api.types.is_numeric_dtype(merged[col]):
            merged[col].fillna(merged[col].median(), inplace=True)
        else:
            merged[col].fillna("Unknown", inplace=True)

# Round float columns
float_cols = merged.select_dtypes(include=['float'])
for col in float_cols.columns:
    merged[col] = merged[col].round(2)

# Always export CSV (safe)
merged.to_csv("merged_dashboard_ready.csv", index=False, date_format='%Y-%m-%d %H:%M:%S')

# Try Excel export (only if openpyxl or xlsxwriter is installed)
excel_saved = False
for engine in ['openpyxl', 'xlsxwriter']:
    try:
        merged.to_excel("merged_dashboard_ready.xlsx", index=False, engine=engine)
        excel_saved = True
        break
    except ImportError:
        continue
if not excel_saved:
    print("⚠️ Skipped Excel export (requires openpyxl or xlsxwriter).")

# Try Parquet export
try:
    merged.to_parquet("merged_dashboard_ready.parquet", index=False)
except ImportError:
    print("⚠️ Skipped Parquet export (requires pyarrow or fastparquet).")

# Try Feather export
try:
    merged.to_feather("merged_dashboard_ready.feather")
except ImportError:
    print("⚠️ Skipped Feather export (requires pyarrow).")

# Print summary
print("\n📦 Merged Data Types:")
print(merged.dtypes)
print("\n✅ Final cleaned and merged dataset saved as:")
print("- merged_dashboard_ready.csv")
if excel_saved:
    print("- merged_dashboard_ready.xlsx")
print("- merged_dashboard_ready.parquet (if library installed)")
print("- merged_dashboard_ready.feather (if library installed)")
