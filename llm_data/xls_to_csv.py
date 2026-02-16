import os
import pandas as pd
from pathlib import Path

def convert_all_xls_to_csv(source_folder):
   
    folder = Path(source_folder)
    
    if not folder.exists():
        print(f"Error: The folder '{folder}' does not exist.")
        return

    # Get a list of all .xls files
    xls_files = list(folder.glob("*.xlsx"))
    
    if not xls_files:
        print("No .xls files found in this folder.")
        return

    print(f"Found {len(xls_files)} files. Starting conversion...\n")

    for file_path in xls_files:
        try:
            # Construct the new CSV filename (e.g., file.xls -> file.csv)
            csv_path = file_path.with_suffix('.csv')
            
            # Read the Excel file
            # header=None keeps the structure exactly as is (no assumptions about headers)
            df = pd.read_excel(file_path, header=None)
            
            # Save as CSV
            df.to_csv(csv_path, index=False, header=False, encoding='utf-8-sig')
            
            print(f"[OK] Converted: {file_path.name} -> {csv_path.name}")
            
        except Exception as e:
            print(f"[FAILED] Could not convert {file_path.name}. Reason: {e}")

    print("\nBatch conversion complete!")

# --- UPDATE THIS PATH ---
# Use 'r' before the string to handle backslashes in Windows paths safely
folder_to_convert1 = r"C:\Users\Slajmi\Documents\TAMU_Research\llm_data\exam1"
folder_to_convert2 = r"C:\Users\Slajmi\Documents\TAMU_Research\llm_data\exam2"
folder_to_convert3 = r"C:\Users\Slajmi\Documents\TAMU_Research\llm_data\final"

convert_all_xls_to_csv(folder_to_convert1)
convert_all_xls_to_csv(folder_to_convert2)
convert_all_xls_to_csv(folder_to_convert3)
