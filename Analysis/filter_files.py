import argparse
import pandas as pd
import os
import shutil
import sys

def filter_structure_files():
    parser = argparse.ArgumentParser(
        description="Filter PDB/PQR files based on a CSV of UniProt IDs.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "-c", "--csv_file",
        required=True,
        help="Path to the input CSV file. Must contain a 'uniprot_id' column."
    )
    
    parser.add_argument(
        "-i", "--input_folder",
        required=True,
        help="Path to the source folder containing PDB or PQR files to filter."
    )
    
    parser.add_argument(
        "-o", "--output_folder",
        required=True,
        help="Path to the destination folder where matched files will be copied."
    )
    
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {args.csv_file}")
        sys.exit(1)
    
    if 'uniprot_id' not in df.columns:
        print(f"Error: Column 'uniprot_id' not found in {args.csv_file}")
        print(f"Available columns are: {', '.join(df.columns)}")
        sys.exit(1)
        
    uniprot_ids = set(df['uniprot_id'].dropna().astype(str))
    
    if not uniprot_ids:
        print("Error: No UniProt IDs found in the CSV file.")
        sys.exit(1)
        
    print(f"Loaded {len(uniprot_ids)} unique UniProt IDs from {args.csv_file}")

    try:
        os.makedirs(args.output_folder, exist_ok=True)
        print(f"Output will be saved to: {os.path.abspath(args.output_folder)}")
    except OSError as e:
        print(f"Error: Could not create output directory {args.output_folder}. {e}")
        sys.exit(1)

    copied_count = 0
    found_ids = set()
    
    try:
        all_files_in_source = os.listdir(args.input_folder)
    except FileNotFoundError:
        print(f"Error: Input folder not found at {args.input_folder}")
        sys.exit(1)
        
    print(f"\nScanning {len(all_files_in_source)} files in {args.input_folder}...")

    for filename in all_files_in_source:
        base_name, ext = os.path.splitext(filename)
        
        if base_name in uniprot_ids and ext.lower() in ['.pdb', '.pqr']:
            
            source_path = os.path.join(args.input_folder, filename)
            dest_path = os.path.join(args.output_folder, filename)
            
            try:
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                found_ids.add(base_name)
            except Exception as e:
                print(f"Warning: Could not copy {filename}. Error: {e}")

    print(f"\n--- Process Complete ---")
    print(f"Total files copied: {copied_count}")
    
    missing_ids = uniprot_ids - found_ids
    if missing_ids:
        print(f"Could not find matching .pdb/.pqr files for {len(missing_ids)} UniProt IDs (out of {len(uniprot_ids)} total).")
    else:
        print(f"Successfully found and copied files for all {len(uniprot_ids)} UniProt IDs.")

if __name__ == "__main__":
    filter_structure_files()