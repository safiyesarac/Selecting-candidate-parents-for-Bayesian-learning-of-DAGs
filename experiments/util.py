

import re
import sys
import pandas as pd


def parse_coverage_log(log_file_path):
    """
    Reads a coverage log that has lines of the form:
      Rep=X, ALGORITHM=foo, K=Y
      ...
      Coverage with PRIOR = a, Coverage with GOBNILP = b

    Returns a list of tuples: (replicate, algorithm, K, coverage_prior, coverage_gobnilp).
    """
    pattern_header = re.compile(r'^Rep=(\d+),\s+ALGORITHM=([^,]+),\s+K=(\d+)')
    pattern_cov    = re.compile(r'^Coverage with PRIOR\s*=\s*([\d.]+),\s*Coverage with GOBNILP\s*=\s*([\d.]+)')

    coverage_data = []

    
    
    current_rep = None
    current_algo = None
    current_k = None

    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            
            match_header = pattern_header.match(line)
            if match_header:
                current_rep = int(match_header.group(1))
                current_algo = match_header.group(2).strip()
                current_k = int(match_header.group(3))
                continue

            
            match_cov = pattern_cov.match(line)
            if match_cov:
                cov_prior_str = match_cov.group(1)
                cov_gob_str   = match_cov.group(2)
                
                coverage_prior   = float(cov_prior_str)
                coverage_gobnilp = float(cov_gob_str)

                
                if current_rep is not None and current_algo is not None and current_k is not None:
                    coverage_data.append((
                        current_rep,
                        current_algo,
                        current_k,
                        coverage_prior,
                        coverage_gobnilp
                    ))

    return coverage_data


def main(log_file_path, output_raw_csv, output_means_csv):
    
    rows = parse_coverage_log(log_file_path)

    
    df = pd.DataFrame(
        rows,
        columns=['replicate', 'method', 'K', 'coverage_prior', 'coverage_gobnilp']
    )

    
    df.to_csv(output_raw_csv, index=False)
    print(f"Wrote raw coverage data to {output_raw_csv} (rows={len(df)})")

    
    df_mean = df.groupby(['method', 'K'], as_index=False)[['coverage_prior','coverage_gobnilp']].mean()

    
    df_mean.to_csv(output_means_csv, index=False)
    print(f"Wrote coverage means to {output_means_csv}")



if __name__ == "__main__":
    """
    Usage:
        ./coverage_parser.py <coverage_log_file> <raw_csv> <means_csv>
    Example:
        ./coverage_parser.py coverage_nVars=10_maxInDeg=3_alpha=1.0_smallData=500_refData=10000.log \
                             coverage_raw.csv coverage_means.csv
    """
    if len(sys.argv) != 4:
        print("Usage: ./coverage_parser.py <coverage_log_file> <raw_csv> <means_csv>")
        sys.exit(1)

    log_file = sys.argv[1]
    raw_csv  = sys.argv[2]
    mean_csv = sys.argv[3]
    main(log_file, raw_csv, mean_csv)
