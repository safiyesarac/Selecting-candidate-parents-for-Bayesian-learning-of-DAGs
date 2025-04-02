# mybnexp/data_io.py

import logging
import pandas as pd
import sumu


def read_csv_as_sumu_data(csv_path, skiprows=None, discrete=True):
    """
    Reads a CSV file into a sumu.Data object.
    
    Args:
        csv_path (str): Path to CSV file.
        skiprows (list[int] or None): Rows to skip in the CSV (e.g. [1] if row2 is metadata).
        discrete (bool): Whether the data is discrete for sumu.
    
    Returns:
        sumu.Data
    """
    df = pd.read_csv(csv_path, skiprows=skiprows)
    data_matrix = df.values
    logging.info(f"Loaded CSV data from {csv_path} with shape {data_matrix.shape}")
    return sumu.Data(data_matrix, discrete=discrete)


def parse_dag_line(line: str) -> dict:
    """
    Given a line like:
        '0 <- {}, 1 <- {3}, 2 <- {3}, 3 <- {}, ...'
    return a dict {0: set(), 1: {3}, 2: {3}, 3: set(), ...}.
    """
    dag = {}
    chunks = line.split("},")
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.endswith("}"):
            chunk += "}"
        if "<-" not in chunk:
            continue

        node_str, parents_str = chunk.split("<-")
        node_str = node_str.strip()
        parents_str = parents_str.strip()

        # Remove outer braces
        if parents_str.startswith("{"):
            parents_str = parents_str[1:]
        if parents_str.endswith("}"):
            parents_str = parents_str[:-1]

        if parents_str.strip():
            parent_list = [p.strip() for p in parents_str.split(",") if p.strip()]
            parents = set(int(p) for p in parent_list)
        else:
            parents = set()

        dag[int(node_str)] = parents
    return dag


def parse_dag_file(dag_file: str) -> list:
    """
    Read each line from dag_file, parse into a DAG dict: {node: set_of_parents}.
    Returns a list of such DAGs.
    """
    all_dags = []
    with open(dag_file, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            dag = parse_dag_line(line)
            if dag:
                all_dags.append(dag)
            else:
                logging.warning(f"Line {line_no} in {dag_file} was empty or invalid.")
                return 
    logging.info(f"Parsed {len(all_dags)} DAGs from {dag_file}")
    return all_dags


def parse_gobnilp_jkl(file_path: str) -> dict:
    """
    Parse a Gobnilp .jkl file.
    Returns a dict of node -> [ (score, (parents...)), (score, (parents...)), ... ].
    """
    scores = {}
    current_node = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 1 and parts[0].isdigit():
                # Possibly a metadata line, skip
                continue
            elif len(parts) == 2 and not line.startswith("-"):
                # Node header line (e.g., "6 64")
                try:
                    current_node = int(parts[0])
                    scores[current_node] = []
                except ValueError:
                    logging.warning(f"Unexpected node header: {line}")
            elif current_node is not None and len(parts) >= 2:
                try:
                    score = float(parts[0])
                    num_parents = int(parts[1])
                    parent_nodes = tuple(map(int, parts[2:])) if num_parents > 0 else ()
                    scores[current_node].append((score, parent_nodes))
                except ValueError:
                    logging.warning(f"Invalid line for score/parents: {line}")
            else:
                logging.warning(f"Unrecognized line: {line}")

    logging.info(f"Parsed Gobnilp .jkl file: {file_path}, found {len(scores)} nodes.")
    return scores
