import bnlearn as bn
   
import pandas as pd
from pgmpy.readwrite import BIFReader

def sample_datapoints_from_model(model_file, n,csv_file, dat_file):


    
    model = bn.import_DAG(model_file)

    model_data = bn.sampling(model, n=n)
    df=pd.DataFrame(model_data)
    columns = [str(i) for i in range(len(df.columns))]
    print(columns)
    data =[df[col].nunique() for col in df.columns]
    data=[data]
    
    
    df_arity = pd.DataFrame(data, columns=columns)

    column_mapping = {old: new for old, new in zip(df.columns, df_arity.columns)}
    df = df.rename(columns=column_mapping)

    
    df_combined_correct = pd.concat([df_arity, df], ignore_index=True)
    df_combined_correct.to_csv(csv_file, index=False)


    
    df_combined_correct = pd.concat([df_arity, df])
    
    
    
    df_combined = pd.concat([df_arity, df], ignore_index=True)

    
    df_combined.to_csv(dat_file, index=False, sep=' ')


def save_data(df,csv_file, dat_file):
    columns = [str(i) for i in range(len(df.columns))]
    print(columns)
    data =[df[col].nunique() for col in df.columns]
    data=[data]
    
    
    df_arity = pd.DataFrame(data, columns=columns)

    column_mapping = {old: new for old, new in zip(df.columns, df_arity.columns)}
    df = df.rename(columns=column_mapping)

    
    df_combined_correct = pd.concat([df_arity, df], ignore_index=True)
    df_combined_correct.to_csv(csv_file, index=False)


    
    df_combined_correct = pd.concat([df_arity, df])
    
    
    
    df_combined = pd.concat([df_arity, df], ignore_index=True)

    
    df_combined.to_csv(dat_file, index=False, sep=' ')    
    
def compute_bdeu_scores(dat_file,jkl_file):
    import subprocess

    
    command = [
        "python3",
        "/home/gulce/Downloads/thesis/pygobnilp-1.0/rungobnilp.py",
       dat_file,
        "--output_scores", jkl_file,
        "--score", "BDeu",
        "--nopruning",
        "--end", "local scores"
    ]

    
    output_file = "command_output.txt"

    
    with open(output_file, "w") as file:
        try:
            result = subprocess.run(command, check=True, text=True, stdout=file, stderr=subprocess.PIPE)
            print("Command executed successfully! Output written to", output_file)
        except subprocess.CalledProcessError as e:
            print("An error occurred while executing the command.")
            print("Error message:", e.stderr)
sample_datapoints_from_model("data/sachs/sachs.bif",1000000, "data/sachs/sachs_10000.csv","data/sachs/sachs_1000000.dat")
compute_bdeu_scores("data/sachs/sachs_10000.dat","data/sachs/sachs_10000.jkl")


