import bnlearn as bn
   
import pandas as pd
from pgmpy.readwrite import BIFReader

def sample_datapoints_from_model(model_file, n,csv_file, dat_file):


    # Load a predefined Bayesian network structure
    model = bn.import_DAG(model_file)

    model_data = bn.sampling(model, n=n)
    df=pd.DataFrame(model_data)
    columns = [str(i) for i in range(len(df.columns))]
    print(columns)
    data =[df[col].nunique() for col in df.columns]
    data=[data]
    # data = [[2] *( len(columns)+0)]
    # Convert to DataFrame
    df_arity = pd.DataFrame(data, columns=columns)

    column_mapping = {old: new for old, new in zip(df.columns, df_arity.columns)}
    df = df.rename(columns=column_mapping)

    # Now, combine the data as before
    df_combined_correct = pd.concat([df_arity, df], ignore_index=True)
    df_combined_correct.to_csv(csv_file, index=False)


    # Combine the arity information (first row) with the rest of the dataset
    df_combined_correct = pd.concat([df_arity, df])
    # Save the DataFrame to a CSV file for inspection or future use
    #df_arity.to_csv("data/original_hailfinder_dataset.csv", index=False)
    # Append the arity row (df_arity) on top of the data_samples
    df_combined = pd.concat([df_arity, df], ignore_index=True)

    # Save the combined dataset to a CSV file
    df_combined.to_csv(dat_file, index=False, sep=' ')


def save_data(df,csv_file, dat_file):
    columns = [str(i) for i in range(len(df.columns))]
    print(columns)
    data =[df[col].nunique() for col in df.columns]
    data=[data]
    # data = [[2] *( len(columns)+0)]
    # Convert to DataFrame
    df_arity = pd.DataFrame(data, columns=columns)

    column_mapping = {old: new for old, new in zip(df.columns, df_arity.columns)}
    df = df.rename(columns=column_mapping)

    # Now, combine the data as before
    df_combined_correct = pd.concat([df_arity, df], ignore_index=True)
    df_combined_correct.to_csv(csv_file, index=False)


    # Combine the arity information (first row) with the rest of the dataset
    df_combined_correct = pd.concat([df_arity, df])
    # Save the DataFrame to a CSV file for inspection or future use
    #df_arity.to_csv("data/original_hailfinder_dataset.csv", index=False)
    # Append the arity row (df_arity) on top of the data_samples
    df_combined = pd.concat([df_arity, df], ignore_index=True)

    # Save the combined dataset to a CSV file
    df_combined.to_csv(dat_file, index=False, sep=' ')    
    
def compute_bdeu_scores(dat_file,jkl_file):
    import subprocess

    # Define the command as a list
    command = [
        "python3",
        "/home/gulce/Downloads/thesis/pygobnilp-1.0/rungobnilp.py",
       dat_file,
        "--output_scores", jkl_file,
        "--score", "BDeu",
        "--nopruning",# "--palim","1",
        "--end", "local scores"
    ]

    # Specify the output file
    output_file = "command_output.txt"

    # Run the command and write its output to a file
    with open(output_file, "w") as file:
        try:
            result = subprocess.run(command, check=True, text=True, stdout=file, stderr=subprocess.PIPE)
            print("Command executed successfully! Output written to", output_file)
        except subprocess.CalledProcessError as e:
            print("An error occurred while executing the command.")
            print("Error message:", e.stderr)

sample_datapoints_from_model('/home/gulce/Downloads/thesis/data/sachs/sachs_rounded.bif',100000,  '/home/gulce/Downloads/thesis/data/sachs/sachs_rounded_10000.csv','/home/gulce/Downloads/thesis/data/sachs/sachs_rounded_10000.dat')
compute_bdeu_scores( '/home/gulce/Downloads/thesis/data/sachs/sachs_rounded_10000.dat', '/home/gulce/Downloads/thesis/data/sachs/sachs_rounded_10000.jkl')


# sample_datapoints_from_model('/home/gulce/Downloads/thesis/data/synt/synt.bif',1000,  '/home/gulce/Downloads/thesis/data/synt/synt.csv','/home/gulce/Downloads/thesis/data/synt/synt.dat')
# compute_bdeu_scores( '/home/gulce/Downloads/thesis/data/synt/synt.dat', '/home/gulce/Downloads/thesis/data/synt/synt.jkl')