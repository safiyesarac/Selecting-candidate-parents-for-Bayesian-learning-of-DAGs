import subprocess
def sample_from_exact_modular_sampler(jkl_file,n):

    # Define the command as a list of arguments
    command = ["./modular-dag-sampling-master/sampler", "nonsymmetric", jkl_file, n]

    # Specify the output file
    output_file = "asia_sampled.txt"

    # Run the command and write its output to the file
    with open(output_file, "w") as file:
        try:
            result = subprocess.run(command, check=True, text=True, stdout=file, stderr=subprocess.PIPE)
            print("Command executed successfully! Output written to", output_file)
        except subprocess.CalledProcessError as e:
            print("An error occurred while executing the command.")
            print("Error message:", e.stderr)

















def sample_from_naive_mcmc_sampler(jkl_file,burn_in,txt_file,n):
    # Define the command and arguments as a list
    cmd = [
        "Rscript",
        "/home/gulce/Downloads/thesis/mcmc_sampler.r",
        jkl_file,
        burn_in,
        txt_file,
        n
    ]

    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print the standard output and error (if any)
    print("Standard Output:")
    print(result.stdout)
    print("Standard Error:")
    print(result.stderr)
