import subprocess
def sample_from_exact_modular_sampler(jkl_file,n,output_file):

    
    command = ["/home/gulce/Downloads/thesis/modular-dag-sampling-master/sampler", "nonsymmetric", jkl_file, n]

    
    

    
    with open(output_file, "w") as file:
        with open('/home/gulce/Downloads/thesis/data/child/error.log', 'w') as err_file:
            try:
                result = subprocess.run(command, check=True, text=True, stdout=file, stderr=err_file)
                print("Command executed successfully! Output written to", output_file)
            except subprocess.CalledProcessError as e:
                print("An error occurred while executing the command.")
                print("Error message:", e.stderr)
                
                





   


def sample_from_exact_modular_fair_sampler(n,k,m,output_file):

    
    command = ["/home/gulce/Downloads/thesis/modular-dag-sampling-master/sampler", "symmetric",'fair',  str(n),str(k), str(m)]

    
    

    
    with open(output_file, "w") as file:
        with open('/home/gulce/Downloads/thesis/data/barleyfungal/error.log', 'w') as err_file:
            try:
                result = subprocess.run(command, check=True, text=True, stdout=file, stderr=err_file)
                print("Command executed successfully! Output written to", output_file)
            except subprocess.CalledProcessError as e:
                print("An error occurred while executing the command.")
                print("Error message:", e.stderr)


            
    
import heuristics
import data_io
def mcmc_sample_pymc(jkl_file,n,output_file):
    scores=data_io.parse_gobnilp_jkl(jkl_file)
    parsed_scores=heuristics.GobnilpScores(scores)
    import logging
    import numpy as np
    import pymc as pm


    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

        
    import pymc as pm
    import numpy as np

    def build_model(scores_obj, num_nodes):
        with pm.Model() as model:
            parent_choices = {}

            
            for node in range(num_nodes):
                if node in scores_obj.scores:
                    node_scores = scores_obj.scores[node]  
                    
                    jittered_probs = []
                    
                    for parents, score in node_scores.items():  
                        jittered_score = score 
                        jittered_probs.append(jittered_score)

                    
                    jittered_probs = np.clip(jittered_probs, 0, 1)  
                    total_prob = np.sum(jittered_probs)
                    
                    if total_prob == 0:
                        jittered_probs = np.ones_like(jittered_probs) / len(jittered_probs)  
                    else:
                        jittered_probs /= total_prob  

                    
                    parent_choices[node] = pm.Categorical(f'parents_of_{node}', p=jittered_probs, shape=(1,))

        return model


    
    model = build_model(parsed_scores, num_nodes=parsed_scores.n)


    with model:
        
        step = pm.Metropolis()  
        trace = pm.sample(n, tune=1000000, step=step, return_inferencedata=False)




        
        

    
    sampled_parent_sets = {varname: trace.get_values(varname) for varname in model.named_vars}

    def decode_samples(sampled_parent_sets, scores):
        decoded_samples = set()  

        for sample in range(len(sampled_parent_sets[list(sampled_parent_sets.keys())[0]])):
            dag = {}

            for node, choices in sampled_parent_sets.items():
                node_index = int(node.split('_')[-1])
                parent_index = choices[sample][0]  

                
                try:
                    parents_dict = scores.scores[node_index]  
                    parent_sets = list(parents_dict.keys())  
                    parents = parent_sets[parent_index]  
                except (KeyError, IndexError):
                    print(f"Warning: Invalid access for node {node_index} at index {parent_index}")
                    parents = ()

                dag[node_index] = frozenset(parents)

            decoded_samples.add(frozenset((k, frozenset(v)) for k, v in dag.items()))

        return decoded_samples


    decoded_dags = decode_samples(sampled_parent_sets, parsed_scores)



    def write_dags_to_file(dags, filename):
        with open(filename, 'w') as file:
            for dag_frozenset in dags:
                dag = {k: set(v) for k, v in dag_frozenset}
                dag_str = ', '.join(f"{node} <- {{{', '.join(map(str, sorted(parents)))}}}" 
                                    for node, parents in sorted(dag.items()))
                file.write(dag_str + "\n")

    write_dags_to_file(decoded_dags,output_file)
    def write_dags_to_file(dags, filename):
        with open(filename, 'w') as file:
            for dag_frozenset in dags:
                dag = {k: set(v) for k, v in dag_frozenset}
                dag_str = ', '.join(f"{node} <- {{{', '.join(map(str, sorted(parents)))}}}" 
                                    for node, parents in sorted(dag.items()))
                file.write(dag_str + "\n")

    write_dags_to_file(decoded_dags,output_file)


















def sample_from_naive_mcmc_sampler(jkl_file,burn_in,txt_file,n):
    
    cmd = [
        "Rscript",
        "/home/gulce/Downloads/thesis/mcmc_sampler.r",
        jkl_file,
        burn_in,
        txt_file,
        n
    ]

    
    result = subprocess.run(cmd, capture_output=True, text=True)

    
    print("Standard Output:")
    print(result.stdout)
    print("Standard Error:")
    print(result.stderr)
