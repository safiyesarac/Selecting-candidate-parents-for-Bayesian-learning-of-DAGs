import data_io
import heuristics

parsed_scores = data_io.parse_gobnilp_jkl('/home/gulce/Downloads/thesis/data/sachs/sachs_scores.jkl')
scores = heuristics.GobnilpScores(parsed_scores)

import sumu
from sumu.candidates import candidate_parent_algorithm as cpa
cpa.keys()
