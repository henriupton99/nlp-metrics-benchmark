# NLP Project ENSAE 2023
Repository for Natural Langage Processing Project (Subject 4)
ENSAE Summer Semester 2023 

## Autors :
Henri UPTON
Robin GUILLOT

## Project Description :

A plethora of applications of natural language processing (NLP) performs text-to-text transformation. Given an input,
these systems are required to produce an output text that is coherent, readable and informative. Due to both high
annotation costs and time, researchers tend to rely on automatic evaluation to compare the outputs of such systems.
Reference-based automatic evaluation relies on comparing a candidate text produced by the NLG system and one or multiple
reference texts (‘gold standard’) created by a human annotator. Generic automatic evaluation of NLG is a huge challenge
as it requires building a metric that evaluates the similarity between a candidate and one or several gold-standard
reference texts. However, the definition of success criteria is task-specific: as an example, evaluation of text
summarization focuses on content, coherence, grammatically, conciseness, and readability, whereas machine translation
focuses on fidelity, fluency and adequacy of the translation and data2text generation consider criteria such as data
coverage, correctness and text structure.

The goal is to benchmark the correlation of existing metrics with human scores. Support repository [here](https://github.com/PierreColombo/nlg_eval_via_simi_measures). Different possible generation tasks top work on : translation , data2text generation , story generation.

## The Data : MLQE (MultiLingual Quality Estimation)

GitHub Link : (https://github.com/facebookresearch/mlqe)

Contains data for the 2020 Quality Estimation Shared Task:
(http://www.statmt.org/wmt20/quality-estimation-task.html)

Each source-target language ($sl-$tl) directory has a `*.tsv` file (training and dev) with the following columns:

1) index: segment id
2) original: original sentence
3) translation: MT output
4) scores: list of DA scores by all annotators - the number of annotators may vary
5) mean: average of DA scores
6) z_scores: list of z-standardized DA scores
7) z_mean: average of z-standardized DA scores
8) model_scores: NMT model score for sentence


## Ressources papers :
[0] A Pseudo-Metric between Probability Distributions based on Depth-Trimmed Regions G Staerman, P Mozharovskyi, P
Colombo, S Clémençon, F d'Alché-Buc

[1] Pierre Colombo, Nathan Noiry, Ekhine Irurozki, Stephan Clemencon What are the best systems? New perspectives on NLP
Benchmarking NeurIPS 2022

[2] Cyril Chhun, Pierre Colombo, Fabian Suchanek, Chloe Clavel Of Human Criteria and Automatic Metrics: A Benchmark of
the Evaluation of Story Generation (oral) COLING 2022

[3] Pierre Colombo, Chloé Clavel and Pablo Piantanida. InfoLM: A New Metric to Evaluate Summarization & Data2Text
Generation. Student Outstanding Paper Award (oral) AAAI 2022

[4] Pierre Colombo, Guillaume Staerman, Chloé Clavel, Pablo Piantanida. Automatic Text Evaluation through the Lens of
Wasserstein Barycenters. (oral) EMNLP 2021

[5] Pierre Colombo, Maxime Peyrard, Nathan Noiry, Robert West, Pablo Piantanida. The Glass Ceiling of Automatic
Evaluation in Natural Language Generation

[6] Hamid Jalalzai, Pierre Colombo , Chloe Clavel, Eric Gaussier, Giovanna Varni, Emmanuel Vignon, and Anne Sabourin.
Heavy-tailed representations, text polarity classification & data augmentation. NeurIPS 2020

[7] Alexandre Garcia,Pierre Colombo, Slim Essid, Florence d’Alché-Buc, and Chloé Clavel. From the token to the review: A
hierarchical multimodal approach to opinion mining. EMNLP 2020



