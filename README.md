# Text Similarity - NLP Project ENSAE 2023

## Auteurs
Henri UPTON & Robin GUILLOT

## Description du projet

### Amorce :

La tâche à l'étude est la traduction de texte par modèle NMT (Natural Text Translation). Nous disposons de divers langages dits *source* ($sl$ pour "source langage") et langages dits *cibles* ($tl$ pour "target langage"). L'objectif du modèle de NMT est de traduire un ensemble de phrases du langage source en langage cible le plus qualitativement possible. Les phrases d'entrée sont issues de diverses pages Wikipedia. Voici la liste des couples (sl, tl) accompagnés de leur diminutif (sl-tl) :

- English-German (en-de)
- English-Chinese (en-zh)
- Romanian-English (ro-en)
- Estonian-English (et-en)
- Nepalese-English (ne-en)
- Sinhala-English (si-en)

Ce dernier aspect de qualité de traduction est le point central de notre étude. En effet la tâche principale de nos analyses est de constituer un ensemble de métriques permettant d'évaluer la qualité d'une traduction unique. Le critère le plus important lors de l'évaluation de la qualité de telles métriques est leur corrélation avec le jugement humain. Conceptuellement, pour chaque couple (*sl*,*tl*), nous disposons d'un dataset $D = {R_{i}, {C_{i}, h(C_{i})}}_{i = 1}^{N}$ où pour une observation $i$, $R_{i}$ correspond à la séquence source à traduire, $C_{i}$ la traduction candidate par le modèle de NMT à l'étude, et $h(C_i)$ correspond à l'évaluation de la traduction $C_i$ par un humain. 

### Les données :

La base de données MLQE (MultiLingual Quality Estimation) provient du GitHub *facebookresearch* suivant : https://github.com/facebookresearch/mlqe

Elle a fait l'objet de recherches dans le cadre du concours "2020 Quality Estimation Shared Task" : http://www.statmt.org/wmt20/quality-estimation-task.html

Pour ce faire, chaque traduction unique est accompagnée d'un groupe de scores qui représente des évaluations humaines de la qualité d'une traduction par des experts en traduction. Les scores sont nommés DA Scores (Direct Assessment Score) et représentent un jugement sur une échelle de 0 à 100, 100 étant la note maximale. 

Les données sont organisées de la façon suivante : chaque élément concernant un couple ($sl-$tl) est regroupé dans un dossier compressé au format **.tar.gz**. 
- sl-tl_test.tar.gz : contient les données de dev et de train
- sl-tl.tar.gz : contient les données de test

Pour un dataset de données on dispose des variables suivantes :

1) index: l'identifiant unique de l'observation
2) original: phrase source (dans le langage source $sl$)
3) translation: phrase candidate (dans le langage target $tl$)
4) scores: liste des DA scores de plusieurs experts pour le couple (original, translation) associé
5) mean: moyenne des DA scores associés
6) z_scores: liste des z-standardizé DA scores
7) z_mean: moyenne des z-standardizé DA scores
8) model_scores: NMT model score for sentence (à omettre)

D'autres informations annexes sont disponibles :

`doc_ids` : fichier listant la provenance de chaque phrase source (la page Wikipedia dans laquelle elle provient)

`word-probas` : repertoire contenant :
 
* `word_probas.*.$sl$tl`: log-probabilities from the NMT model for each decoded token including the <eos> token
* `mt.*.$sl$tl`: the actual output of the NMT model before any post-processing, corresponding to the log-probas
 above (the <eos> token is not printed, so the number of log-probabilities equals the number of tokens plus 1)


## Ressources :

### GitHubs :

Facebook Research - MLQE Dataset : https://github.com/facebookresearch/mlqe

Benchmark correlation of existing metrics with human scores :(https://github.com/PierreColombo/nlg_eval_via_simi_measures). Different possible generation tasks top work on : translation , data2text generation , story generation.


### Papiers de recherche :
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



