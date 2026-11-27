# Baseline multimodale HABS V0

Régressions Ridge par essai pour l’activation déclarée (`rating_arousal`, cible principale) et la valence (`rating_valence`). Le code compare l’EEG, l’ECG, la conductance cutanée et la respiration, seuls et combinés, sur des plis groupés par identifiant de participant. [Guide de transfert V1](../HABS_V1_TRANSFER.md)

## Exécuter depuis la racine du dépôt

Installer les dépendances avec `python -m pip install -r requirements.txt` dans l’environnement Python du projet. Depuis la racine du dépôt, `--data` utilise `./data` par défaut. Pour un autre emplacement, indiquer le dossier qui contient `PREPROCESSED/` et `labels_train.csv`. Le programme utilise `markers.csv` à cette racine s’il existe, sinon la copie des notebooks. Les chemins absolus vers un disque externe sont acceptés.

```sh
python -m baseline_multimodale --data sample --out out/baseline_multimodale_sample --smoke
python -m baseline_multimodale --data "/Volumes/Nom du disque/HABS" --out out/baseline_multimodale
python -m unittest -v baseline_multimodale.test_baseline
```

Le terminal affiche les étapes au fil du calcul, du chargement des annotations à l’écriture des résultats. Pendant l’extraction, il indique le nombre de BDF traités et de fenêtres vérifiées. Pendant l’entraînement, il indique la cible et le pli en cours.

Pour utiliser des annotations stockées localement avec les enregistrements sur SSD, passer `--csv-dir ./data_csv`. Ce dossier doit contenir `labels_train.csv` et `markers.csv`.

Le premier appel contrôle les métadonnées de visionnage et mesure les quatre modalités sur les dix premières secondes du BDF de repos de P001. Les résultats du test de lecture sont dans `sample_smoke.json`. L’échantillon fournit 36 essais à inventorier. L’entraînement sur des essais réels demande les BDF de visionnage et les notes. Le test automatisé génère aussi des BDF annotés pour six IDs fictifs, avec un doublon, une discordance temporelle et une note manquante.

Le second appel produit les résultats de l’étude sur les données fournies. La commande rend le code de sortie 2 si des entrées manquent ou si la population principale ne peut pas être évaluée. Les CSV et `config.json` restent disponibles pour diagnostiquer la couverture. L’option répétable `--development-id P001` réserve les IDs inspectés pour ajuster l’extraction. Ils figurent dans le manifeste et sont retirés des plis. Le point d’entrée HABS dans `process.py` utilise V0 dès qu’il détecte un BDF de visionnage prétraité.

## Lire les résultats

| Fichier | Contenu |
| --- | --- |
| `manifest.csv` | Notes, source du jour de session, horodatages des deux sources, fenêtre retenue, statut de l’essai, canaux et éligibilité par cible. |
| `coverage.csv` | Essais et IDs dénombrés par étape, cible et cohorte. |
| `features.csv` | Descripteurs des fenêtres vérifiées, avec une clé par essai. |
| `folds.csv` | Affectation externe commune et rôles utilisés pour chaque cible aux deux niveaux. |
| `predictions.csv` | Prédictions hors pli, modèle, alpha, présence du clip en entraînement et capteur sélectionné le cas échéant. |
| `metrics.csv` | MAE par ID, moyennes et gains appariés avec intervalles bootstrap. |
| `config.json` | État de l’exécution, réglages, empreintes et versions. |

La comparaison principale se lit dans `metrics.csv` pour `population=four_modalities`, `target=rating_arousal`, `subset=all` et `name=best_single_sensor_vs_four_modalities`. Un gain positif indique une baisse d’erreur. Le contrôle `clip_ridge_vs_clip_plus_physiology` mesure l’apport de la physiologie quand l’ID du clip est connu. La ligne `clip_median` sert de référence supplémentaire.

Si le jour est vide dans `labels_train.csv`, la note est récupérée lorsque le participant, les deux marqueurs et le clip désignent un seul essai dont la fenêtre BDF a été vérifiée. La colonne `label_day_source` distingue ce cas. Les partitions utilisent `participant_id` et les cohortes viennent des lignes d’essai. Les contrôles par clip apprennent leurs statistiques sur les seuls plis d’entraînement.

La tolérance de 20 ms s’applique aux débuts et aux fins mesurées. Si le JSON indique une durée nominale (`protocol_csv`), le calcul s’arrête à la plus proche des bornes de fin du protocole, du BDF et du fichier. Une fenêtre trop courte est signalée dans le manifeste. Les critères de qualité restent à valider sur des sessions de visionnage. Si les quatre modalités couvrent moins de la moitié des essais ou des IDs annotés et synchronisés pour l’activation, ou si leurs plis ne permettent pas l’évaluation, V0 tente aussi l’analyse ECG + GSR sur les essais où ces deux capteurs sont exploitables. Les lignes `population=ecg_gsr` décrivent cette population. La sensibilité aux essais signalés recalcule les métriques sur les prédictions des essais dont `flagged_outliers` diffère de 1.

## V1 — encodeurs par capteur

V1 prédit une note par visionnage. L’expérience principale compare la fusion neuronale au Ridge quatre capteurs de V0 sur les mêmes clés, bornes, notes et plis. L’activation est la cible principale. La valence est secondaire. Le dossier V0 doit contenir un résultat évalué. Pour la référence complète, il représente 5 568 essais par cible et 269 IDs.

Depuis la racine, le pilote ECG exécute le premier pli externe d’activation, avec une époque et une configuration. Il vérifie la sélection interne, le réentraînement, le rechargement des poids et l’écriture des prédictions. Adapter `--data` au dossier contenant `PREPROCESSED`.

```sh
python -m baseline_multimodale --data ./data --v0-dir out/baseline_multimodale_full --out out/v1_pilote --cache-dir out/v1_cache --check-ecg
python -m unittest -v baseline_multimodale.test_baseline baseline_multimodale.test_v1
```

Pour la comparaison complète sur une machine CUDA disposant des artefacts V0, utiliser la commande suivante.

```sh
python -m baseline_multimodale --data /data --v0-dir out/baseline_multimodale_full --out out/v1 --cache-dir out/v1_cache --device cuda
```

Sur le GPU HABS, le workflow appelle directement `process(data_dir, out_dir)`. Avec les BDF de visionnage à `/data`, ce point d’entrée produit la référence dans `out_dir/v0` puis lance le test V1 sur CUDA dans `out_dir/v1`. Le test couvre les sept modèles, les deux cibles et tous les plis, avec trois époques au maximum, un taux d’apprentissage de 0,001 et une régularisation de 0,0001. La commande complète ci-dessus utilise la grille de recherche et les 30 époques. Les chemins de données locaux utilisent V0 par défaut. `HABS_BASELINE=v0` sélectionne explicitement V0. `HABS_BASELINE=v1` sélectionne le test V1 pour un autre chemin et `HABS_V0_DIR` permet de fournir une référence existante.

Le lecteur découpe chaque fenêtre dans le BDF à sa fréquence native. Il contrôle la médiane et la pente GSR en µS contre V0 avant de rééchantillonner. Le cache conserve une session par fichier, en µV pour l’EEG, mV pour l’ECG, µS pour la GSR et unités arbitraires pour la respiration. Les fréquences d’entrée des réseaux sont de 250, 250, 25 et 25 Hz. Chaque encodeur produit cinq vecteurs de 32 valeurs sur des intervalles successifs de deux secondes. La normalisation par canal utilise les seuls échantillons valides de l’entraînement du pli. Les nouveaux manifestes V0 exposent `*_trace_status` pour les contrôles de lecture, de signal plat et de saturation. Le champ `*_status` détermine l’éligibilité V0, qui dépend aussi du calcul des descripteurs.

Les sept modèles sont les quatre encodeurs seuls, les trois périphériques, les quatre capteurs et le MLP sur les 16 descripteurs de V0. L’entraînement utilise AdamW et l’erreur absolue avec des lots de 32. La MAE moyenne par ID sur les validations internes sélectionne l’époque (1 à 30), le taux d’apprentissage (0,0003 ou 0,001) et la régularisation (0,0001 ou 0,01). Avec cinq plis externes, trois internes et deux cibles, le protocole représente 840 entraînements internes et 70 réentraînements. Relancer la même commande dans le même dossier reprend les ajustements terminés. La reprise demande de conserver le dossier `checkpoints`.

| Sortie V1 | Lecture |
|---|---|
| `comparison.csv` | Par cible, couverture, exclusions respiratoires, MAE V1, Ridge, médiane et clip seul, gain et intervalle de confiance. Le protocole `ecg_pilot` identifie le test sur un pli. |
| `predictions.csv`, `scores.csv` | Prédictions externes par clé, cible et modèle, comparateurs V0 inclus, puis tableau des MAE. |
| `metrics.csv` | MAE et gains par ID, moyennes et IC à 95 % avec 2 000 rééchantillonnages appariés des IDs, et sensibilité aux essais signalés. |
| `selection.csv`, `checkpoints/` | Choix de chaque pli et modèle externe rechargeable avec `v1.load_model(path)`. Les poids contiennent aussi les statistiques de normalisation, et celles d’imputation pour le MLP. |
| `config.json`, `coverage.csv` | Statut, paramètres, graines, versions, empreintes des données et du code, couverture de la population source. |

Pour l’interprétation, un gain positif représente une réduction d’erreur face à la référence. Un intervalle contenant zéro indique un gain incertain. Lire aussi la comparaison à la médiane et la proportion de clips vus en entraînement. Les cinq vecteurs horodatés de chaque branche sont accessibles avant leur moyenne pour la future comparaison de fusion avec V2.
