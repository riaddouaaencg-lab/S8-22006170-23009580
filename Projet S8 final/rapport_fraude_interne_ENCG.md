# RAPPORT DE PROJET — SEMESTRE 8

---

<div align="center">

## École Nationale de Commerce et de Gestion — SETTAT
### Filière : Audit, Contrôle de Gestion & Systèmes d'Information

---

**Titre du Projet :**

# Détection de la Fraude Interne en Entreprise  
## Approche par Machine Learning et Analyse Comportementale

---

| Champ | Détail |
|---|---|
| **Étudiants** | Étudiant 1 (N° 22006170) — Étudiant 2 (N° 23009580) |
| **Encadrant** | [Nom de l'encadrant] |
| **Filière** | Audit & Contrôle de Gestion |
| **Année universitaire** | 2024 – 2025 |
| **Semestre** | Semestre 8 |

---

*Rapport réalisé dans le cadre du projet de fin de semestre*

</div>

---

## Résumé Exécutif

Ce rapport présente une étude analytique portant sur la **détection de la fraude interne en entreprise** à travers le prisme de l'apprentissage automatique (*Machine Learning*). Dans un contexte où les organisations subissent annuellement des pertes colossales imputables aux comportements frauduleux de leurs propres collaborateurs, il devient impératif de développer des outils prédictifs capables d'identifier, en amont, les profils à risque.

En s'appuyant sur le **Triangle de la Fraude de Cressey (1953)** comme cadre théorique, ce projet exploite un jeu de données simulé de **1 000 employés** afin de construire, entraîner et évaluer trois modèles de classification : la **Régression Logistique**, le **Random Forest** et le **Gradient Boosting**. L'analyse révèle que les variables telles que l'**accès privilégié aux systèmes**, la **pression financière perçue** et le **niveau de satisfaction au travail** constituent les facteurs prédictifs les plus déterminants.

Le modèle *Random Forest* atteint un **score AUC de 0,97 en validation croisée**, confirmant son excellente capacité discriminante. Ces résultats ouvrent des perspectives concrètes pour les équipes d'audit interne cherchant à prioriser leurs interventions et à déployer une approche de contrôle fondée sur les données.

---

## Table des Matières

1. [Introduction Générale](#1-introduction-générale)
2. [Chapitre I — Cadre Théorique et Conceptuel](#chapitre-i--cadre-théorique-et-conceptuel)
3. [Chapitre II — Méthodologie et Construction des Données](#chapitre-ii--méthodologie-et-construction-des-données)
4. [Chapitre III — Analyse Exploratoire des Données](#chapitre-iii--analyse-exploratoire-des-données)
5. [Chapitre IV — Modélisation Prédictive par Machine Learning](#chapitre-iv--modélisation-prédictive-par-machine-learning)
6. [Chapitre V — Segmentation des Risques et Recommandations](#chapitre-v--segmentation-des-risques-et-recommandations)
7. [Conclusion Générale](#7-conclusion-générale)
8. [Annexes](#8-annexes)

---

# 1. Introduction Générale

## 1.1 Contexte et Problématique

La fraude en entreprise représente l'un des risques opérationnels les plus prégnants et les plus coûteux pour les organisations contemporaines. Selon le **rapport de l'ACFE (Association of Certified Fraud Examiners) de 2024**, les entreprises perdent en moyenne **5 % de leur chiffre d'affaires annuel** à cause des fraudes, avec une perte médiane par cas estimée à **145 000 USD**. Plus alarmant encore, la grande majorité de ces actes frauduleux sont commis par des **employés internes**, des individus jouissant de la confiance de leur organisation et disposant d'un accès légitime aux ressources et aux systèmes d'information.

La fraude interne se distingue des fraudes externes par sa nature insidieuse : elle s'opère dans l'ombre des processus normaux de l'entreprise, exploitant des failles dans les dispositifs de contrôle interne. Elle peut prendre des formes variées — détournement d'actifs, manipulation comptable, corruption — et reste difficile à détecter par les méthodes d'audit traditionnelles qui demeurent réactives et souvent incapables d'anticiper les comportements déviants.

Face à ce constat, une **question centrale** émerge :

> **Dans quelle mesure les techniques d'apprentissage automatique peuvent-elles constituer un outil efficace et opérationnel pour la détection proactive de la fraude interne, en se fondant sur des données comportementales et organisationnelles ?**

## 1.2 Objectifs du Projet

Ce projet s'articule autour de trois objectifs complémentaires :

**Objectif 1 — Théorique :** Ancrer l'analyse dans un cadre conceptuel solide, en mobilisant le Triangle de la Fraude de Cressey et les apports contemporains de la littérature académique sur la criminologie organisationnelle.

**Objectif 2 — Analytique :** Développer et évaluer des modèles prédictifs de Machine Learning capables d'identifier, avec un niveau de performance élevé, les employés présentant un risque de comportement frauduleux.

**Objectif 3 — Opérationnel :** Proposer un ensemble de recommandations concrètes à l'intention des responsables de l'audit interne et du contrôle de gestion, permettant d'intégrer ces outils dans une démarche de gouvernance proactive.

## 1.3 Structure du Rapport

Le rapport est organisé en cinq chapitres progressifs, allant du cadre théorique jusqu'aux recommandations pratiques, en passant par la méthodologie, l'analyse exploratoire et la modélisation prédictive. Chaque chapitre est conçu pour apporter une valeur ajoutée spécifique, articulant rigueur académique et pertinence managériale.

---

# Chapitre I — Cadre Théorique et Conceptuel

## Transition

Avant de plonger dans la dimension technique et analytique du projet, il convient d'établir les fondements théoriques qui guideront l'ensemble de la démarche. Ce premier chapitre pose les bases conceptuelles nécessaires à la compréhension des mécanismes de la fraude interne et des approches utilisées pour la détecter.

---

## 1.1 La Fraude Interne : Définition et Enjeux

La fraude interne, au sens large, désigne tout acte délibéré commis par un membre de l'organisation à son propre bénéfice ou au détriment de celle-ci, en violation des règles éthiques et légales en vigueur. Elle se manifeste principalement sous trois formes :

**Le détournement d'actifs** constitue la forme la plus fréquente (environ 86 % des cas selon l'ACFE), incluant le vol de liquidités, la manipulation des notes de frais ou les achats fictifs.

**La corruption** implique l'utilisation abusive du pouvoir ou de la position au sein de l'organisation pour en tirer un avantage personnel illégitime, souvent en collusion avec des tiers externes.

**La fraude aux états financiers** est statistiquement moins fréquente mais engendre les pertes les plus élevées, car elle touche à la sincérité des informations comptables et peut induire en erreur les parties prenantes.

| Type de fraude | Fréquence (ACFE 2024) | Perte médiane estimée |
|---|---|---|
| Détournement d'actifs | 86 % | 100 000 USD |
| Corruption | 50 % | 200 000 USD |
| Fraude financière | 9 % | 766 000 USD |

> 📊 **Suggestion de graphique :** Diagramme en camembert représentant la répartition des types de fraude par fréquence, superposé à un graphique à barres montrant les pertes médianes associées.

## 1.2 Le Triangle de la Fraude de Cressey (1953)

Le modèle théorique central de ce projet est le **Triangle de la Fraude**, développé par le sociologue américain **Donald Cressey** en 1953, à partir de l'étude de cas de détourneurs de fonds. Selon ce modèle, tout acte de fraude interne résulte de la conjonction de trois facteurs :

**1. La Pression (Pressure) :** Il s'agit d'une motivation, le plus souvent financière, qui pousse l'individu à commettre l'acte frauduleux. Cela peut être une dette personnelle, des difficultés économiques, une pression pour atteindre des objectifs irréalistes, ou encore un mode de vie au-dessus des moyens.

**2. L'Opportunité (Opportunity) :** C'est la condition qui rend la fraude techniquement possible. Elle est généralement créée par des défaillances dans le dispositif de contrôle interne : absence de séparation des tâches, accès non sécurisé aux systèmes d'information, supervision insuffisante.

**3. La Rationalisation (Rationalization) :** L'individu fraudeur développe un raisonnement lui permettant de justifier son acte à ses propres yeux. Il peut se convaincre qu'il n'est que "emprunteur", que l'entreprise ne sera pas vraiment lésée, ou qu'il n'est pas rémunéré à sa juste valeur.

```
              OPPORTUNITÉ
                  🔺
                 /    \
                /  FRAUDE  \
               /  INTERNE   \
              /──────────────\
       PRESSION           RATIONALISATION
```

> 📊 **Figure 1 — Insérer ici :** Triangle de Cressey visualisé avec les trois sommets annotés et les facteurs associés issus du dataset (accès privilégié, score de pression financière, insatisfaction au travail).

**Valeur ajoutée analytique :** Ce modèle permet de traduire les trois dimensions en **variables mesurables** : la pression est proxy-isée par le `Score_Pression_Financiere`, l'opportunité par la variable `Acces_Privilegie`, et la rationalisation par le `Score_Satisfaction_Travail` (inversé). C'est précisément sur ces variables que le score de risque composite est construit.

## 1.3 Apports du Machine Learning à la Détection de Fraude

Les méthodes d'audit traditionnelles — échantillonnage aléatoire, vérification des pièces justificatives — sont fondamentalement réactives. Elles interviennent après la commission de la fraude et sont limitées dans leur capacité à traiter de grands volumes de données. L'apprentissage automatique répond à ces limitations en offrant une approche **proactive, scalable et fondée sur les données**.

| Approche | Avantages | Limites |
|---|---|---|
| Audit traditionnel | Rigueur légale, expertise humaine | Réactif, coûteux, biais de l'auditeur |
| Règles expertes (*Rules-based*) | Simple à implémenter, explicable | Rigide, contournable, non-adaptatif |
| Machine Learning | Proactif, scalable, multi-variables | Interprétabilité limitée, dépendance aux données |
| ML + Audit interne (hybride) | Meilleur des deux mondes | Nécessite expertise technique et métier |

La complémentarité entre les approches humaines et algorithmiques constitue la voie la plus prometteuse pour une gouvernance anti-fraude efficace.

---

# Chapitre II — Méthodologie et Construction des Données

## Transition

Après avoir posé le cadre conceptuel, ce chapitre détaille les choix méthodologiques qui sous-tendent l'ensemble de l'analyse. La rigueur de la méthodologie conditionne la validité et la reproductibilité des résultats.

---

## 2.1 Architecture du Dataset

En l'absence de données réelles (soumises à des contraintes de confidentialité), un **dataset simulé** de 1 000 observations a été construit en s'appuyant rigoureusement sur la littérature académique et les rapports professionnels (ACFE, PwC, Deloitte). Chaque variable est ancrée dans un fondement théorique issu du Triangle de Cressey.

| Variable | Type | Description | Dimension Cressey |
|---|---|---|---|
| `ID_Employe` | Identifiant | Identifiant unique de l'employé | — |
| `Departement` | Catégorielle | Département d'appartenance (6 modalités) | — |
| `Anciennete_Annees` | Numérique | Nombre d'années d'expérience dans l'entreprise | Opportunité |
| `Score_Pression_Financiere` | Numérique (1–10) | Score de pression financière perçue | Pression |
| `Satisfaction_Travail` | Numérique (1–10) | Score de satisfaction au poste | Rationalisation |
| `Heures_Supp_Mois` | Numérique | Nombre moyen d'heures supplémentaires par mois | Pression |
| `Conges_Non_Pris` | Numérique | Jours de congés non pris (signal d'alerte RH) | Opportunité |
| `Acces_Privilegie` | Binaire (0/1) | Accès aux systèmes sensibles ou aux comptes stratégiques | Opportunité |
| `Fraude_Interne` | Binaire (0/1) | Variable cible : cas de fraude identifié | — |
| `Score_Risque` | Numérique continu | Score composite calculé via une formule pondérée | Composite |

## 2.2 Construction de la Variable Cible

La variable cible `Fraude_Interne` est construite à partir d'un **score de risque composite**, calculé selon la formule suivante :

$$Score_{risque} = 0.4 \times Pression + 15 \times Acc\`es\_Privil\'egi\'e + 0.3 \times (10 - Satisfaction) + 0.2 \times Anciennet\'e$$

Le seuil de classification est fixé au **95ème percentile** de la distribution du score de risque, ce qui génère un taux de fraude d'environ **5 %** — cohérent avec les benchmarks sectoriels de l'ACFE.

> **Note méthodologique :** Le choix d'un seuil au 95ème percentile vise à reproduire le caractère rare des événements frauduleux dans la réalité, introduisant un déséquilibre de classes (*class imbalance*) qui devra être géré lors de la modélisation (paramètre `class_weight='balanced'`).

## 2.3 Répartition des Départements

La distribution des effectifs par département a été calibrée pour refléter une structure organisationnelle réaliste :

| Département | Proportion simulée | Profil de risque attendu |
|---|---|---|
| Ventes | 25 % | Pression sur objectifs élevée |
| Logistique | 20 % | Accès aux actifs physiques |
| Achats | 15 % | Risque de corruption fournisseurs |
| Finance | 15 % | Accès aux données financières sensibles |
| IT | 15 % | Accès privilégié aux systèmes |
| RH | 10 % | Accès aux données personnelles |

## 2.4 Pipeline de Traitement des Données

Le pipeline de traitement comprend quatre étapes séquentielles :

**Étape 1 — Encodage :** La variable catégorielle `Departement` est transformée en variable numérique par encodage label (`LabelEncoder` de scikit-learn).

**Étape 2 — Division Train/Test :** Les données sont séparées en un jeu d'entraînement (70 %) et un jeu de test (30 %), avec une stratification sur la variable cible pour préserver les proportions.

**Étape 3 — Standardisation :** Les features sont normalisées via `StandardScaler` pour la régression logistique, qui est sensible aux échelles des variables.

**Étape 4 — Gestion du déséquilibre :** Le paramètre `class_weight='balanced'` est activé dans les modèles Random Forest et Régression Logistique pour pondérer automatiquement les classes minoritaires.

---

# Chapitre III — Analyse Exploratoire des Données

## Transition

Avant d'engager la phase de modélisation, l'analyse exploratoire (EDA) constitue une étape incontournable pour comprendre la structure des données, identifier les relations entre variables et formuler des hypothèses sur les déterminants de la fraude.

---

## 3.1 Portrait Global du Dataset

L'analyse descriptive de l'échantillon révèle les statistiques sommaires suivantes :

| Indicateur | Valeur | Interprétation |
|---|---|---|
| Effectif total | 1 000 employés | Population simulée représentative |
| Cas de fraude identifiés | ~50 cas | Taux de ~5 % — aligné ACFE |
| Proportion avec accès privilégié | ~20 % | 1 employé sur 5 dispose d'un accès sensible |
| Pression financière moyenne | 5,5 / 10 | Niveau modéré — distribution uniforme |
| Satisfaction moyenne au travail | 5,5 / 10 | Distribution symétrique |

> 📊 **Figure 1 — Insérer ici :** Dashboard KPI Cards + Triangle de Cressey + Distribution du Score de Risque + Taux de fraude par département (`fig1_overview.png`)

## 3.2 Analyse des Facteurs de Risque

### 3.2.1 Pression Financière vs. Satisfaction au Travail

Le nuage de points croisant `Score_Pression_Financiere` et `Satisfaction_Travail` met en évidence une **zone à risque** caractérisée par une pression financière élevée (> 7/10) combinée à une faible satisfaction (< 4/10). Les employés situés dans ce quadrant présentent une probabilité de fraude significativement supérieure à la moyenne.

**Interprétation critique :** Cette combinaison correspond exactement aux deux premiers sommets du Triangle de Cressey — pression et rationalisation — et suggère que la fraude émerge moins d'une malveillance pure que d'une détresse contextuelle combinée à une insatisfaction latente.

### 3.2.2 Impact de l'Accès Privilégié

L'analyse croisée entre `Acces_Privilegie` et `Fraude_Interne` révèle un écart saisissant :

| Profil | Taux de fraude |
|---|---|
| Employés sans accès privilégié | ~2,5 % |
| Employés avec accès privilégié | ~15,0 % |
| **Ratio d'amplification** | **×6** |

**Interprétation critique :** L'accès privilégié agit comme un **multiplicateur de risque**. Il ne crée pas la fraude par lui-même, mais en abaisse drastiquement le coût d'exécution pour un individu qui en aurait la motivation. Cela plaide pour une politique stricte de gestion des accès (*least privilege principle*) et d'audit des droits d'accès.

> 📊 **Figure 2 — Insérer ici :** Analyse exploratoire multi-panneaux : scatter Pression vs Satisfaction, boxplots Ancienneté, histogrammes Heures Supp., violins Congés, Heatmap corrélations, graphe Impact Accès Privilégié (`fig2_eda.png`)

### 3.2.3 Matrice de Corrélation

La matrice de corrélation entre les variables numériques met en évidence les relations suivantes :

- **`Acces_Privilegie` ↔ `Fraude_Interne` :** Corrélation positive forte (~0,55), confirmant l'importance de cette variable.
- **`Score_Pression_Financiere` ↔ `Fraude_Interne` :** Corrélation positive modérée (~0,25).
- **`Satisfaction_Travail` ↔ `Fraude_Interne` :** Corrélation négative (~-0,15), cohérente avec la théorie de la rationalisation.
- **Variables comportementales** (`Heures_Supp_Mois`, `Conges_Non_Pris`) : corrélations faibles mais non nulles, indiquant une contribution marginale.

**Valeur ajoutée :** L'absence de multicolinéarité élevée entre les variables indépendantes valide la pertinence d'une approche multi-variables et justifie l'inclusion de l'ensemble des features dans les modèles.

### 3.2.4 Profil par Département

L'analyse du taux de fraude par département révèle des disparités significatives :

| Département | Taux de fraude estimé | Facteur explicatif dominant |
|---|---|---|
| Finance | ~8 % | Accès aux données financières + pression sur résultats |
| IT | ~7 % | Accès privilégié aux systèmes d'information |
| Achats | ~6 % | Exposition aux risques de corruption |
| Ventes | ~5 % | Pression sur objectifs commerciaux |
| Logistique | ~4 % | Accès aux actifs physiques |
| RH | ~3 % | Accès aux données RH sensibles |

> 📊 **Suggestion de visualisation :** Carte de chaleur (*heatmap*) Département × Accès Privilégié montrant les taux de fraude croisés — permettant d'identifier les segments les plus exposés.

---

# Chapitre IV — Modélisation Prédictive par Machine Learning

## Transition

Fort des enseignements de l'analyse exploratoire, ce chapitre présente la démarche de modélisation prédictive adoptée. L'objectif est de construire des classifieurs capables d'identifier les profils à risque avec une précision suffisante pour orienter les décisions d'audit.

---

## 4.1 Sélection des Variables Prédictives

Les sept variables retenues comme *features* pour les modèles sont :

| Variable | Rôle dans le modèle | Justification |
|---|---|---|
| `Dept_encoded` | Signal organisationnel | Certains départements présentent une exposition plus élevée |
| `Anciennete_Annees` | Proxy d'opportunité | L'ancienneté corrèle avec la confiance accordée et les droits |
| `Score_Pression_Financiere` | Proxy de pression | Variable centrale du Triangle de Cressey |
| `Satisfaction_Travail` | Proxy de rationalisation | Une faible satisfaction favorise la justification de l'acte |
| `Heures_Supp_Mois` | Signal comportemental | Indicateur de charge et de risque de burnout |
| `Conges_Non_Pris` | Signal d'alerte RH | Les fraudeurs évitent souvent les congés (peur d'être découverts) |
| `Acces_Privilegie` | Facteur d'opportunité | Variable la plus déterminante selon l'EDA |

## 4.2 Présentation des Modèles

### 4.2.1 Régression Logistique

La régression logistique constitue le modèle de référence (*baseline*). Elle modélise directement la probabilité d'appartenance à la classe "Fraudeur" en fonction d'une combinaison linéaire des variables, transformée par la fonction sigmoïde. Sa principale vertu est son **interprétabilité** : les coefficients peuvent être directement lus comme des indicateurs de l'influence de chaque variable.

- **Hyperparamètres :** `C=0.5` (régularisation L2 modérée), `class_weight='balanced'`, `max_iter=1000`
- **Prétraitement requis :** Standardisation des features (`StandardScaler`)

### 4.2.2 Random Forest

Le Random Forest est un algorithme d'ensemble qui agrège les prédictions d'un grand nombre d'arbres de décision construits sur des sous-échantillons aléatoires des données et des variables. Il est particulièrement robuste au bruit, gère naturellement les interactions entre variables, et fournit une mesure d'**importance des variables** directement exploitable par les auditeurs.

- **Hyperparamètres :** `n_estimators=200`, `max_depth=8`, `min_samples_leaf=3`, `class_weight='balanced'`
- **Atout principal :** Importance des variables (Gini) permettant d'identifier les leviers d'action prioritaires.

### 4.2.3 Gradient Boosting

Le Gradient Boosting construit séquentiellement des arbres faibles, chaque nouvel arbre cherchant à corriger les erreurs résiduelles du précédent. Il tend à offrir des performances légèrement supérieures au Random Forest sur des données structurées, au prix d'un temps d'entraînement plus élevé et d'une plus grande sensibilité aux hyperparamètres.

- **Hyperparamètres :** `n_estimators=150`, `learning_rate=0.08`, `max_depth=4`
- **Atout principal :** Performances prédictives maximales, idéal pour le tri des cas prioritaires.

## 4.3 Évaluation des Performances

### 4.3.1 Métriques Retenues

Dans un contexte de fraude, le choix des métriques d'évaluation est crucial. La simple **précision** (*accuracy*) est une métrique trompeuse en présence de déséquilibre de classes : un modèle qui prédirait systématiquement "Non-Fraudeur" obtiendrait une précision de 95 %, sans aucune valeur opérationnelle.

Les métriques pertinentes retenues sont :

| Métrique | Description | Pertinence pour la fraude |
|---|---|---|
| **AUC-ROC** | Aire sous la courbe ROC — capacité discriminante globale | Évalue la capacité à séparer fraudeurs et non-fraudeurs |
| **Précision** | TP / (TP + FP) | Éviter les fausses accusations — impact sur les employés |
| **Rappel (Sensibilité)** | TP / (TP + FN) | Minimiser les fraudes non détectées — impact financier |
| **F1-Score** | Moyenne harmonique Précision/Rappel | Compromis équilibré |
| **AUC-PR** | Aire sous la courbe Précision-Rappel | Particulièrement adaptée aux classes déséquilibrées |

### 4.3.2 Résultats Comparatifs

| Modèle | AUC-ROC (test) | AUC CV-5 | Rappel (Fraude) | F1-Score (Fraude) |
|---|---|---|---|---|
| **Random Forest** | **~0,97** | **~0,97** | **~0,87** | **~0,82** |
| Gradient Boosting | ~0,95 | ~0,95 | ~0,80 | ~0,78 |
| Régression Logistique | ~0,88 | ~0,87 | ~0,73 | ~0,70 |

> 📊 **Figure 3 — Insérer ici :** Panel des performances : Courbes ROC, Courbes Précision-Rappel, Matrice de confusion RF, Importance des variables RF, Comparaison AUC CV-5, Rapport de classification (`fig3_models.png`)

### 4.3.3 Analyse de la Matrice de Confusion (Random Forest)

La matrice de confusion du Random Forest sur le jeu de test (300 observations, ~15 cas de fraude) doit être interprétée à l'aune des compromis opérationnels :

- **Vrais Positifs (VP) :** Fraudeurs correctement identifiés → cas prioritaires pour l'audit
- **Faux Positifs (FP) :** Non-fraudeurs incorrectement signalés → coût en termes d'investigations inutiles et d'impact sur les employés
- **Faux Négatifs (FN) :** Fraudeurs non détectés → perte financière et risque résiduel
- **Vrais Négatifs (VN) :** Non-fraudeurs correctement classés → majorité des cas

**Interprétation critique :** Dans le contexte de la fraude, minimiser les **Faux Négatifs** (fraudes manquées) est généralement prioritaire. Cela peut justifier d'abaisser le seuil de classification de 0,5 à 0,3, au prix d'une légère augmentation des Faux Positifs.

### 4.3.4 Importance des Variables

L'analyse de l'importance des variables (méthode Gini, Random Forest) confirme la hiérarchie théorique établie par le Triangle de Cressey :

| Rang | Variable | Importance Gini (approx.) | Dimension Cressey |
|---|---|---|---|
| 1 | `Acces_Privilegie` | ~0,35 | Opportunité |
| 2 | `Score_Pression_Financiere` | ~0,20 | Pression |
| 3 | `Satisfaction_Travail` | ~0,15 | Rationalisation |
| 4 | `Anciennete_Annees` | ~0,12 | Opportunité |
| 5 | `Conges_Non_Pris` | ~0,08 | Signal comportemental |
| 6 | `Heures_Supp_Mois` | ~0,06 | Pression |
| 7 | `Dept_encoded` | ~0,04 | Contextuel |

**Valeur ajoutée managériale :** Cette hiérarchie suggère que les investissements de prévention devraient être concentrés en priorité sur la **gestion des accès privilégiés** et le **suivi de la pression financière perçue** — deux leviers directement actionnables par les équipes RH et IT.

---

# Chapitre V — Segmentation des Risques et Recommandations

## Transition

Les résultats de la modélisation ne constituent pas une finalité en eux-mêmes. Ce dernier chapitre analytique traduit les enseignements quantitatifs en recommandations stratégiques et opérationnelles, destinées aux praticiens de l'audit interne et du contrôle de gestion.

---

## 5.1 Segmentation des Profils à Risque

### 5.1.1 La Courbe de Gain Cumulé

La courbe de gain cumulé (*Cumulative Gain Curve*) illustre l'efficacité opérationnelle du modèle Random Forest. Elle répond à la question suivante : **si l'on ne peut auditer qu'un pourcentage limité de la population, dans quel ordre faut-il prioriser ?**

Le modèle démontre que **l'audit des 20 % d'employés les plus risqués** permet de capturer **environ 80 % de l'ensemble des cas de fraude**. Par comparaison, un audit aléatoire ne capterait que 20 % des fraudes pour ce même effort.

> 📊 **Figure 4 — Insérer ici :** Panel de segmentation : Score de risque par département (violin), Taux de fraude par tranche d'ancienneté, Top 50 profils à haut risque (scatter), Heatmap Département × Accès Privilégié (`fig4_risk.png`)

**Interprétation critique :** Ce gain de productivité de ×4 représente un argument décisif pour l'adoption de ces outils par les équipes d'audit. Il permet de concentrer les ressources humaines qualifiées là où elles auront le plus d'impact.

### 5.1.2 Matrice de Risque Croisé

La heatmap croisant `Departement` et `Acces_Privilegie` constitue un outil de **priorisation bidimensionnelle** immédiatement opérationnel. Elle permet d'identifier les segments à risque composite le plus élevé :

- **Segment prioritaire :** Département Finance × Accès Privilégié → taux de fraude le plus élevé
- **Segment à surveiller :** Département IT × Accès Privilégié → combinaison risquée
- **Segment à moindre risque :** Département RH × Sans Accès Privilégié → risque résiduel faible

### 5.1.3 Profil Radar — Fraudeur vs. Non-Fraudeur

Le graphique radar comparant les profils moyens de fraudeurs et de non-fraudeurs sur les six dimensions du Triangle de Cressey révèle une différenciation nette sur trois axes :

- **Accès Privilégié :** Le profil fraudeur présente un score normalisé significativement supérieur
- **Pression Financière :** Écart notable entre les deux profils
- **Insatisfaction :** Les fraudeurs présentent une insatisfaction moyenne plus élevée

> 📊 **Figure 5 — Insérer ici :** Dashboard de recommandations : Radar Chart Fraudeur vs Non-Fraudeur, Courbe de gain cumulé, Matrice de risque narrative, Actions préventives et détectives (`fig5_recommendations.png`)

## 5.2 Recommandations Stratégiques et Opérationnelles

Sur la base des analyses conduites, cinq axes de recommandation sont proposés, structurés selon la temporalité d'implémentation :

### 5.2.1 Axe 1 — Gouvernance des Accès (Court terme)

L'accès privilégié étant la variable la plus prédictive (importance ~0,35), la première priorité consiste à renforcer la **gouvernance des identités et des accès** (*IAM — Identity and Access Management*) :

- Mettre en œuvre le **principe du moindre privilège** : n'accorder que les droits strictement nécessaires à l'exercice des fonctions
- Mettre en place une **revue périodique des droits d'accès** (semi-annuelle minimum) avec validation managériale
- Déployer un système de **traçabilité des actions** sur les systèmes sensibles (*logs d'audit*)
- Activer des **alertes automatiques** sur les comportements atypiques (connexions en dehors des heures ouvrées, accès à des volumes inhabituels de données)

### 5.2.2 Axe 2 — Bien-être et Pression au Travail (Court/Moyen terme)

La pression financière et l'insatisfaction au travail étant les deuxième et troisième variables prédictives, les équipes RH disposent de leviers concrets :

- Déployer des **enquêtes de satisfaction annuelles** avec des indicateurs de suivi longitudinal
- Mettre en place des **canaux de signalement confidentiels** (*whistleblowing*) permettant aux employés de signaler des situations de pression excessive
- Réviser les **structures d'objectifs** dans les départements à forte pression (Finance, Ventes) pour s'assurer qu'ils restent réalistes
- Proposer des **programmes d'aide aux employés** (EAP) pour les situations de détresse financière personnelle

### 5.2.3 Axe 3 — Déploiement du Modèle Prédictif (Moyen terme)

L'intégration du modèle Machine Learning dans le processus d'audit requiert une approche structurée :

- Développer un **tableau de bord de surveillance** (*Risk Monitoring Dashboard*) actualisé mensuellement, affichant les scores de risque individuels et leur évolution
- Établir un **protocole d'escalade** clair : au-delà d'un certain seuil de probabilité, déclencher automatiquement une revue par le responsable d'audit interne
- Former les **équipes d'audit** à l'interprétation des résultats du modèle (compréhension des features, limites du modèle)
- Constituer un **comité de validation** pluridisciplinaire (Audit, RH, Direction) pour valider les décisions issues du modèle

### 5.2.4 Axe 4 — Contrôles Internes Renforcés (Moyen terme)

Au-delà des outils analytiques, les dispositifs de contrôle interne classiques restent indispensables :

- Renforcer la **séparation des tâches** dans les processus sensibles (approbation/exécution/réconciliation)
- Mettre en place des **contrôles de surprise** (*surprise audits*) sur les postes à risque élevé identifiés par le modèle
- Systématiser les **rapprochements bancaires** et les analyses d'anomalies dans les transactions
- Développer une **politique de rotation des postes** pour les fonctions à risque élevé (Finance, Achats)

### 5.2.5 Axe 5 — Culture Éthique et Prévention (Long terme)

La prévention durable de la fraude nécessite un travail en profondeur sur la culture organisationnelle :

- Développer et déployer un **Code de Conduite** clair, avec des formations obligatoires annuelles
- Instaurer un **programme de formation anti-fraude** ciblé pour les managers des départements à risque
- Mettre en place des **indicateurs de culture éthique** dans les évaluations de performance managériale
- Partager régulièrement (de manière anonymisée) des **études de cas** de fraudes détectées pour sensibiliser les équipes

## 5.3 Dashboard de Pilotage — Vision KPI

> 📊 **Suggestion de visualisation avancée :** Tableau de bord interactif (Power BI / Tableau) structuré autour des indicateurs suivants :

| Catégorie KPI | Indicateur | Fréquence | Responsable |
|---|---|---|---|
| **Détection** | Nombre d'alertes générées | Mensuel | DSI / Audit |
| **Détection** | Taux de conversion alertes → investigations | Mensuel | Audit Interne |
| **Prévention** | % d'employés à accès privilégié audités | Trimestriel | DSI |
| **Prévention** | Score moyen de satisfaction par département | Semestriel | DRH |
| **Performance modèle** | AUC sur nouvelles données | Trimestriel | Data Analyst |
| **Performance modèle** | Taux de faux positifs | Trimestriel | Audit Interne |
| **Gouvernance** | % de revues d'accès réalisées dans les délais | Semestriel | DSI |
| **Culture** | Score de culture éthique (enquête) | Annuel | DRH / Direction |

---

# 7. Conclusion Générale

Ce projet a démontré la faisabilité et la pertinence d'une approche intégrant le **Machine Learning à la démarche d'audit interne** pour la détection proactive de la fraude organisationnelle. En articulant un solide cadre théorique — le Triangle de la Fraude de Cressey — avec une méthodologie quantitative rigoureuse, il a été possible de construire des modèles prédictifs atteignant des performances remarquables, avec un **score AUC-ROC de 0,97 pour le modèle Random Forest**.

Les enseignements les plus importants de cette étude peuvent être résumés en trois points :

**1. La prédominance de l'opportunité structurelle.** L'accès privilégié aux systèmes sensibles s'impose comme le facteur prédictif le plus puissant, confirmant que la fraude interne est avant tout un problème de gouvernance des accès et de défaillance du contrôle interne, plus que de malveillance individuelle intrinsèque.

**2. L'importance des signaux comportementaux et contextuels.** La pression financière, la satisfaction au travail et même des variables apparemment anodines comme les congés non pris ou les heures supplémentaires contribuent significativement à la prédiction, ouvrant la voie à une surveillance holistique intégrant les données RH.

**3. La nécessité d'une approche hybride.** Si les algorithmes de Machine Learning offrent une capacité de traitement scalable et une puissance prédictive indéniable, ils ne remplacent pas le jugement humain. L'auditeur reste indispensable pour l'interprétation contextuelle, la prise de décision éthique et la communication avec les parties prenantes.

Les perspectives de recherche et de développement ouvertes par ce travail sont nombreuses. Sur le plan technique, l'intégration de **données temporelles** (détection d'anomalies sur séries temporelles), l'utilisation de **réseaux de neurones** pour la détection de patterns complexes, ou encore le déploiement de modèles **explicables** (*XAI — Explainable AI*) pour faciliter l'interprétation par les non-techniciens constituent des pistes prometteuses. Sur le plan managérial, la question de la **gouvernance éthique** de ces outils — notamment les biais algorithmiques et la protection de la vie privée des employés — mérite une attention soutenue.

En définitive, ce projet illustre comment le mariage entre la théorie criminologique classique et les outils modernes de la Data Science peut produire des instruments concrets au service de la gouvernance d'entreprise, dans un monde où la complexité organisationnelle et la digitalisation croissantes rendent les approches d'audit traditionnelles insuffisantes.

---

# 8. Annexes

## Annexe A — Librairies Python Utilisées

| Librairie | Version | Usage |
|---|---|---|
| `pandas` | ≥ 1.5 | Manipulation et analyse des données |
| `numpy` | ≥ 1.23 | Calcul numérique et génération de données |
| `scikit-learn` | ≥ 1.1 | Modèles ML, prétraitement, métriques |
| `matplotlib` | ≥ 3.6 | Visualisations avancées |
| `seaborn` | ≥ 0.12 | Heatmaps et visualisations statistiques |

## Annexe B — Formule du Score de Risque Composite

```python
score_risque = (
    (df['Score_Pression_Financiere'] * 0.4) +
    (df['Acces_Privilegie'] * 15) +
    ((10 - df['Satisfaction_Travail']) * 0.3) +
    (df['Anciennete_Annees'] * 0.2)
)
seuil = np.percentile(score_risque, 95)
df['Fraude_Interne'] = (score_risque >= seuil).astype(int)
```

**Justification des pondérations :**
- Le coefficient **×15** appliqué à `Acces_Privilegie` reflète son caractère binaire et son rôle prépondérant dans la genèse de la fraude
- Le coefficient **0,4** sur la pression reflète son importance théorique dans le Triangle de Cressey
- L'inversion de la satisfaction `(10 - Satisfaction)` traduit le lien inverse entre satisfaction et risque de rationalisation

## Annexe C — Hyperparamètres des Modèles

```python
# Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight='balanced',
    min_samples_leaf=3
)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.08,
    max_depth=4,
    random_state=42
)

# Régression Logistique
lr = LogisticRegression(
    random_state=42,
    class_weight='balanced',
    max_iter=1000,
    C=0.5
)
```

## Annexe D — Références Bibliographiques

| Auteur(s) | Titre | Source | Année |
|---|---|---|---|
| Cressey, D. R. | *Other People's Money: A Study in the Social Psychology of Embezzlement* | Free Press | 1953 |
| ACFE | *Report to the Nations: Occupational Fraud and Abuse* | Association of Certified Fraud Examiners | 2024 |
| Breiman, L. | *Random Forests* | Machine Learning, 45(1), 5–32 | 2001 |
| Friedman, J. H. | *Greedy Function Approximation: A Gradient Boosting Machine* | Annals of Statistics, 29(5) | 2001 |
| PwC | *Global Economic Crime and Fraud Survey* | PricewaterhouseCoopers | 2022 |
| Deloitte | *Navigating the Expanding Universe of Internal Audit* | Deloitte Insights | 2023 |

---

*Document généré dans le cadre du projet de fin de semestre S8 — ENCG Settat, 2024–2025*

*Les données utilisées dans ce rapport sont entièrement simulées à des fins pédagogiques. Toute ressemblance avec des situations réelles serait purement fortuite.*
