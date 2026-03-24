# Rapport d'Étude : Prédiction de Fraude Interne

## 1. SYNTHÈSE EXÉCUTIVE (EXECUTIVE SUMMARY)

La fraude interne représente un risque financier et réputationnel majeur pour les organisations modernes. Dans un contexte où les contrôles traditionnels peinent à identifier des schémas comportementaux complexes, ce projet propose une solution technologique proactive basée sur l'apprentissage automatique (Machine Learning). L'objectif est de modéliser et de prédire la probabilité qu'un employé commette un acte de fraude, en s'appuyant sur les principes du Triangle de la Fraude (Pression, Opportunité, Rationalisation).

La solution technique optimale retenue est un modèle de **Logistic Regression** (Régression Logistique). À travers une validation croisée rigoureuse (CV=5) sur un échantillon de 1000 employés comprenant 5% de cas avérés, ce modèle a surpassé ses concurrents (Random Forest, Gradient Boosting) avec une performance exceptionnelle, atteignant une AUC de **0.9997**. Le résultat majeur de notre analyse met en exergue trois vecteurs de risque dominants : la possession d'un **Accès Privilégié** (importance de 0.524), une forte **Ancienneté** (0.147), et une **Pression Financière** élevée (0.142).

Parmi les départements, les **Ressources Humaines (RH)** et les **Ventes** constituent les zones de vulnérabilité critiques avec un taux de fraude de 6.2%. La recommandation stratégique centrale est d'initier un audit prioritaire et ciblé sur le Top 5% des profils à très haut risque générés par cet algorithme, garantissant un retour sur investissement maximal via l'optimisation des efforts de contrôle, tout en appliquant systématiquement le principe du moindre privilège informatique.

---

## 2. FRONT MATTER

### 2.1. Sommaire détaillé

1. Synthèse Exécutive (Executive Summary)
2. Front Matter
   2.1. Sommaire détaillé
   2.2. Introduction
3. Étude Technique & Méthodologie
   3.1. Descriptif du Dataset
   3.2. Revue de Littérature & Jargon
   3.3. Méthode d'Apprentissage
4. Analyse des Figures et Captures d'Écran
   4.1. Vue Globale et Taux de Fraude
   4.2. Analyse Exploratoire : Profils à Risque
   4.3. Évaluation des Performances des Modèles
   4.4. Segmentation des Risques et Profilage
   4.5. Gouvernance et Recommandations
5. Architecture du Code (Notebook)
6. Compte Rendu Final & Perspectives

### 2.2. Introduction

Dans un monde professionnel de plus en plus numérisé, la menace sécuritaire la plus redoutable provient souvent de l'intérieur de l'organisation. L'ACFE (Association of Certified Fraud Examiners) rapporte chaque année que la fraude interne ampute considérablement les revenus des entreprises. La complexité de la fraude interne, souvent perpétrée par des employés disposant de la confiance de la hiérarchie et connaissant les failles du contrôle interne, la rend extrêmement difficile à détecter par de simples audits aléatoires.

Cette étude de cas vise à répondre à la problématique suivante : *Comment les algorithmes d'Intelligence Artificielle peuvent-ils transformer les données comportementales et organisationnelles en un radar tridimensionnel (Opportunité, Pression, Rationalisation) capable d'anticiper la fraude interne ?* 
À travers l'évaluation algorithmique, notre but est de passer d'un mode "réactif" historique à un système de gouvernance "prédictif".

---

## 3. ÉTUDE TECHNIQUE & MÉTHODOLOGIE

### 3.1. Descriptif du Dataset

L'analyse est construite sur un jeu de données simulant une entreprise de **1 000 collaborateurs**, comprenant une variable cible binaire : la commission d'une fraude (1) ou non (0).

- **Structure et Déséquilibre :** Le dataset se compose de 950 employés non-fraudeurs et **50 cas de fraude visés**, établissant un taux de fraude de base de **5.0%**. Il s'agit classiquement d'un jeu de données déséquilibré (imbalanced dataset), requérant des métriques spécifiques (AUC) plutôt qu'une simple exactitude (Accuracy).
- **Variables explicatives :** Le dataset intègre un mix riche de données continues (Ancienneté en années, Heures supplémentaires par mois, jours de congés non pris) et de données ordinales/catégorielles (Score de Pression Financière sur 10, Satisfaction au travail sur 10, Département d'affectation, Accès Privilégié binaire).

### 3.2. Revue de Littérature & Jargon

L'analyse prédictive de la fraude trouve un écho particulier dans les travaux liés aux variables binaires complexes. Les fondements théoriques (ex: *Breiman, 2001* sur le Machine Learning algorithmique face au Machine Learning statistique) justifient l'emploi de plusieurs algorithmes comparatifs. La modélisation est inspirée du "Triangle de la Fraude" (Cressey, 1953), concept fondamental en audit.

> **Définitions Techniques :**
> - **$R^2$ (Coefficient de Détermination) :** Mesure statistique déterminant la proportion de variance de la variable expliquée par le modèle. Moins adapté à la classification binaire, on lui préfère l'AUC (Area Under ROC Curve).
> - **Overfitting (Surapprentissage) :** Piège méthodologique où un algorithme "mémorise" le bruit du jeu de données d'entraînement au lieu de capturer les relations générales, conduisant à de mauvaises prédictions sur de nouvelles données.
> - **Gradient Descent (Descente de Gradient) :** Algorithme mathématique d'optimisation itérative servant à minimiser la fonction d'erreur d'un modèle en modifiant ses paramètres dans le sens inverse du gradient (la pente de l'erreur).

### 3.3. Méthode d'Apprentissage

Notre banc d'essai a confronté la Régression Logistique à des méthodes d'ensemble avancées telles que le *Random Forest* (Arbres Aléatoires) et le *Gradient Boosting*. 

**Choix Algorithmique Final : La Logistic Regression.**
La probabilité qu'un événement survienne (la fraude) est définie par l'équation de la fonction sigmoïde :
$$ P(y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \dots + \beta_n X_n)}} $$

La justification de ce choix est double : d'une part, une stabilité et une performance exceptionnelles prouvées en validation croisée à 5 blocs (CV-5) où le modèle conserve une moyenne AUC de **0.9997**. D'autre part, son **interprétabilité** (explainability) supérieure : contrairement à une "boîte noire", la régression logistique permet d'attribuer un poids univoque à une modification de la situation d’un salarié, facilitant l’explication de ses prédictions (ex: passage à un accès IT privilégié).

---

## 4. ANALYSE DES FIGURES ET CAPTURES D'ÉCRAN (SECTION PHOTOS)

*Note pour l'intégration : Veuillez insérer ci-dessous les captures du Dashboard réalisées via Python/Matplotlib/Seaborn sous leurs titres respectifs.*

### [FIGURE 1 : DASHBOARD GLOBAL - PRÉDICTION DE LA FRAUDE INTERNE]
***(Insérer la Capture de la Vue Globale)***
- **Description :** Le panneau du haut condense les KPI globaux : 1000 collaborateurs, 50 fraudes (5%), avec un score de pression moyen de 5.4/10. En dessous, on distingue un diagramme en barres horizontales du taux de fraude par département, et un histogramme illustrant la distribution du Score de Risque avec une nette ségrégation bicolore (Bleu vs Rouge).
- **Interprétation :** Les départements des Ventes et des Ressources Humaines affichent une prévalence anormale (6.2%). Plus significatif encore, la distribution du score de risque montre un seuil discriminant net ($\text{Seuil} = 22.3$). Le modèle sépare de façon chirurgicale la masse globale saine de la population "rouge" à très haut risque, dénotant d'une excellente discrétisation (convergence mathématique efficace).

### [FIGURE 2 : ANALYSE EXPLORATOIRE — PROFILS À RISQUE]
***(Insérer la Capture des Boxplots et Corrélations)***
- **Description :** On observe plusieurs graphiques multivariés. Un nuage de points "Pression vs Satisfaction", des diagrammes en boîte (boxplots) de l'ancienneté (les fraudeurs ont une médiane supérieure à 20 ans au lieu de ~12 ans), la Heatmap de la Matrice de Corrélation, et un diagramme à barres comparant l'impact de l'accès privilégié (0% des sans-accès vs 26.3% de fraudes chez les personnes avec accès).
- **Interprétation :** Un cluster comportemental émerge nettement : le fraudeur typique compte une forte ancienneté limitant ses jours de congé (indicateur classique pour empêcher la découverte d'actes prohibés). La corrélation directe entre l'accès privilégié (+0.47) et la fraude justifie le pôle "Opportunité" du modèle de Cressey. Aucun employé sans accès n'a commis de fraude, prouvant que la motivation ne suffit pas sans le droit d'action SI.

### [FIGURE 3 : PERFORMANCES DES MODÈLES DE CLASSIFICATION]
***(Insérer la Capture des Courbes ROC et Matrice Confusion)***
- **Description :** Apparaissent simultanément la Courbe ROC parfaite (modèles au-dessus de 0.99 de score), la matrice de confusion pour la matrice RF/LogReg, les importances des variables (Feature Importance, avec "Accès Privilégié" menant à 0.524), ainsi que le rapport Scikit-Learn de validation croisée affichant $0.9997$ pour la LogReg.
- **Interprétation :** Le modèle n'est affecté par quasiment aucun faux positif ou faux négatif. La courbe ROC collée à l'axe ordonné signifie une distinction totale et sans compromis du biais d'erreur (AUC idéalement proche de 1). Surtout, la variable "Accès Privilégié" capte plus de la moitié (52.4%) du pouvoir prédictif. Le modèle a donc identifié la faille physique avant la faille psychologique.

### [FIGURE 4 : SEGMENTATION DES PROFILS À RISQUE]
***(Insérer la Capture de la Heatmap Dept x Accès)***
- **Description :** Cette planche dévoile un "Violin Plot" des scores de risque selon les départements, mais aussi l'évolution du taux de fraude selon l'ancienneté (+13.5% après 18 ans de service). La Heatmap "Département $\times$ Accès" montre des cellules virant au rouge sombre. Une courbe décroissante (Waterfall style) pointe vers le Top 50 des suspectés.
- **Interprétation :** Ce bloc permet de cibler chirurgicalement l'audit. Le risque n'est pas homogène. Ainsi, un membre du département RH disposant d'un Accès Privilégié a un taux d'occurrence alarmant de 35% ! En Ventes avec Accès, c'est 30%. Ces cellules isolées forment les îlots majeurs de risque à auditer d'urgence.

### [FIGURE 5 : RECOMMANDATIONS DE CONTRÔLE INTERNE]
***(Insérer la Capture du Radar Profil et Courbe de Gain)***
- **Description :** Le radar oppose visuellement la surface d'un fraudeur (très étendue sur l'Accès, Pression et Ancienneté) face à un non-fraudeur (au centre). La courbe de Gain prouve qu'auditer le top 20% de la population permet de découvrir 100% des fraudes. Enfin, une grille stratégique distribue les politiques internes (Surveillance accrue vs Risque Critique).
- **Interprétation :** La courbe de Gain constitue l'argument décisif pour la direction financière : le modèle machine learning permet de couvrir la totalité du gisement des fraudes en n'essayant (et c'est un gain budgétaire immédiat) que 20% du volume des employés. Cela légitime l'utilisation de l'intelligence artificielle pour prioriser les coûts d'investigation.

---

## 5. ARCHITECTURE DU CODE (NOTEBOOK)

Afin d'assurer la reproductibilité scientifique et industrielle, notre pipeline (Notebook Jupyter / Python) s'est structurée via le framework `scikit-learn` et `pandas` :

1. **Importation et Setup :** Chargement des dépendances natives (Numpy, Pandas, Matplotlib, Seaborn) pour manipuler les données tabulaires et la matrice.
2. **Exploratory Data Analysis (EDA) :** Implémentation de visualisations bivariées avec `seaborn` (Boxplots, Violins, Heatmap de corrélation) pour inspecter l'asymétrie de la "Target" de 5%.
3. **Feature Engineering & Preprocessing :** Sélection et ingénierie des variables. Séparation via un `train_test_split` (80% / 20%) et préparation via la standardisation `StandardScaler` (essentielle pour la descente de gradient d'une régression logistique) pour mettre sur une échelle commune la pression et l'ancienneté.
4. **Training (Modélisation Supervisée) :** Instanciation des estimateurs `LogisticRegression`, `RandomForestClassifier` et `GradientBoostingClassifier`. Usage de paramètres ajustés au déséquilibre comme `class_weight='balanced'`. 
5. **Évaluation (Evaluation & Metrics) :** Le module `cross_val_score` avec `K-Fold` scinde les données en 5 parties tournantes, renvoyant finalement l'AUC. Présentation de la matrice de prédiction `confusion_matrix` et un `classification_report`.

---

## 6. COMPTE RENDU FINAL & PERSPECTIVES

### 6.1. Analyse Métrique Comparative
La performance absolue démontre que l'association des comportements chroniques (ancienneté / pression) et techniques (accès TI) décrit parfaitement un comportement non éthique.

| Modèle Utilisé            | Score AUC (CV-5) | Variable Majeure d'Explication |
|:--------------------------|:----------------:|:-------------------------------|
| **Logistic Regression**   | **0.9997**       | Accès Privilégié (52.4%)       |
| Gradient Boosting         | 0.9952           | Accès Privilégié / Ancienneté  |
| Random Forest             | 0.9951           | Pression Financière            |

> 🚨 **AVIS DE CONFORMITÉ PRIORITAIRE :**
> Le Top 10 des profils listés par notre algorithme doit faire l'objet d'un audit immédiat. 
> Exemples critiques à vérifier sous 48h :
> - **ID 403** (Ventes, 15 ans) → Probabilité Fraude : **97.6%**
> - **ID 661** (Ventes, 23 ans) → Probabilité Fraude : **96.2%**
> - **ID 566** (RH, 24 ans) → Probabilité Fraude : **95.9%**

### 6.2. Limites de la Modélisation
- **Endogénéité du Score de 0.9997 :** Une telle précision découle d'un Dataset synthétisé avec des règles précises. En production sur des données réelles (Real-World Data), l'introduction de valeurs aberrantes (Outliers) et de bruits liés à l'incertitude humaine devrait stabiliser l'AUC entre 0.85 et 0.90, niveau qui reste extrêmement performant en système de détection d'Audit.
- **Paramètre Psychologique Invisible :** La "rationalisation" (le dernier pilier du triangle, i.e. l'auto-justification de la fraude) manque souvent de points de données quantitatives et est inférée indirectement via l'insatisfaction ou l'ancienneté.

### 6.3. Perspectives et Pistes d'Amélioration
- **Connexion Active (Real-Time ML Pipeline) :** Lier le système de prédiction directement aux journaux de l'Active Directory et à l'ERP (Enterprise Resource Planning) pour réévaluer chaque mois dynamiquement le *Score de Pression* et alerter automatiquement le service de conformité (Compliance).
- **Analyse des Réseaux Sociaux Internes (Graph Analytics) :** Introduire la notion de collusion (qui déjeune avec qui, utilisation partagée d'outils) via la Data Augmentation. L'orchestration de la fraude en bande organisée demeure une menace majeure que ce modèle pourrait traiter dans une version 2.0.
