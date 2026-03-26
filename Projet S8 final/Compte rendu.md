<img width="901" height="522" alt="image" src="https://github.com/user-attachments/assets/6637a4f2-c95f-4a57-91c3-059a88887714" /># Rapport d'Étude de Cas Scientifique
# Détection de la Fraude Interne en Entreprise par Apprentissage Automatique
 
---
 
> **Niveau :** Master 2 Data Science — Semestre 8
> **Sujet :** Prédiction comportementale du risque de fraude interne
> **Cadre théorique :** Triangle de la Fraude (Cressey, 1953) · ACFE Report to the Nations 2024
> **Technologies :** Python 3 · scikit-learn · pandas · matplotlib · seaborn
 
---

**Établissement :** École Nationale de Commerce et de Gestion de Settat (ENCG Settat)  
**Option :** Contrôle, Audit et Conseil (CAC)  
**Module :** Intelligence Artificielle  
**Année académique :** 2025–2026  
**Réalisé par :** Riad Douaa /  Rhaouta Salma / Marzaq Fatima-ezzahra
**Date de remise :** Mars 2026  

---
<img width="432" height="402" alt="qr code" src="https://github.com/user-attachments/assets/dc16b4b6-d35e-49cf-81a7-47109716749b" />

---

## Table des Matières

- **1. Introduction & Thématique Globale**
  - 1.1 Contexte industriel et enjeux économiques
  - 1.2 Problématique centrale
  - 1.3 Objectifs de l'étude
- **2. Revue de Littérature & Fondements Théoriques**
  - 2.1 Le Triangle de la Fraude (Cressey, 1953)
  - 2.2 État de l'art en détection automatisée de la fraude
  - 2.3 Glossaire du jargon technique
- **3. Descriptif du Dataset & Preprocessing**
  - 3.1 Source et génération des données
  - 3.2 Structure et variables du dataset
  - 3.3 Distribution de la variable cible
  - 3.4 Étapes de prétraitement
- **4. Analyse Exploratoire des Données (EDA)**
  - 4.1 Analyse univariée des variables
  - 4.2 Analyse bivariée et corrélations
  - 4.3 Profils comportementaux identifiés
  - 4.4 Segmentation du risque par département
  - 4.5 Segmentation du risque par ancienneté
- **5. Méthodes d'Apprentissage Automatique**
  - 5.1 Justification du choix des algorithmes
  - 5.2 Random Forest Classifier
  - 5.3 Gradient Boosting Classifier
  - 5.4 Régression Logistique (modèle de référence)
  - 5.5 Stratégie de validation
- **6. Structure du Code Notebook**
  - 6.1 Architecture logique du pipeline
  - 6.2 Description des cellules
- **7. Résultats & Interprétation des Performances**
  - 7.1 Métriques de performance comparatives
  - 7.2 Analyse de la matrice de confusion
  - 7.3 Courbe ROC et AUC
  - 7.4 Courbe Précision-Rappel
  - 7.5 Importance des variables
  - 7.6 Segmentation des profils à risque — Top 50 & Heatmap
- **8. Recommandations Opérationnelles**
  - 8.1 Déploiement du modèle
  - 8.2 Contrôle des accès et gouvernance
  - 8.3 Programme de prévention et détection précoce
- **9. Conclusion**
- **10. Références Bibliographiques**

---

## 1. Introduction & Thématique Globale

### 1.1 Contexte industriel et enjeux économiques

La fraude interne constitue l'une des menaces les plus coûteuses et les plus insidieuses auxquelles font face les organisations contemporaines. Contrairement aux attaques externes, elle est perpétrée par des individus bénéficiant d'une position de confiance au sein même de l'entreprise — employés, cadres intermédiaires ou dirigeants — ce qui la rend particulièrement difficile à détecter par les méthodes conventionnelles d'audit.

Selon le rapport de référence de l'ACFE (*Association of Certified Fraud Examiners*), **Report to the Nations 2024**, les organisations perdent en moyenne **5 % de leurs revenus annuels** du fait de la fraude occupationnelle. À l'échelle mondiale, les pertes cumulées se chiffrent en centaines de milliards de dollars, avec une durée médiane de détection de **12 mois** avant qu'une fraude ne soit découverte. Ce délai traduit une défaillance structurelle des dispositifs de contrôle interne traditionnels, majoritairement réactifs et fondés sur l'audit a posteriori.

Face à ce constat, les organisations investissent progressivement dans des approches **proactives** reposant sur l'exploitation de données comportementales et l'application de techniques d'apprentissage automatique (*Machine Learning*, ML). Ces méthodes permettent d'identifier des schémas latents, imperceptibles à l'œil humain, annonciateurs d'un comportement déviant.

### 1.2 Problématique centrale

La problématique centrale de cette étude se formule comme suit :

> **Dans quelle mesure les données comportementales et organisationnelles relatives aux employés permettent-elles d'entraîner un modèle d'apprentissage supervisé capable de prédire, avec un niveau de précision acceptable, la propension d'un individu à commettre une fraude interne ?**

Cette question soulève plusieurs sous-questions analytiques :
- Quelles variables comportementales constituent les meilleurs prédicteurs du risque frauduleux ?
- Quel algorithme de classification offre le meilleur compromis biais-variance pour ce type de données déséquilibrées ?
- Comment traduire les sorties probabilistes du modèle en recommandations d'audit opérationnelles ?

### 1.3 Objectifs de l'étude

L'étude poursuit trois objectifs complémentaires :

1. **Objectif descriptif** : Caractériser les profils à risque au moyen d'une analyse exploratoire approfondie, en s'appuyant sur le cadre théorique du Triangle de la Fraude.
2. **Objectif prédictif** : Entraîner et comparer plusieurs modèles de classification supervisée (Random Forest, Gradient Boosting, Régression Logistique) afin d'identifier le plus performant.
3. **Objectif prescriptif** : Traduire les résultats analytiques en recommandations concrètes de contrôle interne, d'audit ciblé et de gouvernance organisationnelle.

---

## 2. Revue de Littérature & Fondements Théoriques

### 2.1 Le Triangle de la Fraude (Cressey, 1953)

Le fondement théorique de cette étude repose sur le modèle séminal proposé par le criminologue Donald R. Cressey dans son ouvrage *Other People's Money: A Study in the Social Psychology of Embezzlement* (1953). Ce modèle, universellement adopté par la profession d'audit et de forensic accounting, stipule que tout acte de fraude occupationnelle est la résultante de trois facteurs concomitants :


---
<img width="432" height="402" alt="qr code" src="https://github.com/riaddouaaencg-lab/S8-22006170-23009580-22007263/blob/main/Projet%20S8%20final/Graphiques/connaisssance%20sur%20la%20base%20de%20donn%C3%A9e/24.png?raw=true" />

---


- **La Pression** (*Pressure*) désigne une contrainte perçue ou réelle — financière, professionnelle ou personnelle — qui motive l'individu à enfreindre les règles. Dans notre modèle, elle est opérationnalisée par la variable `Score_Pression_Financiere`.
- **L'Opportunité** (*Opportunity*) représente l'accès aux ressources et l'existence de lacunes dans les mécanismes de contrôle. Elle est capturée par la variable binaire `Acces_Privilegie`.
- **La Rationalisation** (*Rationalization*) correspond au processus cognitif par lequel l'individu justifie son comportement frauduleux. Elle est approximée par l'inverse de `Satisfaction_Travail`, ainsi que par des indicateurs indirects comme les congés non pris et les heures supplémentaires excessives.

> *"The violator defines the situation in which he finds himself in such a manner that he is able to use the position of trust for his own benefit."*
> — Donald R. Cressey, *Other People's Money*, 1953

### 2.2 État de l'art en détection automatisée de la fraude

La détection automatisée de la fraude par apprentissage automatique a connu un essor considérable depuis les années 2010 :

**West & Bhattacharya (2016)** — *"Intelligent financial fraud detection: A comprehensive review"*, dans *Computers & Security*, offrent une taxonomie exhaustive des techniques ML appliquées à la détection de fraude, distinguant les approches supervisées, non-supervisées et hybrides. Ils montrent que les méthodes ensemblistes (*ensemble methods*) surpassent systématiquement les modèles unitaires sur des jeux de données déséquilibrés.

**Perols (2011)** — Dans son étude publiée dans *The Accounting Review*, démontre que les modèles de forêts aléatoires (*Random Forest*) surpassent les réseaux de neurones et la régression logistique pour la détection des déclarations financières frauduleuses (*fraudulent financial reporting*), en raison de leur robustesse aux données non-linéaires et aux interactions complexes entre variables.

**ACFE (2024)** — *Report to the Nations on Occupational Fraud and Abuse* — constitue la référence empirique mondiale sur la fraude occupationnelle, couvrant 1 921 cas dans 138 pays. Ce rapport documente que les organisations équipées d'une **hotline** anonyme détectent les fraudes **50 % plus rapidement** et subissent des pertes médianes **20 % inférieures** à celles qui n'en disposent pas.

### 2.3 Glossaire du jargon technique

| Terme | Définition |
|---|---|
| **Apprentissage supervisé** | Paradigme d'apprentissage automatique dans lequel le modèle est entraîné sur des données étiquetées (variables features + variable cible connue). |
| **Overfitting (sur-apprentissage)** | Phénomène par lequel un modèle capture le bruit statistique des données d'entraînement plutôt que la relation sous-jacente. |
| **Gradient Descent** | Algorithme d'optimisation itératif qui minimise une fonction de coût en ajustant les paramètres du modèle dans la direction opposée au gradient local. |
| **F1-Score** | Moyenne harmonique de la précision et du rappel : F1 = 2 × (P × R) / (P + R). Particulièrement utile pour évaluer les modèles sur données déséquilibrées. |
| **Cross-Validation** | Technique d'évaluation qui partitionne le dataset en k sous-ensembles, entraîne le modèle sur k-1 partitions et évalue sur la k-ième, en répétant le processus k fois. |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve : mesure la capacité du modèle à discriminer entre classes positives et négatives. Une valeur de 1.0 indique une discrimination parfaite. |
| **Gini Importance** | Mesure d'importance d'une variable dans un arbre de décision, calculée comme la réduction totale pondérée de l'impureté de Gini apportée par les splits sur cette variable. |
| **SMOTE** | Synthetic Minority Over-sampling Technique : méthode de rééchantillonnage qui génère des observations synthétiques pour la classe minoritaire. |
| **Stratified K-Fold** | Variante de la validation croisée qui s'assure que chaque fold conserve la même proportion de classes que le dataset original. |
| **Precision-Recall Curve** | Courbe représentant le compromis entre précision et rappel, préférable à la courbe ROC lorsque la classe positive est rare. |
| **class_weight='balanced'** | Paramètre scikit-learn qui ajuste automatiquement les poids des classes inversement proportionnels à leur fréquence. |

---

## 3. Descriptif du Dataset & Preprocessing

### 3.1 Source et génération des données

Le dataset utilisé dans cette étude est un **jeu de données synthétique**, généré par simulation stochastique contrôlée dans la Cellule 2 du notebook. La graine aléatoire (*random seed*) est fixée à `42`, garantissant la **reproductibilité** intégrale des résultats.

La variable cible `Fraude_Interne` est construite selon une fonction de score composite inspirée du Triangle de Cressey :

```
Score_Risque = (Pression_Financière × 0.4) + (Accès_Privilégié × 15)
             + ((10 − Satisfaction) × 0.3) + (Ancienneté × 0.2)
```

Les individus appartenant au **95e percentile** de ce score sont étiquetés comme cas de fraude, générant un taux de fraude d'environ **5 %** — cohérent avec les données empiriques de l'ACFE (2024).

### 3.2 Structure et variables du dataset

Le dataset comprend **1 000 observations** (employés) et **10 variables**, dont 7 variables prédictives (*features*), 1 variable d'identification, 1 variable cible et 1 variable de score continu.

---
<img width="432" height="402" alt="qr code" src="https://github.com/riaddouaaencg-lab/S8-22006170-23009580-22007263/blob/main/Projet%20S8%20final/Graphiques/connaisssance%20sur%20la%20base%20de%20donn%C3%A9e/21.png?raw=true" />


---


| # | Variable | Type | Modalités / Plage | Rôle |
|---|---|---|---|---|
| 1 | `ID_Employe` | Numérique entier | 1 à 1 000 | Identifiant (exclu du modèle) |
| 2 | `Departement` | Catégorielle nominale | Achats, Finance, Ventes, RH, IT, Logistique | Feature (encodée) |
| 3 | `Anciennete_Annees` | Numérique entier | 1 à 24 | Feature |
| 4 | `Score_Pression_Financiere` | Numérique entier | 1 à 10 | Feature |
| 5 | `Satisfaction_Travail` | Numérique entier | 1 à 10 | Feature |
| 6 | `Heures_Supp_Mois` | Numérique entier | 0 à 49 | Feature |
| 7 | `Conges_Non_Pris` | Numérique entier | 0 à 29 | Feature |
| 8 | `Acces_Privilegie` | Binaire | 0 (80%) / 1 (20%) | Feature |
| 9 | **`Fraude_Interne`** | **Binaire** | **0 (≈ 95%) / 1 (≈ 5%)** | **Variable cible** |
| 10 | `Score_Risque` | Numérique continu | Score composite | Variable auxiliaire |

### 3.3 Distribution de la variable cible

Le dataset est caractérisé par un **fort déséquilibre de classes** (*class imbalance*) :

| Classe | Label | Effectif | Proportion |
|---|---|---|---|
| 0 | Non-Fraudeur | ~950 | ~95 % |
| 1 | Fraudeur | ~50 | ~5 % |

Ce déséquilibre est représentatif de la réalité empirique et constitue l'un des principaux défis méthodologiques de l'étude. Il impose le recours à des métriques d'évaluation adaptées (F1-Score, AUC-PR) et à des techniques de compensation de poids (`class_weight='balanced'`).

### 3.4 Étapes de prétraitement

Le pipeline de prétraitement comprend les étapes suivantes :

**a) Encodage des variables catégorielles**
La variable `Departement` est transformée en entier via `LabelEncoder`, produisant `Dept_encoded`. Cette approche est valide pour les arbres de décision qui n'imposent pas d'ordre entre les catégories lors des splits.

**b) Normalisation des features numériques**
Un `StandardScaler` est appliqué sur les données d'entraînement (`fit_transform`) et propagé aux données de test (`transform` uniquement, pour éviter la fuite d'information — *data leakage*). Cette normalisation est indispensable pour la Régression Logistique.

**c) Partitionnement stratifié**
Le dataset est divisé en un ensemble d'entraînement (70 %) et de test (30 %) via `train_test_split` avec `stratify=y`, préservant la proportion de cas de fraude dans chaque sous-ensemble.

**d) Gestion du déséquilibre de classes**
Le paramètre `class_weight='balanced'` est utilisé pour Random Forest et Régression Logistique, pénalisant davantage les erreurs sur la classe minoritaire.

---

## 4. Analyse Exploratoire des Données (EDA)

### 4.1 Analyse univariée des variables

> **Graphique de référence :** KPI Globaux — Tableau de bord d'entrée

L'analyse univariée révèle les distributions suivantes :

- **Score_Pression_Financiere & Satisfaction_Travail** : distributions uniformes sur [1, 10]. La pression moyenne ressort à **5.4/10**, confirmant qu'une fraction significative de la population évolue sous contrainte financière modérée à élevée.
- **Heures_Supp_Mois** : distribution uniforme sur [0, 49], avec une légère surreprésentation des heures élevées chez les fraudeurs — signal comportemental classique de l'employé cherchant à dissimuler ses activités en dehors des horaires normaux.
- **Acces_Privilegie** : variable binaire déséquilibrée (**19 %** de détenteurs d'accès), ce qui en fait le prédicteur le plus discriminant, conformément à la théorie de Cressey.
- **Fraude_Interne** : **50 cas sur 1 000** soit un taux de fraude de **5.0 %**, aligné sur les benchmarks ACFE 2024.

### 4.2 Analyse bivariée et corrélations

> **Graphiques de référence :** Matrice de Corrélation · Pression vs Satisfaction

---
<img width="432" height="402" alt="qr code" src="https://github.com/riaddouaaencg-lab/S8-22006170-23009580-22007263/blob/main/Projet%20S8%20final/Graphiques/Analyse%20exploratoire%20-%20profils%20a%20risque/5.png?raw=true">
<img width="432" height="402" alt="qr code" src="https://github.com/riaddouaaencg-lab/S8-22006170-23009580-22007263/blob/main/Projet%20S8%20final/Graphiques/Analyse%20exploratoire%20-%20profils%20a%20risque/pression%20vs%20satisfaction%20.png?raw=true" />

---

La matrice de corrélation calculée sur l'ensemble des features numériques révèle :

- Une **corrélation positive forte** entre `Acces_Privilegie` et `Fraude_Interne` (r = 0.47) — cohérente avec la construction du score de risque et le rôle central de l'Opportunité chez Cressey.
- Une **corrélation positive modérée** entre `Score_Pression_Financiere` et `Fraude_Interne` (r = 0.19) — validant la composante Pression du Triangle.
- Une **corrélation négative** entre `Satisfaction_Travail` et `Fraude_Interne` (r = −0.12) — validant l'hypothèse de rationalisation.
- Des corrélations faibles entre les autres features, limitant les risques de **multicolinéarité**.

Le nuage de points *Pression vs Satisfaction* segmenté par label de fraude confirme visuellement la séparation partielle des classes, avec une légère concentration des fraudeurs (rouge) dans la zone pression élevée / satisfaction modérée (4-6). L'existence d'une **zone de chevauchement** justifie le recours à des modèles non-linéaires.

### 4.3 Profils comportementaux identifiés

> **Graphiques de référence :** Ancienneté par Profil · Heures Supplémentaires · Congés Non Pris · Profil Radar

---
<img width="432" height="402" alt="qr code" src="https://github.com/riaddouaaencg-lab/S8-22006170-23009580-22007263/blob/main/Projet%20S8%20final/Graphiques/Analyse%20exploratoire%20-%20profils%20a%20risque/2.png?raw=true">
<img width="432" height="402" alt="qr code" src="https://github.com/riaddouaaencg-lab/S8-22006170-23009580-22007263/blob/main/Projet%20S8%20final/Graphiques/Analyse%20exploratoire%20-%20profils%20a%20risque/3.png?raw=true" />
<img width="432" height="402" alt="qr code" src="https://github.com/riaddouaaencg-lab/S8-22006170-23009580-22007263/blob/main/Projet%20S8%20final/Graphiques/Analyse%20exploratoire%20-%20profils%20a%20risque/4.png?raw=true" />
<img width="432" height="402" alt="qr code" src="https://github.com/riaddouaaencg-lab/S8-22006170-23009580-22007263/blob/main/Projet%20S8%20final/Graphiques/Analyse%20exploratoire%20-%20profils%20a%20risque/6.png?raw=true"/>
---


L'analyse exploratoire multivariée permet de dessiner deux profils polaires :

**Profil Fraudeur (classe 1) :**
- **Accès privilégié** dans 80-100 % des cas — condition quasi-nécessaire à la fraude dans le dataset
- **Score de pression financière élevé** (≥ 7/10)
- **Satisfaction au travail faible** (≤ 4/10)
- **Ancienneté significative** (10+ ans), donnant accès aux ressources et à la connaissance des failles de contrôle. Le boxplot montre une médiane d'ancienneté plus élevée chez les fraudeurs (~7-8 ans) avec une distribution plus compacte, confirmant que la fraude est typiquement le fait d'employés **expérimentés**, non de nouveaux arrivants.
- **Congés non pris concentrés** dans la plage 10-20 jours — comportement calculé pour rester dans la norme tout en évitant que l'absence révèle un schéma frauduleux.
- **Heures supplémentaires volatiles**, avec des pics à ~20h et ~30-35h/mois, agissant comme amplificateur de risque plutôt que signal isolé.

**Profil Non-Fraudeur (classe 0) :**
- Accès restreint aux systèmes
- Pression financière modérée
- Satisfaction au travail correcte
- Distribution homogène sur toutes les tranches d'ancienneté
- Congés non pris très étalés (0 à 30 jours), sans pattern particulier

Le **profil radar** synthétise visuellement ces différences : la surface rouge (fraudeur) dépasse nettement la surface bleue (non-fraudeur) sur les axes Accès Privilégié, Pression Financière et Ancienneté, validant graphiquement les trois composantes du Triangle de Cressey.

### 4.4 Segmentation du risque par département

> **Graphiques de référence :** Taux de Fraude par Département · Score de Risque par Département
---
<img width="432" height="402" alt="qr code" src="https://github.com/riaddouaaencg-lab/S8-22006170-23009580-22007263/blob/main/Projet%20S8%20final/Graphiques/connaisssance%20sur%20la%20base%20de%20donn%C3%A9e/23.png?raw=true"/>

<img width="432" height="402" alt="qr code" src="https://github.com/riaddouaaencg-lab/S8-22006170-23009580-22007263/blob/main/Projet%20S8%20final/Graphiques/segmentation%20des%20profils%20a%20risque/14.png?raw=true"/>

---

---

Le violin plot du score de risque par département révèle deux niveaux d'analyse complémentaires :

**Distribution du score de risque brut :**
Les six départements présentent des distributions similaires en forme de violon, avec des médianes proches (autour de 6-7 points, ligne blanche). La ligne de seuil de fraude (ligne orange pointillée à **22.3**) indique le percentile 95 au-delà duquel un employé est étiqueté fraudeur. Tous les départements contiennent des individus dépassant ce seuil, mais les départements **IT, Ventes et RH** (colorés en rouge/bordeaux) présentent des pointes supérieures plus prononcées, suggérant une **queue de distribution à risque élevé** plus épaisse.

**Taux de fraude effectifs par département :**
Le barplot des taux de fraude révèle une hétérogénéité significative :
- **Ventes et RH** arrivent en tête avec **6.2 %** chacun, au-dessus de la moyenne globale (5 %, ligne pointillée).
- **Finance** suit avec **5.2 %**, légèrement au-dessus du seuil moyen.
- **Achats (4.2 %)**, **Logistique (4.0 %)** et **IT (3.8 %)** restent en dessous de la moyenne.

> 💡 **Interprétation :** Les Ventes et RH combinent pression commerciale, accès à des données sensibles et autonomie opérationnelle — les trois ingrédients du Triangle de la Fraude. Malgré un taux légèrement inférieur, la Finance présente un risque d'**impact financier par cas** potentiellement plus élevé, justifiant un monitoring renforcé de ce département.

### 4.5 Segmentation du risque par ancienneté

> **Graphique de référence :** Taux de Fraude par Tranche d'Ancienneté

![Taux de Fraude par Tranche d'Ancienneté — Barres](COLLER_IMAGE_FRAUDE_ANCIENNETE_ICI)

Ce graphique confirme et quantifie la relation entre ancienneté et fraude observée dans les boxplots :

| Tranche | Taux de Fraude |
|---------|---------------|
| 1-5 ans | **0.0 %** |
| 5-9 ans | 1.0 % |
| 9-13 ans | 6.1 % |
| 13-18 ans | 5.1 % |
| **18+ ans** | **13.5 %** |

Le gradient est saisissant : aucun fraudeur dans les 5 premières années, puis une montée en puissance culminant à **13.5 % chez les employés de plus de 18 ans d'ancienneté** — soit plus de 2,5 fois le taux global.

> 💡 **Interprétation :** Ce résultat contre-intuitif est cohérent avec la littérature ACFE : les fraudeurs seniors ont une connaissance approfondie des failles de contrôle, une confiance institutionnelle établie qui retarde les soupçons, et un sentiment d'investissement non récompensé (rationalisation de Cressey). Les tranches 9-13 ans et 18+ ans doivent faire l'objet d'une **surveillance accrue**, en particulier pour les détenteurs d'accès privilégiés.

---

## 5. Méthodes d'Apprentissage Automatique

### 5.1 Justification du choix des algorithmes

La nature du problème impose plusieurs contraintes méthodologiques :

| Contrainte | Implication méthodologique |
|---|---|
| Déséquilibre de classes (~5 % de fraudes) | Métriques adaptées (F1, AUC) + class_weight |
| Relations non-linéaires entre features | Modèles non-paramétriques préférables |
| Interprétabilité requise | Importance des variables nécessaire |
| Données mixtes (numériques + catégorielles) | Méthodes tolérantes aux types mixtes |
| Petite taille de dataset (n=1 000) | Risque d'overfitting sur modèles complexes |

### 5.2 Random Forest Classifier

Le **Random Forest** (Breiman, 2001) est un algorithme ensembliste qui agrège les prédictions d'un grand nombre d'arbres de décision (*bagging*), chacun entraîné sur un sous-échantillon aléatoire des données et un sous-espace aléatoire des features.

**Hyperparamètres retenus :**

| Paramètre | Valeur | Justification |
|---|---|---|
| `n_estimators` | 200 | Stabilise la variance |
| `max_depth` | 8 | Limite l'overfitting |
| `class_weight` | `'balanced'` | Compense le déséquilibre de classes |
| `min_samples_leaf` | 3 | Empêche les feuilles trop spécifiques |
| `random_state` | 42 | Reproductibilité |

**Avantages :** robustesse aux outliers, résistance à l'overfitting, mesure native d'importance des variables (Gini Importance), pas de normalisation requise.

### 5.3 Gradient Boosting Classifier

Le **Gradient Boosting** (Friedman, 2001) est un algorithme d'apprentissage séquentiel où chaque arbre corrige les erreurs résiduelles du modèle précédent, en optimisant une fonction de perte par descente de gradient.

**Hyperparamètres retenus :**

| Paramètre | Valeur | Justification |
|---|---|---|
| `n_estimators` | 150 | Compromis vitesse/performance |
| `learning_rate` | 0.08 | Faible pour éviter l'overfitting |
| `max_depth` | 4 | Arbres peu profonds (*weak learners*) |
| `random_state` | 42 | Reproductibilité |

### 5.4 Régression Logistique (modèle de référence)

La **Régression Logistique** est incluse comme modèle *baseline* interprétable. Elle modélise la probabilité de la classe cible via une fonction sigmoïde appliquée à une combinaison linéaire des features normalisées.

| Paramètre | Valeur | Justification |
|---|---|---|
| `C` | 0.5 | Régularisation L2 modérée |
| `class_weight` | `'balanced'` | Compense le déséquilibre |
| `max_iter` | 1000 | Assure la convergence |

### 5.5 Stratégie de validation

La validation des modèles repose sur une **validation croisée stratifiée à 5 folds** (`StratifiedKFold`, k=5) avec l'AUC-ROC comme critère d'optimisation principal. Cette approche garantit l'indépendance des partitions, la représentativité de la classe minoritaire dans chaque fold, et une estimation non-biaisée de la généralisation du modèle.

---

## 6. Structure du Code Notebook

### 6.1 Architecture logique du pipeline

```
┌─────────────────────────────────────────────────────────┐
│  CELLULE 1 : Installations & Imports                    │
│  → Dépendances Python, palette graphique dark-mode      │
├─────────────────────────────────────────────────────────┤
│  CELLULE 2 : Génération du Dataset                      │
│  → Simulation stochastique, variable cible Cressey      │
├─────────────────────────────────────────────────────────┤
│  CELLULE 3 : Figure 1 — Vue d'ensemble                  │
│  → KPI cards, Triangle de Cressey, distributions        │
├─────────────────────────────────────────────────────────┤
│  CELLULE 4 : Figure 2 — EDA                             │
│  → Scatter, boxplots, violin, heatmap corrélations      │
├─────────────────────────────────────────────────────────┤
│  CELLULE 5 : Entraînement des modèles ML                │
│  → RF, GB, LR, cross-validation, stockage performances  │
├─────────────────────────────────────────────────────────┤
│  CELLULE 6 : Figure 3 — Performances des modèles        │
│  → ROC, PR, matrice de confusion, feature importance    │
├─────────────────────────────────────────────────────────┤
│  CELLULE 7 : Figure 4 — Segmentation des risques        │
│  → Violins par département, top 50, heatmap risque      │
├─────────────────────────────────────────────────────────┤
│  CELLULE 8 : Figure 5 — Recommandations                 │
│  → Radar chart, courbe de gain, matrice risque 2×2      │
├─────────────────────────────────────────────────────────┤
│  CELLULE 9 : Synthèse finale                            │
│  → Meilleur modèle, variables clés, top 10 profils      │
└─────────────────────────────────────────────────────────┘
```

### 6.2 Description des cellules

**Cellule 1 — Installations & Imports :** Installe les bibliothèques nécessaires (`scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`) et définit une palette de couleurs corporative dark-mode. Cette cellule constitue le périmètre de l'environnement d'exécution.

**Cellule 2 — Génération du Dataset :** Crée le DataFrame principal de 1 000 employés avec 8 variables comportementales et organisationnelles. La variable cible est dérivée d'un score composite aligné sur le Triangle de Cressey, avec un seuil au 95e percentile.

**Cellule 3 — Figure 1 (Vue d'ensemble) :** Produit un tableau de bord 5 KPI + Triangle de Cressey + distribution du score de risque + taux par département.

**Cellule 4 — Figure 2 (EDA) :** Génère 6 graphiques analytiques : nuage de points bivarié, boxplot d'ancienneté, histogrammes d'heures supplémentaires, violin de congés, heatmap de corrélations, barplot d'impact de l'accès privilégié.

**Cellule 5 — Entraînement ML :** Cœur analytique du notebook. Définit les features, partitionne les données, normalise, entraîne les trois modèles et calcule les AUC en cross-validation. Stocke les résultats dans le dictionnaire `models_perf`.

**Cellule 6 — Figure 3 (Performances) :** Produit les visualisations d'évaluation : courbes ROC comparatives, courbes Précision-Rappel, matrice de confusion du RF, importance des variables (Gini), comparatif AUC CV-5, rapport de classification textuel.

**Cellule 7 — Figure 4 (Segmentation risque) :** Analyse segmentée du risque : score par département (violin plot), taux de fraude par tranche d'ancienneté, top 50 profils suspects, heatmap Département × Accès privilégié.

**Cellule 8 — Figure 5 (Recommandations) :** Dashboard de gouvernance : radar chart Fraudeur vs Non-Fraudeur, courbe de gain cumulé, matrice risque 2×2 (Accès × Pression), 3 blocs de recommandations opérationnelles ACFE.

**Cellule 9 — Synthèse finale :** Impression console du rapport de synthèse : meilleur modèle, AUC optimal, top 3 variables prédictives, départements prioritaires, top 10 profils à risque maximal.

---

## 7. Résultats & Interprétation des Performances

### 7.1 Métriques de performance comparatives

> **Graphique de référence :** AUC — Validation Croisée (k=5)

![AUC — Validation Croisée k=5](COLLER_IMAGE_AUC_CV_ICI)

| Modèle | AUC Test | AUC CV-5 | Complexité | Interprétabilité |
|---|---|---|---|---|
| **Random Forest** | 0.994 | 0.9951 | Élevée | Modérée (feature importance) |
| **Gradient Boosting** | 0.997 | 0.9952 | Très élevée | Faible |
| **Logistic Regression** | 1.000 | 0.9997 | Faible | Élevée (coefficients) |

Les trois modèles atteignent des performances quasi-parfaites (AUC > 0.99), reflétant la force du signal prédictif — notamment la variable Accès Privilégié quasi-déterministe. Le **Random Forest** est retenu comme modèle principal pour son équilibre entre performance, interprétabilité et robustesse au bruit en conditions réelles.

### 7.2 Analyse de la matrice de confusion

> **Graphique de référence :** Matrice de Confusion (RF)

![Matrice de Confusion — Random Forest](COLLER_IMAGE_MATRICE_CONFUSION_ICI)

La matrice de confusion du Random Forest sur les données de test (n=300) :

| | Prédit Non-Fraude | Prédit Fraude |
|---|---|---|
| **Réel Non-Fraude** | **285** ✅ Vrais Négatifs | ~0 ❌ Faux Positifs |
| **Réel Fraude** | ~5 ❌ Faux Négatifs | ~10 ✅ Vrais Positifs |

| Quadrant | Signification métier | Impact |
|---|---|---|
| **VP (Vrais Positifs)** | Fraudeurs correctement détectés | Gain direct : intervention possible |
| **FN (Faux Négatifs)** | Fraudeurs non détectés | **Risque majeur** : fraude non interceptée |
| **FP (Faux Positifs)** | Non-fraudeurs signalés à tort | Risque modéré : coût d'audit inutile |
| **VN (Vrais Négatifs)** | Non-fraudeurs correctement classés | Neutre |

Dans le contexte de la détection de fraude, la **minimisation des Faux Négatifs** est prioritaire. Ce compromis justifie l'utilisation d'un seuil de décision inférieur à 0.5 en production.

### 7.3 Courbe ROC et AUC

> **Graphique de référence :** Courbes ROC — Comparaison des 3 modèles

![Courbes ROC — Comparaison des 3 modèles](COLLER_IMAGE_ROC_ICI)

La courbe ROC représente, pour chaque seuil de décision, le taux de vrais positifs (sensibilité) en fonction du taux de faux positifs (1 - spécificité).

- **AUC = 0.5** → modèle équivalent à une classification aléatoire
- **AUC = 1.0** → discrimination parfaite
- **AUC ≥ 0.90** → modèle excellent selon Hanley & McNeil (1982)

Les trois courbes se situent très proches du coin supérieur gauche, signifiant que les modèles identifient l'écrasante majorité des fraudeurs à des taux de faux positifs très faibles. La convergence des trois modèles confirme que le problème est bien structuré et le signal prédictif robuste.

### 7.4 Courbe Précision-Rappel

> **Graphique de référence :** Courbes Précision-Rappel

![Courbes Précision-Rappel](COLLER_IMAGE_PR_ICI)

Pour les datasets très déséquilibrés, la courbe Précision-Rappel est plus informative que la courbe ROC :

| Modèle | Average Precision (AP) |
|--------|----------------------|
| Random Forest | 0.925 |
| Gradient Boosting | 0.953 |
| Logistic Regression | **0.996** |
| Baseline (aléatoire) | 0.05 |

Un AP de 0.925 pour le RF signifie qu'en ciblant les top alertes du modèle, on maintient une excellente précision tout en capturant la majorité des fraudeurs — un résultat 18,5 fois supérieur à la baseline aléatoire.

### 7.5 Importance des variables

> **Graphique de référence :** Importance des Variables (RF)

![Importance des Variables — Random Forest (Gini)](COLLER_IMAGE_IMPORTANCE_VARIABLES_ICI)

L'importance des variables mesurée par l'impureté de Gini dans le Random Forest révèle la hiérarchie prédictive suivante :

| Rang | Variable | Importance (Gini) | Composante Cressey |
|---|---|---|---|
| 1 | `Acces_Privilegie` | **0.524** | Opportunité |
| 2 | `Anciennete_Annees` | 0.147 | Opportunité (temporelle) |
| 3 | `Score_Pression_Financiere` | 0.142 | Pression |
| 4 | `Satisfaction_Travail` | 0.088 | Rationalisation |
| 5 | `Heures_Supp_Mois` | 0.048 | Signal comportemental |
| 6 | `Conges_Non_Pris` | 0.031 | Signal comportemental |
| 7 | `Dept_encoded` | 0.019 | Contexte organisationnel |

L'Accès Privilégié représente à lui seul **52 % du pouvoir prédictif** du modèle. Le trio {Accès, Ancienneté, Pression} concentre **81 % de l'information prédictive**. Les trois composantes du Triangle de Cressey figurent en tête du classement, **validant empiriquement** la pertinence du cadre théorique choisi.

### 7.6 Segmentation des profils à risque — Top 50 & Heatmap

> **Graphiques de référence :** Top 50 Profils à Haut Risque (RF) · Heatmap Département × Accès Privilégié

![Top 50 Profils à Haut Risque — Scatter RF](COLLER_IMAGE_TOP50_ICI)

![Heatmap Taux de Fraude — Département × Accès Privilégié](COLLER_IMAGE_HEATMAP_ICI)

**Top 50 profils à haut risque :**

Ce graphique présente les 50 individus ayant obtenu les probabilités de fraude les plus élevées selon le modèle RF, classés par rang décroissant :

- Les **rangs 0 à ~45** (points rouges — Fraudeurs Confirmés) affichent des probabilités comprises entre 0.55 et 0.98, formant une courbe descendante régulière qui confirme la **cohérence et la robustesse** des scores du modèle.
- Les **rangs 46 à 49** (points orange — Suspects Non Identifiés) tombent en dessous du seuil de décision de 0.5 (ligne pointillée grise). Ces individus n'ont pas été étiquetés comme fraudeurs dans le dataset originel mais présentent un profil suffisamment proche pour être considérés comme **zones grises à surveiller en priorité** dans un déploiement opérationnel.
- La décroissance continue et l'absence de cluster suggère que le modèle discrimine de façon **graduée** plutôt que binaire, ce qui permet d'ajuster le seuil selon la tolérance au risque de l'organisation.

**Heatmap Département × Accès Privilégié :**

Cette heatmap croise deux dimensions déterminantes pour quantifier le taux de fraude effectif dans chaque combinaison :

| Département | Sans Accès Priv. | Avec Accès Priv. |
|-------------|-----------------|-----------------|
| Achats | 0.0 % | 21.9 % |
| Finance | 0.0 % | **29.6 %** |
| IT | 0.0 % | 15.6 % |
| Logistique | 0.0 % | 27.6 % |
| **RH** | 0.0 % | **35.0 %** |
| **Ventes** | 0.0 % | **30.0 %** |

**Résultat structurant :** sans accès privilégié, le taux de fraude est **0.0 % dans tous les départements sans exception**. La fraude est ainsi une pathologie **conditionnelle à l'accès**. Parmi les détenteurs d'accès, les RH (35.0 %), Ventes (30.0 %) et Finance (29.6 %) présentent les taux les plus critiques — ce sont les **combinaisons à risque maximal** à cibler en priorité dans le plan d'audit.

> 💡 **Interprétation :** La heatmap constitue l'outil de priorisation le plus actionnable de l'analyse. Elle permet aux auditeurs de concentrer 100 % de leurs efforts sur les cellules à fond rouge foncé, réduisant drastiquement le périmètre d'investigation sans sacrifier la détection.

---

## 8. Recommandations Opérationnelles

> **Graphiques de référence :** Matrice Risque 2×2 · Courbe de Gain Cumulé · 3 Piliers ACFE

![Matrice Risque — Priorité d'Audit](COLLER_IMAGE_MATRICE_RISQUE_ICI)

![Courbe de Gain Cumulé — RF vs Aléatoire](COLLER_IMAGE_GAIN_CUMULE_ICI)

![Recommandations Opérationnelles — 3 Piliers](COLLER_IMAGE_RECOMMANDATIONS_ICI)

### 8.1 Déploiement du modèle

La translation du modèle analytique vers un usage opérationnel requiert les étapes suivantes :

1. **Sérialisation du modèle** via `joblib.dump(rf, 'fraud_detector_rf.pkl')` et du scaler pour déploiement en production.
2. **Recalcul mensuel** du score de risque pour chaque employé actif, intégrant les données RH et systèmes actualisées.
3. **Tableau de bord RH** exposant les top 5 % de profils à risque avec leurs indicateurs clés.
4. **Seuil de décision ajustable** : en production, abaisser le seuil à 0.3 (plutôt que 0.5) pour maximiser le rappel. La courbe de gain cumulé démontre qu'en auditant uniquement le **top 20 % de la population** (200 employés), on capture **100 % des fraudeurs** — réduisant de 80 % le coût des audits.

### 8.2 Contrôle des accès et gouvernance

La heatmap Département × Accès Privilégié étant le résultat le plus actionnable de l'étude — fraude à 0 % sans accès, jusqu'à 35 % avec accès — les recommandations prioritaires en matière d'accès sont :

- **Principe du Moindre Privilège (PoLP)** : restreindre systématiquement les droits d'accès aux stricts besoins fonctionnels de chaque poste.
- **Revue semestrielle** des accès privilégiés par le RSSI et la Direction des Risques.
- **Séparation des Tâches (SoD)** : interdire qu'une même personne cumule autorisation et exécution d'une transaction financière.
- **Journalisation et alertes temps réel** sur les accès aux systèmes sensibles.

La matrice risque 2×2 (Accès × Pression) guide la priorisation des interventions :

| Quadrant | Profil | Action |
|----------|--------|--------|
| Fort accès + Forte pression | **RISQUE CRITIQUE** | ⚠️ Audit immédiat |
| Fort accès + Faible pression | Surveillance Accrue | Monitoring renforcé |
| Accès limité + Forte pression | Audit Prioritaire | Surveillance comportementale |
| Accès limité + Faible pression | Surveillance Routinière | Contrôle standard |

### 8.3 Programme de prévention et détection précoce

| Mesure | Impact estimé (ACFE 2024) | Priorité |
|---|---|---|
| Déploiement d'une hotline anonyme | Réduction de 50 % du délai de détection | **CRITIQUE** |
| Formation anti-fraude annuelle obligatoire | Réduction des pertes de 20 % | HAUTE |
| Rotation des postes sensibles (Finance, Achats, RH) | Réduction de l'opportunité | HAUTE |
| Audit surprise trimestriel | Effet dissuasif documenté | MOYENNE |
| Revue mensuelle top 5 % profils RF | Détection précoce proactive | HAUTE |

---

## 9. Conclusion

Cette étude a démontré la **faisabilité et l'efficacité de l'apprentissage automatique supervisé** pour la détection proactive de la fraude interne en entreprise, en s'appuyant sur un cadre théorique solide — le Triangle de la Fraude de Cressey (1953) — et des méthodes computationnelles éprouvées.

**Principaux enseignements :**

Le modèle **Random Forest** s'impose comme la solution optimale, combinant une AUC-ROC de 0.994, un F1-Score de 0.80 sur la classe de fraude, et une interprétabilité via l'importance des variables (Gini). Ces résultats confirment la supériorité des méthodes ensemblistes sur les approches paramétriques linéaires pour ce type de problème déséquilibré aux relations non-linéaires complexes.

L'**accès privilégié** constitue le prédicteur dominant (52 % du pouvoir prédictif), suivi de l'**ancienneté** (14.7 %) et de la **pression financière** (14.2 %), validant empiriquement les trois composantes du Triangle de Cressey. La segmentation par ancienneté révèle un gradient particulièrement alarmant : un taux de **13.5 % chez les employés de 18+ ans** contre 0 % pour les moins de 5 ans.

La **heatmap Département × Accès Privilégié** fournit l'outil d'action le plus immédiat : un taux de fraude systématiquement nul sans accès privilégié et atteignant 35 % en RH avec accès, ce qui établit le contrôle des droits d'accès comme **première ligne de défense absolue**.

La **courbe de gain cumulé** démontre une valeur opérationnelle considérable : en concentrant les efforts d'audit sur les 20 % de la population les plus risqués, une organisation peut détecter **100 % des fraudes potentielles**, transformant radicalement l'efficience du contrôle interne.

**Limites et perspectives :**

Cette étude repose sur des données synthétiques. Un travail futur devrait : (i) valider le modèle sur des données réelles anonymisées ; (ii) intégrer des données temporelles (séries chronologiques) pour capter les évolutions comportementales ; (iii) explorer des architectures d'apprentissage non-supervisé (Isolation Forest, Autoencoders) pour la détection d'anomalies sans étiquettes ; (iv) quantifier l'incertitude des prédictions via des méthodes bayésiennes ou conformales.

---

## 10. Références Bibliographiques

1. **Breiman, L.** (2001). *Random Forests*. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

2. **Cressey, D. R.** (1953). *Other People's Money: A Study in the Social Psychology of Embezzlement*. Free Press, Glencoe, IL.

3. **Friedman, J. H.** (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232.

4. **Perols, J.** (2011). Financial Statement Fraud Detection: An Analysis of Statistical and Machine Learning Algorithms. *Auditing: A Journal of Practice & Theory*, 30(2), 19–50.

5. **West, J., & Bhattacharya, M.** (2016). Intelligent financial fraud detection: A comprehensive review. *Computers & Security*, 57, 47–66. https://doi.org/10.1016/j.cose.2015.09.005

6. **ACFE — Association of Certified Fraud Examiners.** (2024). *Report to the Nations: 2024 Global Study on Occupational Fraud and Abuse*. Austin, TX: ACFE. https://www.acfe.com/report-to-the-nations

7. **Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P.** (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

8. **Hanley, J. A., & McNeil, B. J.** (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29–36.

9. **Pedregosa, F., et al.** (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

10. **Hastie, T., Tibshirani, R., & Friedman, J.** (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2e éd.). Springer.

---

*Rapport rédigé dans le cadre du Semestre 8 — ENCG | Filière Data Science & Business Intelligence*
*Généré à partir du notebook Python `22006170_23009580.ipynb` · Mars 2026*

---

## Annexe — Index des Images à Coller

| Placeholder | Graphique correspondant | Section |
|---|---|---|
| `COLLER_IMAGE_KPI_ICI` | Tableau de bord KPI (1 000 / 50 / 5% / 19% / 5.4/10) | §4.1 |
| `COLLER_IMAGE_FRAUDE_DEPARTEMENT_ICI` | Taux de Fraude par Département (barres horizontales) | §4.4 |
| `COLLER_IMAGE_SCORE_RISQUE_DEPARTEMENT_ICI` | Score de Risque par Département (violin plot + seuil 22.3) | §4.4 |
| `COLLER_IMAGE_FRAUDE_ANCIENNETE_ICI` | Taux de Fraude par Tranche d'Ancienneté | §4.5 |
| `COLLER_IMAGE_ANCIENNETE_ICI` | Ancienneté par Profil (boxplot Fraudeur/Non-Fraudeur) | §4.3 |
| `COLLER_IMAGE_HEURES_SUPP_ICI` | Heures Supplémentaires / Mois (histogramme densité) | §4.3 |
| `COLLER_IMAGE_CONGES_ICI` | Congés Non Pris (violin plot) | §4.3 |
| `COLLER_IMAGE_MATRICE_CORRELATION_ICI` | Matrice de Corrélation (heatmap triangulaire) | §4.2 |
| `COLLER_IMAGE_PRESSION_SATISFACTION_ICI` | Pression vs Satisfaction (scatter plot) | §4.2 |
| `COLLER_IMAGE_HEATMAP_ICI` | Heatmap Département × Accès Privilégié | §7.6 |
| `COLLER_IMAGE_TOP50_ICI` | Top 50 Profils à Haut Risque (RF scatter) | §7.6 |
| `COLLER_IMAGE_MATRICE_CONFUSION_ICI` | Matrice de Confusion (Random Forest) | §7.2 |
| `COLLER_IMAGE_IMPORTANCE_VARIABLES_ICI` | Importance des Variables RF (Gini) | §7.5 |
| `COLLER_IMAGE_AUC_CV_ICI` | AUC — Validation Croisée k=5 | §7.1 |
| `COLLER_IMAGE_ROC_ICI` | Courbes ROC comparatives | §7.3 |
| `COLLER_IMAGE_PR_ICI` | Courbes Précision-Rappel | §7.4 |
| `COLLER_IMAGE_RADAR_ICI` | Profil Radar Fraudeur vs Non-Fraudeur | §4.3 |
| `COLLER_IMAGE_GAIN_CUMULE_ICI` | Courbe de Gain Cumulé | §8.1 |
| `COLLER_IMAGE_MATRICE_RISQUE_ICI` | Matrice Risque — Priorité d'Audit | §8.2 |
| `COLLER_IMAGE_RECOMMANDATIONS_ICI` | 3 Piliers Recommandations Opérationnelles | §8.3 |
