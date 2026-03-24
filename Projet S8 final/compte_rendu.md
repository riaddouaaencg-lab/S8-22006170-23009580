# Contrôle Interne & Gouvernance
# Prédiction de la Probabilité de Fraude Interne en Entreprise

> **Projet académique** | Audit, Contrôle Interne & Data Science  
> **Dataset** : 1 000 employés synthétiques | **Fraudes détectées** : 50 (5,0 %)  
> **Référence** : ACFE – Report to the Nations 2024 | Triangle de Cressey

---

## Avant-propos

Ce projet s'inscrit dans le cadre d'une formation en **audit, contrôle interne et gouvernance d'entreprise**. La fraude interne représente l'une des menaces les plus sous-estimées pour la santé financière et organisationnelle des entreprises. Pourtant, les méthodes de détection traditionnelles – audits périodiques, contrôles hiérarchiques, alertes manuelles – peinent à identifier les signaux faibles précurseurs d'un comportement frauduleux.

Ce projet explore comment la **data science et le machine learning** peuvent venir en appui des auditeurs internes pour prédire, anticiper et prévenir la fraude interne. Le choix de ce thème est motivé par :

- La montée en puissance des fraudes occupationnelles dans les organisations (ACFE, 2024)
- Le besoin croissant d'outils analytiques dans les fonctions d'audit et de conformité
- L'opportunité d'articuler une théorie criminologique établie (Triangle de Cressey) avec des techniques modernes de classification supervisée

---

## 1. Introduction Générale

### 1.1 Le phénomène de la fraude interne

La **fraude interne** (ou fraude occupationnelle) désigne tout acte intentionnel commis par un employé, un cadre ou un dirigeant visant à s'approprier illégalement des actifs de l'organisation ou à induire en erreur ses partenaires. Elle se distingue de la fraude externe par le fait que l'auteur bénéficie d'une position de confiance au sein de l'entité.

Selon l'**ACFE – Report to the Nations 2024**, les pertes mondiales liées à la fraude occupationnelle représentent **5 % du chiffre d'affaires** des entreprises, soit plusieurs milliers de milliards de dollars annuellement. La durée médiane d'une fraude avant détection est de **12 mois**, et la plupart des cas ne sont découverts que par accident ou via dénonciation.

### 1.2 Impact économique et organisationnel

| Indicateur | Statistique ACFE 2024 |
|---|---|
| Perte médiane par cas | 145 000 $ |
| Durée médiane avant détection | 12 mois |
| Part des fraudes inférieures à 100 000 $ | 58 % |
| Secteur le plus touché | Services financiers et gouvernements |
| Vecteur de détection n°1 | Dénonciation (43 % des cas) |

### 1.3 Le rôle du contrôle interne

Le contrôle interne constitue la première ligne de défense contre la fraude. Selon le cadre **COSO (Committee of Sponsoring Organizations)**, un système de contrôle interne efficace repose sur cinq composantes : environnement de contrôle, évaluation des risques, activités de contrôle, information et communication, pilotage. Ce projet vise à renforcer la composante "Évaluation des risques" grâce à un modèle prédictif.

---

## 2. Contexte du Projet

### 2.1 Augmentation des fraudes internes

Les rapports successifs de l'ACFE documentent une progression continue des cas de fraudes internes dans les organisations de toutes tailles. Les crises économiques (crise financière de 2008, pandémie de 2020) ont historiquement amplifié les comportements frauduleux en augmentant la pression financière sur les individus.

### 2.2 Enjeux de gouvernance et de conformité

La gouvernance d'entreprise impose aux conseils d'administration et aux comités d'audit de mettre en place des dispositifs robustes de détection. Réglementairement, la **loi Sarbanes-Oxley (SOX)** aux États-Unis, la **8e directive européenne** et les normes **ISA 240** (audit des états financiers) imposent une vigilance renforcée concernant le risque de fraude.

### 2.3 Le rôle croissant de l'intelligence artificielle

Les directions d'audit interne intègrent progressivement des outils d'**analyse de données (data analytics)**, de **détection d'anomalies** et d'apprentissage automatique pour dépasser les limites des échantillonnages statistiques classiques. Ce projet s'inscrit dans cette dynamique.

---

## 3. Intérêt et Importance du Thème

### 3.1 Un problème majeur pour les entreprises

La fraude interne est particulièrement insidieuse car elle provient de personnes bénéficiant déjà de la confiance de l'organisation. Elle génère :

- Des **pertes financières directes** (détournements, surfacturation, fraudes aux remboursements)
- Des **coûts indirects** importants : enquêtes, litiges, atteinte à la réputation, démotivation des équipes
- Une **dégradation de la gouvernance** lorsqu'elle implique des cadres supérieurs (tone at the top)

### 3.2 L'intérêt du machine learning dans la détection de fraude

Les approches par règles métier (règles de Benford, seuils d'anomalie) sont limitées car elles ne capturent pas les interactions complexes entre variables. Les modèles de machine learning présentent plusieurs avantages :

- **Capacité à traiter de grandes volumétries de données** sans fatigue cognitive
- **Détection de patterns non linéaires** et d'interactions entre variables
- **Scalabilité** : applicable à des milliers d'employés simultanément
- **Mise à jour continue** des modèles au fur et à mesure de nouvelles données

---

## 4. L'Intelligence Artificielle au Service de l'Audit

### 4.1 L'IA dans l'audit interne

Les **Institute of Internal Auditors (IIA)** recommandent l'intégration de l'IA dans les pratiques d'audit pour passer d'un audit par sondage à un **audit continu en temps réel**. Des outils comme des algorithmes de clustering, de classification supervisée et de NLP sont déjà utilisés par les Big Four et les grandes directions d'audit internes.

### 4.2 Détection automatisée des anomalies

La détection d'anomalies par machine learning permet d'identifier :
- Des **transactions inhabituelles** hors des patterns historiques
- Des **comportements atypiques** en termes d'accès aux systèmes
- Des **incohérences** entre les déclarations et les données réelles

### 4.3 Contribution de la data science à la gouvernance

La data science permet de **quantifier le risque de fraude** de manière objective et reproductible, en substituant un score probabiliste aux jugements subjectifs des auditeurs. Cela renforce la qualité des rapports d'audit et améliore la prise de décision des comités de gouvernance.

---

## 5. Problématique de Recherche

> **Question centrale :**  
> *« Comment les techniques de machine learning peuvent-elles contribuer à prédire et prévenir la fraude interne en entreprise à partir de données comportementales et organisationnelles ? »*

Cette problématique soulève des questions à l'intersection de l'audit, de la gouvernance et de la data science, domaines rarement articulés de façon aussi directe dans les pratiques académiques et professionnelles.

---

## 6. Objectifs du Projet

### 6.1 Objectif général

Développer un **modèle de classification supervisée** capable de prédire la probabilité de fraude interne pour chaque employé, à partir de données comportementales, organisationnelles et financières.

### 6.2 Objectifs spécifiques

1. Identifier les **facteurs de risque** associés statistiquement à la fraude interne
2. Construire et comparer plusieurs **algorithmes de classification** (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting)
3. Analyser les **variables explicatives** les plus déterminantes du risque de fraude
4. Formuler des **recommandations opérationnelles** pour les auditeurs et les responsables de la conformité

---

## 7. Questions de Recherche

1. Quels **facteurs organisationnels** (département, ancienneté, accès) favorisent la fraude interne ?
2. Comment les **variables comportementales** (pression financière, satisfaction, congés non pris) influencent-elles le risque de fraude ?
3. Les **modèles de machine learning** peuvent-ils dépasser les méthodes traditionnelles dans la détection de fraude interne ?
4. Quels **seuils de probabilité** sont pertinents pour déclencher une alerte audit ?

---

## 8. Revue de Littérature

### 8.1 La fraude interne en entreprise

**Définition.** La fraude interne est définie par l'ACFE comme : *"l'utilisation délibérée de l'emploi à des fins d'enrichissement personnel par le détournement des ressources ou des actifs de l'organisation."*

**Principales typologies** (ACFE 2024) :
| Type | Description | Part des cas |
|---|---|---|
| Détournement d'actifs | Vol, falsification de chèques, fraude aux remboursements | 86 % |
| Corruption | Conflits d'intérêts, pot-de-vin, appels d'offres truqués | 50 % |
| Fraude aux états financiers | Manipulation comptable, surfacturation | 9 % |

### 8.2 Le Triangle de la Fraude de Donald Cressey

Criminologue américain, **Donald Cressey** (1953) a théorisé que toute fraude naît de la convergence de trois éléments :

```
         PRESSION
        /         \
       /           \
RATIONALISATION — OPPORTUNITÉ
```

- **Pression** : besoin financier urgent, dettes, problèmes personnels → *Score_Pression_Financiere*
- **Opportunité** : accès aux actifs, faiblesse des contrôles → *Acces_Privilegie*, *Departement*
- **Rationalisation** : justification morale de l'acte ("je le mérite", "l'entreprise me doit bien ça") → *Satisfaction_Travail*, *Conges_Non_Pris*

Ce triangle constitue le **fondement théorique** de la variable cible et du score de risque utilisés dans ce projet.

### 8.3 Les profils d'employés à risque

Les recherches identifient plusieurs signaux d'alerte comportementaux :
- **Difficultés financières** : endettement, demandes d'avances fréquentes
- **Insatisfaction au travail** : sentiment d'injustice salariale, conflits hiérarchiques
- **Comportements inhabituels** : heures supplémentaires excessives, refus de prendre des congés
- **Accès élargi aux systèmes** : droits d'administration, accès non justifiés aux données financières

### 8.4 Le rôle du contrôle interne et de la gouvernance

La norme **ISA 240** impose aux auditeurs d'évaluer le risque de fraude et de mettre en place des procédures spécifiques. Le cadre **COSO ERM** (Enterprise Risk Management) intègre la fraude comme un risque majeur à cartographier et à mitiger via le contrôle interne.

### 8.5 Machine Learning et détection de fraude

Des études montrent que les algorithmes d'ensemble (Random Forest, Gradient Boosting) surpassent les modèles logistiques classiques dans la détection de fraudes financières (Bhattacharyya et al., 2011 ; dal Pozzolo et al., 2015). L'enjeu principal est la **gestion du déséquilibre de classes** (très peu de fraudeurs par rapport à l'ensemble des employés).

---

## 9. Présentation des Variables Explicatives

Conformément au Triangle de Cressey, les variables du dataset sont organisées en quatre dimensions :

### 9.1 Dimension Pression

| Variable | Description | Lien avec Cressey |
|---|---|---|
| `Score_Pression_Financiere` | Score de 1 à 10 (10 = pression maximale) | Pression (weight : ×0,4) |
| `Heures_Supp_Mois` | Nombre d'heures supplémentaires (0–50) | Indicateur de stress, pression indirecte |

### 9.2 Dimension Opportunité

| Variable | Description | Lien avec Cressey |
|---|---|---|
| `Acces_Privilegie` | 0 = accès standard, 1 = accès privilégié | Opportunité (weight : ×15) |
| `Departement` | Département de l'employé | Finance/Achats = risque plus élevé |

### 9.3 Dimension Rationalisation

| Variable | Description | Lien avec Cressey |
|---|---|---|
| `Satisfaction_Travail` | Score de 1 à 10 (1 = très insatisfait) | Rationalisation (weight : ×0,3 inversé) |
| `Conges_Non_Pris` | Jours de congés non pris (0–30) | Signal d'alerte comportemental |

### 9.4 Facteurs Organisationnels

| Variable | Description | Rôle |
|---|---|---|
| `Anciennete_Annees` | Années dans l'entreprise (1–24) | Accès croissant aux informations sensibles |
| `Dept_encoded` | Encodage numérique du département | Feature pour le modèle ML |

---

## 10. Méthodologie Générale de la Recherche

Ce projet adopte une **approche quantitative et expérimentale** articulée en deux temps :

1. **Phase théorique** : ancrage dans la littérature sur la fraude (Triangle de Cressey, ACFE) pour définir les variables et la logique de génération de la cible
2. **Phase pratique** : construction d'un pipeline complet de data science (génération, exploration, modélisation, évaluation) avec Python

---

# PARTIE PRATIQUE — Modèle de Prédiction de Fraude Interne

---

## Étape 1 : Définition du Problème de Classification

Le problème est formalisé comme un **problème de classification binaire supervisée** :

| Classe | Valeur | Signification |
|---|---|---|
| Non-Fraude | 0 | Employé sans comportement frauduleux détecté |
| Fraude | 1 | Employé identifié comme à haut risque de fraude |

L'objectif du modèle est d'estimer, pour chaque employé, la **probabilité P(Fraude = 1 | X)** où X représente l'ensemble des variables explicatives. Cette probabilité permet de **hiérarchiser le risque** et d'orienter les efforts de contrôle interne.

---

## Étape 2 : Création du Dataset Synthétique

Le dataset est généré en Python à l'aide des bibliothèques **NumPy** et **Pandas**, pour simuler les comportements de **1 000 employés** dans une organisation fictive.

```python
np.random.seed(42)
n_employes = 1000

data = {
    'ID_Employe': range(1, n_employes + 1),
    'Departement': np.random.choice(
        ['Achats', 'Finance', 'Ventes', 'RH', 'IT', 'Logistique'],
        n_employes, p=[0.15, 0.15, 0.25, 0.10, 0.15, 0.20]
    ),
    'Anciennete_Annees':         np.random.randint(1, 25, n_employes),
    'Score_Pression_Financiere': np.random.randint(1, 11, n_employes),
    'Satisfaction_Travail':      np.random.randint(1, 11, n_employes),
    'Heures_Supp_Mois':          np.random.randint(0, 50, n_employes),
    'Conges_Non_Pris':           np.random.randint(0, 30, n_employes),
    'Acces_Privilegie':          np.random.choice([0, 1], n_employes, p=[0.8, 0.2])
}
```

**Répartition par département** :
| Département | Proportion | Justification |
|---|---|---|
| Ventes | 25 % | Effectif le plus large |
| Logistique | 20 % | Accès aux flux physiques |
| Achats | 15 % | Exposition aux fournisseurs |
| Finance | 15 % | Accès aux données financières |
| IT | 15 % | Accès technique aux systèmes |
| RH | 10 % | Moins exposé aux actifs financiers directs |

**Résultat** : `Dataset : 1 000 employés | 50 fraudes (5,0 %)`

![Distribution de la variable cible](graphiques/G01_distribution_cible.png)

---

## Étape 3 : Construction de la Variable Cible (Fraude_Interne)

La variable cible est construite à partir d'un **score de risque** directement inspiré du Triangle de Cressey :

```python
score_risque = (
    (df['Score_Pression_Financiere'] * 0.4) +   # Pression
    (df['Acces_Privilegie']          * 15 ) +   # Opportunité (dominant)
    ((10 - df['Satisfaction_Travail']) * 0.3) + # Rationalisation (inversée)
    (df['Anciennete_Annees']          * 0.2)    # Facteur organisationnel
)
seuil = np.percentile(score_risque, 95)
df['Fraude_Interne'] = (score_risque >= seuil).astype(int)
```

### Pondération des facteurs et justification

| Facteur | Variable | Poids | Justification |
|---|---|---|---|
| Opportunité | `Acces_Privilegie` × 15 | **Dominant** | L'accès privilégié est le facteur le plus discriminant selon l'ACFE |
| Pression | `Score_Pression_Financiere` × 0,4 | Modéré | Facteur déclencheur classique de Cressey |
| Rationalisation | `(10 - Satisfaction_Travail)` × 0,3 | Modéré-faible | Insatisfaction comme facteur de rationalisation |
| Organisationnel | `Anciennete_Annees` × 0,2 | Faible | L'ancienneté amplifie les opportunités par accumulation d'accès |

**Seuil de classification** : 95e percentile du score de risque → seuls les **5 % d'employés les plus à risque** sont étiquetés comme fraudeurs, ce qui est cohérent avec les statistiques ACFE.

![Distribution du Score Risque](graphiques/G03_distribution_score_risque.png)

---

## Étape 4 : Analyse Exploratoire des Données (EDA)

### 4.1 Fraude par département

L'analyse montre que certains départements présentent un taux de fraude significativement plus élevé que la moyenne, notamment ceux combinant accès privilégié aux systèmes et pression financière.

![Fraude par département](graphiques/G02_fraude_par_departement.png)

### 4.2 Impact de l'accès privilégié

Le graphique suivant illustre clairement l'effet dominant de la variable `Acces_Privilegie`, confirmant son poids de ×15 dans le score de risque.

![Fraude et accès privilégié](graphiques/G06_fraude_acces_privilegie.png)

### 4.3 Distribution des variables par statut de fraude

Les boxplots comparent la distribution de chaque variable explicative entre fraudeurs et non-fraudeurs.

![Boxplots variables](graphiques/G04_boxplots_variables.png)

### 4.4 Matrice de corrélation

La heatmap révèle les corrélations entre les variables et avec la cible.

![Matrice de corrélation](graphiques/G05_heatmap_correlation.png)

### 4.5 Pression financière vs Score de risque

Le nuage de points illustre la relation entre la pression financière et le score de risque global, avec le seuil de fraude clairement visible.

![Scatter Pression vs Score Risque](graphiques/G07_scatter_pression_risque.png)

---

## Étape 5 : Préparation des Données

### 5.1 Encodage du département

```python
le = LabelEncoder()
df['Dept_encoded'] = le.fit_transform(df['Departement'])
```

### 5.2 Normalisation des variables

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

La normalisation est essentielle pour les algorithmes sensibles à l'échelle (Régression Logistique), et n'affecte pas les algorithmes basés sur les arbres (Random Forest, Gradient Boosting).

### 5.3 Séparation Train / Test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
```

| Ensemble | Taille | Fraudes |
|---|---|---|
| Entraînement | 800 employés (80 %) | 40 fraudeurs |
| Test | 200 employés (20 %) | 10 fraudeurs |

La stratification garantit que la proportion de fraudes est préservée dans les deux ensembles.

---

## Étape 6 : Construction et Évaluation des Modèles

### 6.1 Choix des algorithmes

Quatre algorithmes de classification ont été sélectionnés pour leurs complémentarités :

| Algorithme | Type | Avantage | Limite |
|---|---|---|---|
| **Logistic Regression** | Linéaire | Interprétable, rapide | Sous-performant si relations non linéaires |
| **Decision Tree** | Non paramétrique | Visualisable, intuitif | Surapprentissage si non élagué |
| **Random Forest** | Ensemble (bagging) | Robuste, précis | Moins interprétable |
| **Gradient Boosting** | Ensemble (boosting) | Meilleure précision | Sensible aux hyperparamètres |

### 6.2 Résultats des modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0,990** | **1,000** | 0,800 | 0,889 | **1,000** |
| Decision Tree | 0,985 | 0,889 | 0,800 | 0,842 | 0,945 |
| Random Forest | 0,970 | 1,000 | 0,400 | 0,571 | 0,998 |
| Gradient Boosting | 0,980 | 0,875 | 0,700 | 0,778 | 0,997 |

> **Interprétation** : La Régression Logistique atteint un AUC parfait de 1,000 car la variable `Acces_Privilegie` est fortement discriminante et la frontière de décision est quasi-linéaire dans ce dataset synthétique. En conditions réelles, le Random Forest ou le Gradient Boosting seraient préférés pour leur robustesse.

### 6.3 Comparaison des métriques

![Comparaison des modèles](graphiques/G08_comparaison_modeles.png)

### 6.4 Matrices de confusion

![Matrices de confusion](graphiques/G09_matrices_confusion.png)

**Lecture de la matrice de confusion** :
- **Vrais Positifs (VP)** : fraudeurs correctement identifiés → priorité pour l'auditeur
- **Faux Négatifs (FN)** : fraudeurs non détectés → risque résiduel
- **Faux Positifs (FP)** : employés non-fraudeurs signalés à tort → coût d'enquête inutile

### 6.5 Courbes ROC

Les courbes ROC illustrent la capacité discriminante de chaque modèle à toutes les valeurs de seuil.

![Courbes ROC](graphiques/G10_courbes_roc.png)

Un **AUC proche de 1** indique qu'un modèle sépare parfaitement les deux classes. Tous les modèles testés présentent des AUC supérieurs à 0,94.

### 6.6 Radar des performances

![Radar des performances](graphiques/G14_radar_performances.png)

---

## Étape 7 : Importance des Variables

### 7.1 Random Forest — Importance Gini

![Importance variables RF](graphiques/G11_feature_importance_rf.png)

### 7.2 Gradient Boosting — Importance

![Importance variables GB](graphiques/G12_feature_importance_gb.png)

**Analyse** : la variable `Acces_Privilegie` domine très largement l'importance des features, suivie de `Score_Pression_Financiere` et `Satisfaction_Travail`. Ce résultat est **cohérent avec le Triangle de Cressey** : l'opportunité (accès) est le facteur déclencheur principal que les contrôles internes doivent cibler en priorité.

---

## Étape 8 : Production de la Probabilité de Fraude

Le modèle produit pour chaque employé une **probabilité continue entre 0 et 1** :

![Distribution des probabilités](graphiques/G13_probabilites_fraude.png)

### Grille d'interprétation du risque

| Probabilité estimée | Niveau de risque | Action recommandée |
|---|---|---|
| P < 0,10 | **Faible** | Surveillance standard |
| 0,10 ≤ P < 0,50 | **Modéré** | Audit ciblé, entretien managérial |
| 0,50 ≤ P < 0,75 | **Élevé** | Audit approfondi, restriction d'accès temporaire |
| P ≥ 0,75 | **Critique** | Alerte immédiate, enquête formelle |

---

## Recommandations pour le Contrôle Interne

### Pour les auditeurs internes

1. **Intégrer le score de risque ML** dans le plan d'audit annuel pour prioriser les audits par entité et par individu
2. **Combiner les résultats du modèle** avec les révélations de témoins et les anomalies transactionnelles
3. **Surveiller en priorité** les employés ayant un accès privilégié aux systèmes financiers combiné à un score de pression élevé

### Pour les responsables du contrôle interne

1. **Restreindre les accès privilégiés** au strict nécessaire (principe du moindre privilège)
2. **Mettre en place une séparation des tâches** systématique dans les départements Finance et Achats
3. **Développer des indicateurs RH** de mal-être au travail pour anticiper la rationalisation
4. **Instaurer un programme de rotation** des postes à risque (3–5 ans maximum sur un même poste sensible)

### Pour les équipes de gouvernance

1. **Adopter un reporting trimestriel** des scores de risque fraude auprès du comité d'audit
2. **Définir des seuils d'alerte** déclenchant une procédure d'escalade formalisée
3. **Former les managers** à la reconnaissance des signaux d'alerte comportementaux
4. **Instaurer une culture de conformité** (tone at the top) valorisant la divulgation éthique

---

## Conclusion

Ce projet a démontré qu'il est possible de **prédire avec un haut degré de fiabilité la probabilité de fraude interne** à partir de données comportementales et organisationnelles, en ancrant la démarche dans la théorie criminologique du Triangle de Cressey.

Les principaux apports du projet :

- **Théorique** : articulation rigoureuse entre la littérature sur la fraude (ACFE, Cressey) et les techniques modernes de data science
- **Méthodologique** : construction d'un pipeline complet de classification (EDA → modélisation → évaluation → interprétation)
- **Pratique** : production d'un outil d'aide à la décision pour les auditeurs internes avec probabilités individuelles de fraude

Les limites à noter :
- Le dataset est **synthétique** : les résultats exceptionnels (AUC = 1,000) sont liés à la logique déterministe de génération de la cible
- En contexte réel, les relations sont plus complexes et le **déséquilibre de classes** nécessite des techniques d'oversampling (SMOTE)
- Le modèle nécessite une **validation externe** sur des données réelles avant tout déploiement opérationnel

---

## Références

- **ACFE** (2024). *Report to the Nations: Global Study on Occupational Fraud and Abuse*. Association of Certified Fraud Examiners.
- **Cressey, D.R.** (1953). *Other People's Money: A Study in the Social Psychology of Embezzlement*. Free Press.
- **COSO** (2013). *Internal Control — Integrated Framework*. Committee of Sponsoring Organizations.
- **Bhattacharyya, S., Jha, S., Tharakunnel, K., & Westland, J.C.** (2011). Data mining for credit card fraud: A comparative study. *Decision Support Systems*, 50(3), 602–613.
- **Dal Pozzolo, A., et al.** (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE Symposium Series on Computational Intelligence*.
- **IIA** (2023). *Global Technology Audit Guide: Auditing with Data Analytics*. Institute of Internal Auditors.
- **ISA 240** – Norme internationale d'audit : *Les obligations de l'auditeur concernant les fraudes*.

---

*Rapport généré dans le cadre d'un projet académique en Audit, Contrôle Interne & Gouvernance.*  
*Dataset de 1 000 employés synthétiques | 14 graphiques produits automatiquement en Python.*
