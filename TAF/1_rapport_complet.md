# MÉMOIRE : PRÉDICTION DE LA FRAUDE INTERNE PAR DES MODÈLES DE CLASSIFICATION

## 1. AVANT-PROPOS

Dans un environnement économique de plus en plus incertain, hybride et interconnecté, le **contrôle interne** s'impose comme un pilier fondamental pour assurer la pérennité et l'intégrité des organisations. L'essor rapide de la digitalisation a certes fluidifié les processus d'entreprise, mais il a également entraîné une multiplication et une sophistication sans précédent des risques de fraude. Selon l'Association of Certified Fraud Examiners (ACFE), les entreprises perdent chaque année près de 5 % de leur chiffre d'affaires en raison de fraudes internes, soulignant ainsi un impact économique mondial massif.

Face aux limites des approches d'audit traditionnelles, l'intelligence artificielle (IA) et l'apprentissage automatique (Machine Learning) émergent comme des solutions incontournables. Ils transforment la détection des anomalies en un processus proactif, continu et de plus en plus précis. C'est dans ce contexte de transformation digitale et sécuritaire que s'inscrit ce travail, justifiant le choix du sujet : **« Prédiction de la fraude interne par des modèles de classification »**.

---

## 2. INTRODUCTION GÉNÉRALE

### Problématique
La numérisation des systèmes d'information génère des volumes immenses de données, rendant obsolètes les méthodes manuelles d'échantillonnage de contrôle interne. Dès lors, la question centrale de cette étude est : **Comment l'intelligence artificielle peut-elle améliorer la détection et la prévention de la fraude interne dans les entreprises ?**

### Objectifs
- **Comprendre** les mécanismes sous-jacents de la fraude interne, en s'appuyant sur les théories existantes.
- **Identifier** les variables explicatives pertinentes (comportementales, organisationnelles).
- **Construire et évaluer** un modèle prédictif fiable basé sur des algorithmes de Machine Learning pour classifier les comportements à risque.

### Hypothèses
- L'IA permet une détection plus précoce, plus précise et à plus grande échelle que les méthodes traditionnelles.
- L'intégration de variables comportementales et liées aux accès renforce significativement les performances des modèles de prédiction de fraude.

### Méthodologie
Ce travail adopte une démarche quantitative basée sur le Machine Learning. Contrairement aux approches standard reposant uniquement sur Kaggle, nous utiliserons un dataset métier spécifiquement modélisé (généré) traduisant le *Triangle de la Fraude*, pour refléter au mieux les dynamiques réelles d'entreprise (pression financière, satisfaction, accès privilégiés).

---

## 3. SOMMAIRE

**PARTIE I : CADRE THÉORIQUE**
1. Contexte : Contrôle interne à l’ère de l’IA
2. Fraude interne : concepts et théories
3. Le Triangle de la Fraude de Donald Cressey
4. Intelligence artificielle et détection de fraude
5. Variables pertinentes de modélisation

**PARTIE II : ÉTUDE EMPIRIQUE**
1. Reformulation du problème et Données
2. Étapes de réalisation du modèle
3. Résultats et interprétation
4. Conclusion et Perspectives

---

# PARTIE I : CADRE THÉORIQUE

## 4. CONTEXTE : LE CONTRÔLE INTERNE À L'ÈRE DE L'IA

Le contrôle interne désigne l’ensemble des dispositifs appliqués par la direction pour s’assurer de la fiabilité de l'information financière, de la conformité aux lois et de la protection des actifs. Historiquement, ces systèmes s'appuient sur la séparation des tâches, des validations hiérarchiques et des audits périodiques. Toutefois, ces systèmes présentent des limites majeures : une détection souvent tardive (la fraude dure en moyenne 12 à 18 mois avant d'être découverte), une forte dépendance au jugement humain (biais) et une incapacité à analyser en temps réel la totalité des transactions (Big Data).

La transformation digitale a vu l'automatisation des processus. En réponse, l'IA apporte un changement de paradigme :
- **Détection proactive :** Les modèles apprennent le comportement « normal » et signalent les déviations en temps réel.
- **Réduction des coûts et préservation de la valeur :** En stoppant la fraude plus tôt, l'impact financier est drastiquement minimisé, augmentant la confiance des parties prenantes et améliorant la gouvernance.

## 5. FRAUDE INTERNE : DÉFINITION ET THÉORIES

La fraude interne, ou fraude occupationnelle, est définie comme l'utilisation d'une profession pour l'enrichissement personnel par le détournement délibéré ou la mauvaise application des ressources de l'organisation. L'ACFE regroupe ces actes en trois catégories (Typologie) :
- **Détournement d'actifs (Asset misappropriation) :** Vol de liquidités, fausses factures, le plus fréquent.
- **Corruption :** Pots-de-vin, conflits d'intérêts.
- **Fraude aux états financiers :** Moins fréquente mais financièrement la plus désastreuse.

## 6. LE TRIANGLE DE LA FRAUDE

Pour qu'un employé fraude, trois éléments doivent généralement converger, modélisés par le criminologue Donald Cressey :
1. **Pression (ou incitatif) :** Souvent financière (dettes, addictions) ou liée à des objectifs professionnels inatteignables.
2. **Opportunité :** Une faille dans le contrôle interne ou des droits informatiques trop vastes (accès privilégiés) permettant de commettre et dissimuler la fraude.
3. **Rationalisation :** La justification psychologique ("je l'emprunte", "je suis sous-payé").

## 7. INTELLIGENCE ARTIFICIELLE ET FRAUDE

L'IA traite les fraudes complexes grâce à des algorithmes de Classification (Régression logistique, Random Forest, XGBoost) capables de traiter des relations non-linéaires entre des centaines de variables. L'intérêt majeur réside dans la réduction drastique des faux positifs par rapport à des systèmes de règles "en dur", et dans la création de scores de risques pour un audit ciblé et efficient.

---

# PARTIE II : ÉTUDE PRATIQUE (MODÉLISATION)

## 8. REFORMULATION DU PROBLÈME ET DONNÉES
L'objectif est de construire un modèle de **classification binaire** avec la variable cible : *Fraude_Interne* (0 = Non, 1 = Oui).

**Données :** Les bases standard (Kaggle) portent souvent sur la fraude carte bancaire (transactions), qui diffère de la fraude interne (comportement RH/organisationnel). Nous avons donc généré une base de données de 1000 employés modélisant précisément les facteurs psychologiques et organisationnels (Ancienneté, Score Pression Financière, Satisfaction, Heures supp, Accès Privilégié).

## 9. ÉTAPES DE RÉALISATION

**Étape 1 : Compréhension et Analyse Exploratoire**
L'EDA (Exploratory Data Analysis) permet d'identifier les profils à risque. Par exemple, observer si une corrélation forte existe entre "Accès privilégié" et un "Score de Pression" élevé chez la minorité frauduleuse (les 5 % selon notre règle de seuil).

**Étape 2 & 3 : Prétraitement et Feature Engineering**
Encodage des départements, mise à l'échelle (scaling). L'avantage de notre jeu de données est que les caractéristiques (features) intègrent déjà les concepts du triangle de fraude.

**Étape 4, 5 & 6 : Séparation, Modèles et Entraînement**
Les données sont séparées en Train (80%) et Test (20%). Nous entraînerons trois modèles concurrents :
- La Régression Logistique (baseline, hautement interprétable).
- Le Random Forest (ensemble de sous-arbres, robuste et non-linéaire).
- Le XGBoost (Gradient boosting, performant sur les données tabulaires et les déséquilibres).

**Étape 7 & 8 : Évaluation et Gestion du Déséquilibre**
La matrice de confusion et le F1-Score seront prioritaires (l'accuracy étant trompeuse avec 95% de non-fraudeurs). En cas de rappel (Recall) trop faible sur la classe 1 (les fraudes), la technique SMOTE (Synthetic Minority Over-sampling Technique) sera utilisée pour over-sampler la classe des fraudeurs dans le Train set.

## 10. RÉSULTATS ATTENDUS

Le modèle (notamment le XGBoost combiné au SMOTE) devrait démontrer sa capacité à capter les signaux faibles générés par un mécontentement au travail combiné à une opportunité d'accès. Le graphique d'importance des variables (Feature Importance) validera l'hypothèse que l'Accès Privilégié et la Pression Financière sont des prédicteurs dominants.

## 11. CONCLUSION ET PERSPECTIVES

L'implémentation de modèles prédictifs améliore radicalement la gouvernance corporative, passant d'un audit rétrospectif à une surveillance proactive continue. Les limites résident dans la fiabilité de la collecte de la donnée RH (subjectivité de la satisfaction, respect de la confidentialité des données). Les futures recherches pourront inclure l'analyse textuelle des emails (NLP) pour évaluer la pression de la rationalisation plus finement, tout en respectant l'éthique de l'IA (RGPD).
