# RAPPORT DÉTAILLÉ : ANALYSE APPROFONDIE EXPERTE

## 1. STRATÉGIE ET CHOIX DES VARIABLES

Contrairement à l'utilisation d'une source générique issue de Kaggle focalisée sur des données transactionnelles (cartes bancaires), le présent travail utilise un jeu de données simulant de manière experte une base RH et organisationnelle d'entreprise. 

### Correspondance des variables avec le Triangle de Cressey
Le modèle théorique de Cressey est entièrement traduit dans l'ingénierie des caractéristiques (*Feature Engineering* et *Target Creation*) :
* **Pression :** Intégration de la variable `Score_Pression_Financiere`.
* **Opportunité :** Modélisée par la variable binaire `Acces_Privilegie`. Un individu ayant accès aux systèmes fondamentaux posera toujours un risque systémique plus élevé en termes d'opportunité d'agir.
* **Rationalisation (et insatisfaction) :** Approximée par `(10 - Satisfaction_Travail)`, considérant qu'un employé peu satisfait au travail rationnalise plus facilement l'acte ou le vol ("l'entreprise me le doit").

Le score final de risque est une pondération mathématique :
> `Risque = (Pression_Financiere * 0.4) + (Acces_Privilegie * 15) + (Insatisfaction * 0.3) + (Ancienneté * 0.2)`

## 2. MODÉLISATION EN ZONE DE DÉSÉQUILIBRE

### La problématique des données rares (Imbalanced Data)
Dans une entreprise saine, les fraudeurs avérés représentent naturellement une faible part de la population (ici artificiellement fixée au Top 5% de notre score de risque, soit 50 cas sur 1000 pour 950 cas normaux).

Si un modèle standard est entraîné dessus, il aura tendance à prédire "0" (Non-Fraude) dans 100% des cas, atteignant une *Accuracy* illusoire de 95%.

### L'approche SMOTE (Synthetic Minority Over-sampling Technique)
Pour remédier à la "naive classification" :
Nous générons des données synthétiques dans la classe minoritaire en utilisant l'algorithme des k-plus-proches voisins (k-NN) sur les fraudeurs existants dans l'ensemble de *Train*. Ceci permet au modèle d'apprentissage (notamment Random Forest ou XGBoost) de créer des règles de décision pour la classe minoritaire sans subir le sur-apprentissage (overfitting) qui se produirait via une simple duplication.

*NB : Il est critique de n'appliquer SMOTE que sur la partition d'entraînement (Train Set) et de conserver un ensemble de Test strict non modifié.*

## 3. INTERPRÉTATION ALGORITHMIQUE : POURQUOI XGBOOST OU RANDOM FOREST ?

Nous privilégions des approches ensemblistes basées sur des arbres de décision.
* **Non-linéarité :** Le passage à l'acte frauduleux n'est pas linéaire. Ce n'est pas parce que la pression augmente que la fraude augmente en ligne droite. C'est le croisement (Pression ALORS Accès = Vrai) qui déclenche le comportement. Les Random Forests gèrent naturellement ces interactions profondes.
* **Transparence avec le "Feature Importance" :** Dans le cadre d'un audit de contrôle interne, les algorithmes "boîtes noires" comme le Deep Learning sont souvent rejetés par la gouvernance. Les arbres de décision permettent d'isoler mathématiquement l'impureté de Gini pour attribuer l'importance d'une variable spécifique dans la déclinaison d'un risque.

## 4. BONUS GOUVERNANCE : EXPLOITATION DES RÉSULTATS POUR L'ENTREPRISE

L'objectif final de cette modélisation n'est pas uniquement scientifique. En entreprise, ces résultats ont des traductions stratégiques directes :

1. **Révocation proactive des droits :** Si le rapport d'importance des variables identifie que `Acces_Privilegie` est le facteur dominant menant à la prédiction de fraude, la direction de l'audit peut automatiser des revues de rôles informatiques (IAM) ciblées sur des employés précis.
2. **Identification de profils à risque latents:** Certains employés pourraient scorer très haut (ex: 85% de probabilité) malgré une absence d'accès privilégié, ce qui indique un signal d'alerte RH latent. Le contrôle interne peut coordonner des actions d'apaisement (gestion de la surcharge de travail `Heures_Supp_Mois` ou gestion du `Stress`).

### Conclusion sur les limites
Les modèles Machine Learning reposent sur le principe que "le passé permet de prévoir l'avenir". Cependant, les fraudeurs sont par définition adaptatifs et cherchent perpétuellement à échapper aux règles de conformité. Les modèles de détection (Supervisés) doivent donc régulièrement être croisés avec de la détection d'anomalies (Non supervisée) pour capter de nouveaux canaux de fraude comportementaux.
