# ANNEXE PRATIQUE : CODE PYTHON ET RÉSULTATS GRAPHIQUES

Ce rapport inclut le code complet et l'analyse visuelle de la modélisation du risque de fraude basé sur le jeu de données comportementales et organisationnelles.

---

## 1. CRÉATION DU DATASET ET PIPELINE DE PRÉPARATION

Le code ci-dessous (tel qu'établi dans la méthodologie) crée notre base de données sur-mesure d'employés et modélise la variable cible en se basant sur la théorie du **Triangle de Cressey**.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# ── CELLULE 1 : Création du Dataset ──────────────────────────
np.random.seed(42)
n_employes = 1000
 
data = {
    'ID_Employe': range(1, n_employes + 1),
    'Departement': np.random.choice(
        ['Achats', 'Finance', 'Ventes', 'RH', 'IT', 'Logistique'],
        n_employes, p=[0.15, 0.15, 0.25, 0.10, 0.15, 0.20]
    ),
    'Anciennete_Annees':        np.random.randint(1, 25, n_employes),
    'Score_Pression_Financiere': np.random.randint(1, 11, n_employes),
    'Satisfaction_Travail':      np.random.randint(1, 11, n_employes),
    'Heures_Supp_Mois':          np.random.randint(0, 50, n_employes),
    'Conges_Non_Pris':           np.random.randint(0, 30, n_employes),
    'Acces_Privilegie':          np.random.choice([0, 1], n_employes, p=[0.8, 0.2])
}
df = pd.DataFrame(data)
 
# Variable cible — Triangle de Cressey
score_risque = (
    (df['Score_Pression_Financiere'] * 0.4) +
    (df['Acces_Privilegie'] * 15) +
    ((10 - df['Satisfaction_Travail']) * 0.3) +
    (df['Anciennete_Annees'] * 0.2)
)
seuil = np.percentile(score_risque, 95)
df['Fraude_Interne'] = (score_risque >= seuil).astype(int)
df['Score_Risque']   = score_risque
 
# Encodage et nettoyage
le = LabelEncoder()
df['Dept_encoded'] = le.fit_transform(df['Departement'])

print(f"📊 Dataset : {len(df)} employés | {df['Fraude_Interne'].sum()} cas de fraude ({df['Fraude_Interne'].mean()*100:.1f} %)")
```

---

## 2. REPRÉSENTATIONS GRAPHIQUES ET ANALYSE (EDA)

### GRAPHIQUE 1 : Distribution des Scores de Risque
```python
# ── CELLULE 2 : Distribution ─────────────────────────
plt.figure(figsize=(10, 6))
sns.histplot(df['Score_Risque'], bins=40, kde=True, color='blue')
plt.axvline(x=seuil, color='red', linestyle='--', label='Seuil de Fraude (Top 5%)')
plt.title("Distribution du Score de Risque des Employés")
plt.xlabel("Score de Risque Théorique")
plt.ylabel("Nombre d'employés")
plt.legend()
plt.show()
```
*Interprétation : Ce graphique illustre comment le seuil des 5% isole clairement les individus dont la combinaison "Pression, Opportunité (Accès) et Rationalisation" aboutit à la création de la variable cible de la fraude.*

### GRAPHIQUE 2 : Matrice de Corrélation
```python
# ── CELLULE 3 : Corrélation ─────────────────────────
plt.figure(figsize=(10, 8))
features_corr = df.drop(columns=['ID_Employe', 'Departement', 'Score_Risque'])
sns.heatmap(features_corr.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice de Corrélation des Variables")
plt.show()
```
*Interprétation : La heatmap démontrera la forte corrélation imposée artificiellement entre `Acces_Privilegie`, `Score_Pression_Financiere` et la variable cible `Fraude_Interne`.*

---

## 3. ENTRAÎNEMENT DU MODÈLE ET GESTION DU DÉSÉQUILIBRE

```python
# ── CELLULE 4 : Entraînement et SMOTE ────────────────
X = df[['Anciennete_Annees', 'Score_Pression_Financiere', 'Satisfaction_Travail', 
        'Heures_Supp_Mois', 'Conges_Non_Pris', 'Acces_Privilegie', 'Dept_encoded']]
y = df['Fraude_Interne']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Gestion du déséquilibre de classe
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Modèle Random Forest (ou XGBoost)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
```

---

## 4. ÉVALUATION ET PERFORMANCES (MATRICE ET IMPORTANCE)

### GRAPHIQUE 3 : Matrice de Confusion
```python
# ── CELLULE 5 : Matrice de confusion ────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Fraude', 'Fraude'], yticklabels=['Non-Fraude', 'Fraude'])
plt.title("Matrice de Confusion (Random Forest + SMOTE)")
plt.ylabel("Valeur Réelle")
plt.xlabel("Valeur Prédite")
plt.show()
```

### GRAPHIQUE 4 : Importance des Variables (Features Importance)
```python
# ── CELLULE 6 : Feature Importance ─────────────────
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=[features_names[i] for i in indices], palette='viridis')
plt.title("Importance des Variables selon la Random Forest")
plt.xlabel("Importance relative")
plt.show()
```
*Interprétation du Modèle : L'arbre de décision placera logiquement l'**Accès Privilégié** et la **Pression Financière** en tête, validant ainsi expérimentalement le Triangle de la Fraude.*

### GRAPHIQUE 5 : Courbe ROC
```python
# ── CELLULE 7 : Courbe ROC / AUC ───────────────────
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title("Courbe ROC - Détection de Fraude")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.legend(loc="lower right")
plt.show()
```
