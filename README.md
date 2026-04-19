# 🌍 Global Startup Advisor : Plateforme d'Intelligence de Marché

## 1. Aperçu du Projet

**Global Startup Advisor** est une plateforme analytique d'aide à la décision destinée aux entrepreneurs et aux fondateurs de startups. Ce projet s'inscrit dans le cadre d'une démarche académique de **Data Mining** et d'**Apprentissage Automatique (Machine Learning)**. Il a pour objectif d'évaluer la viabilité et le potentiel de succès d'un projet entrepreneurial dans un pays donné, en se basant sur les indicateurs macroéconomiques du **CIA World Factbook**.

En combinant des pipelines rigoureux de traitement de données, des modèles d'apprentissage non supervisé et supervisé, et l'Intelligence Artificielle générative (Groq API), cette application fournit des recommandations stratégiques, spécifiques à différents secteurs d'activité (Fintech, Edtech, E-commerce, etc.).

---

## 2. Fonctionnalités Principales

- **Chat IA Contextuel** : Interface de traitement du langage naturel (NLP) permettant à l'utilisateur de décrire son projet. L'IA extrait automatiquement le pays cible et le secteur d'activité pour lancer l'analyse.
- **Pipeline de Data Mining Avancé** : Traitement exhaustif des données incluant la gestion des valeurs manquantes, la détection des valeurs aberrantes (Outliers via IQR), et l'ingénierie des caractéristiques pour générer des scores composites (Taille du marché, Pénétration numérique, Stabilité économique, etc.).
- **Apprentissage Non Supervisé (Clustering)** : Classification des pays ayant des profils économiques similaires à l'aide de l'algorithme **K-Means**, visualisable via une réduction de dimensionnalité **PCA (Analyse en Composantes Principales)**.
- **Apprentissage Supervisé (Classification)** : Évaluation prédictive avancée grâce à un *Model Manager* intégrant **Random Forest**, **SVM (SVC)** et **XGBoost**. Le pipeline inclut une sélection de variables (ANOVA F-test) et une optimisation des hyperparamètres par **GridSearchCV** (Validation Croisée K-Fold).
- **Tableau de Bord Analytique** : Suite de visualisations interactives développées avec **Plotly** (Cartes choroplèthes, Graphiques Radar, Jauges de scores) et **Seaborn** (Matrices de corrélation, Pairplots, Graphiques en violon).

---

## 3. Architecture du Système

L'architecture modulaire du projet assure la séparation des préoccupations et la maintenabilité du code :

```text
├── .streamlit/               # Configuration de l'interface Streamlit
├── .env                      # Variables d'environnement (clés API)
├── app.py                    # Point d'entrée de l'application Streamlit et interface utilisateur
├── data_pipeline.py          # Logique de chargement, nettoyage et ingénierie des données
├── ml_models.py              # Architecture d'apprentissage supervisé (ModelManager, GridSearchCV)
├── results_engine.py         # Moteur de scoring pondéré par secteur et logique de comparaison
├── groq_agent.py             # Agent IA pour le parsing des intentions et la synthèse des rapports
├── visualizations.py         # Fonctions de visualisation interactive (Plotly)
├── seaborn_viz.py            # Fonctions d'analyse exploratoire des données (Seaborn)
├── config.py                 # Configuration globale, pondérations sectorielles et constantes
├── requirements.txt          # Dépendances du projet
├── architecture_plan.md      # Plan détaillé de l'architecture du projet
├── STARTUP_COUNTRY_ADVISOR_PROJECT.md # Cahier des charges et spécifications
├── Description.txt           # Description textuelle du concept et du projet
└── data/
    └── factbook.csv          # Base de données source (CIA World Factbook)
```

**Technologies Utilisées :**
- **Frontend / Dashboard** : Streamlit
- **Traitement de Données & ML** : Pandas, NumPy, Scikit-Learn, XGBoost, SciPy
- **Visualisation** : Plotly Express, Plotly Graph Objects, Seaborn, Matplotlib
- **Modélisation LLM** : Groq API 

---

## 4. Installation

### Prérequis
- **Python 3.9** ou supérieur.
- Une clé d'API valide de [Groq Cloud](https://console.groq.com/).

### Étapes de déploiement

1. **Cloner le dépôt :**
```bash
git clone <URL_DU_DEPOT>
cd Global-Startup-Advise
```

2. **Installer les dépendances :**
Il est recommandé d'utiliser un environnement virtuel (venv ou conda).
```bash
pip install -r requirements.txt
```

3. **Configuration de l'environnement :**
Créez un fichier `.env` à la racine du projet pour stocker vos clés d'API de manière sécurisée :
```env
GROQ_API_KEY=votre_cle_api_ici
```

4. **Préparation des données :**
Assurez-vous que le fichier `factbook.csv` (données macroéconomiques) est bien présent dans le répertoire `data/`.

---

## 5. Utilisation

Pour lancer l'interface utilisateur, exécutez la commande suivante à la racine du projet :

```bash
streamlit run app.py
```

### Parcours Utilisateur Typique :
1. **Accueil** : Visualisez les données globales, filtrez par pays et secteur d'activité, et analysez les indicateurs économiques bruts, la carte mondiale et les profils radars.
2. **AI StartUp Advisor** : Saisissez une requête en langage naturel (ex: *"Je souhaite créer une plateforme e-commerce en Égypte"*). L'IA synthétisera le rapport d'opportunités, de risques et de recommandations d'alternatives basées sur les données.
3. **Exploration** : Plongez dans l'analyse exploratoire (EDA) avec des matrices de corrélation, l'analyse des distributions et la composition des clusters.
4. **Modèles Machine Learning** : Exécutez le pipeline d'apprentissage supervisé en temps réel. Comparez les performances de RandomForest, SVM et XGBoost à travers les matrices de confusion, les courbes ROC et les scores de validation croisée.

---

## 6. Résultats et Évaluation

Le système démontre la pertinence d'une approche orientée données (Data-Driven) pour l'évaluation macroéconomique. 
- La méthode de **scoring dynamique pondérée** offre une vue sectorielle réaliste (les besoins d'une Fintech diffèrent de ceux de la Logistique).
- L'approche hybride **Clustering + Classification supervisée** permet d'identifier formellement les caractéristiques les plus discriminantes pour la réussite d'une startup dans une région donnée.
- L'intégration **LLM** prouve son efficacité pour démocratiser l'accès aux insights complexes de data mining via une synthèse textuelle claire, actionnable et formatée professionnellement.
