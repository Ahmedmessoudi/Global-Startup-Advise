# 🌍 Startup Country Advisor

**Find your next startup home with Data Mining & AI.**

Startup Country Advisor is a data-driven intelligence tool designed to help entrepreneurs and founders identify the best countries for their next venture. By processing the **CIA World Factbook** through a rigorous data mining pipeline and synthesizing insights with **Groq AI (Qwen-QwQ-32b)**, this application provides actionable, sector-specific recommendations.

---

## 🚀 Key Features

- **🧠 Intent-Aware Chat**: Describe your startup idea in natural language, and the AI will extract your target country and industry sector.
- **📊 7-Feature Data Mining Pipeline**:
    - **Market Size**: Evaluates population and GDP scale.
    - **Digital Penetration**: Measures internet usage and connectivity.
    - **Economic Stability**: Factors in inflation and public debt.
    - **Growth Potential**: Looks at GDP growth and labor market health.
    - **Infrastructure & Trade**: Analyzes electricity, mobile subsciptions, and trade openness.
    - **Human Capital**: Weighted by literacy rates and wealth per capita.
- **🤖 Results Synthesis**: Receive a structured analysis including Opportunity Assessments, Risk/Reward breakdowns, and calculated rankings.
- **🗺️ Interactive Visualizations**:
    - **Global Opportunity Map**: Choropleth maps coloring the world by potential.
    - **Radar Profiles**: Compare a country's percentile rank across all 7 features.
    - **K-Means Clustering**: Discover "Similar Markets" to your target country based on economic patterns.
    - **PCA Visualization**: See how countries cluster in a 2D projection of the 7D feature space.

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM Engine**: Groq API (`qwen-qwq-32b`)
- **Data Analysis**: Pandas, NumPy, Scipy
- **Machine Learning**: Scikit-Learn (K-Means, PCA, MinMaxScaler)
- **Visualizations**: Plotly (Radar, Bar, Choropleth, Scatter)

---

## 📂 Project Structure

```text
├── app.py                    # Main Streamlit application & UI
├── groq_agent.py             # LLM logic (Intent Parsing & Result Synthesis)
├── data_pipeline.py          # Data mining pipeline (Load -> Clean -> Feature Eng -> Cluster)
├── results_engine.py         # Scoring, ranking, and comparison logic
├── visualizations.py         # Plotly chart generation
├── config.py                 # Sector weights and app constants
├── data/
│   └── countries.csv         # CIA World Factbook dataset
└── STARTUP_COUNTRY_ADVISOR_PROJECT.md  # Detailed technical specification
```

---

## 🔧 Setup & Installation

### 1. Prerequisites
- Python 3.9+
- A [Groq Cloud API Key](https://console.groq.com/)

### 2. Clone and Install
```bash
git clone <your-repo-url>
cd data-mining-project
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=qwen/qwen3-32b
```

### 4. Prepare Data
Download the `countries.csv` dataset from [Kaggle](https://www.kaggle.com/datasets/lucafrance/the-world-factbook-by-cia/data) and place it in the `data/` directory.

---

## 🚦 Usage

Launch the application:
```bash
streamlit run app.py
```

1. **Describe your startup**: Enter something like *"I want to launch a fintech startup in Tunisia"* or *"My edtech platform targets students in Nigeria"*.
2. **Review Analysis**: Explore the **Analysis** and **Map** tabs to see how your target compares to global leaders and regional peers.
3. **Deep Dive**: Use the **Data Pipeline** tab to see exactly how your sector's weights are applied to the raw Factbook indicators.

---

## ⚖️ License
This project is intended for educational and research purposes using public domain data from the CIA World Factbook.
