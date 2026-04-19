# 🏗️ Architecture Reorganization Plan

## Target Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│  app.py — Pure UI layout + orchestration (no business logic)│
└──────────────────────┬─────────────────────────────────────┘
                       │ calls
┌──────────────────────┼─────────────────────────────────────┐
│              VISUALIZATION LAYER                            │
│  visualizations.py — Plotly charts (interactive dashboards) │
│  seaborn_viz.py    — Seaborn charts (academic/exploration)  │
└──────────────────────┬─────────────────────────────────────┘
                       │ reads from
┌──────────────────────┼─────────────────────────────────────┐
│               INTELLIGENCE LAYER                            │
│  results_engine.py — Extract insights from processed data   │
│  groq_agent.py     — LLM intent parsing + synthesis         │
│  ml_models.py      — Supervised learning ONLY               │
└──────────────────────┬─────────────────────────────────────┘
                       │ consumes
┌──────────────────────┼─────────────────────────────────────┐
│                 DATA LAYER                                  │
│  data_pipeline.py  — Load → Clean → Outliers → Features    │
│                      → Normalize → Cluster → Score          │
│  config.py         — All constants & configuration          │
└────────────────────────────────────────────────────────────┘
```

## Changes Per File

### 1. `data_pipeline.py` — Gains outlier detection
- **ADD** `detect_outliers_iqr()` — moved FROM `ml_models.py`
- Outlier detection is a preprocessing/exploration step, not ML
- Pipeline return dict gains `"outlier_info"` key

### 2. `ml_models.py` — Becomes pure supervised ML
- **REMOVE** `detect_outliers_iqr()` — moved to `data_pipeline.py`
- **KEEP** `select_features()`, `create_target()`, `train_models()`, `get_performance_summary()`
- Remove unused `KNeighborsClassifier` import

### 3. `app.py` — Becomes pure UI orchestration
- **REMOVE** duplicate `import matplotlib.pyplot as plt` on line 612
- **FIX** import of `detect_outliers_iqr` → now from `data_pipeline` instead of `ml_models`
- **REMOVE** `from config import SECTOR_WEIGHTS` inline import on line 757 (already available)
- All data processing delegated to layers below

### 4. `config.py` — Add SECTOR_WEIGHTS to top-level exports
- Add `SECTOR_WEIGHTS` to the imports used in `app.py` so the inline import is unnecessary

### 5. `results_engine.py` — No changes needed ✅
### 6. `groq_agent.py` — No changes needed ✅
### 7. `visualizations.py` — No changes needed ✅
### 8. `seaborn_viz.py` — No changes needed ✅
