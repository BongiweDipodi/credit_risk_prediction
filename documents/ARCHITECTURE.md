# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              User Interface Layer                        │
│                 (Streamlit Web App)                      │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Single     │  │   Batch      │  │  Dashboard &│   │
│  │  Prediction  │  │  Prediction  │  │  Analytics   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
├─────────────────────────────────────────────────────────┤
│           Application Logic Layer                        │
├──────────────────┬──────────────────┬──────────────────┤
│   Inference      │   Validation &   │  Visualization &│
│   Pipeline       │   Error Handling │  Reporting      │
├─────────────────────────────────────────────────────────┤
│            Model Service Layer                          │
├──────────────────┬──────────────────┬──────────────────┤
│  Feature Eng.    │  Model Scoring   │  Explainability │
│  & Transform     │  & Prediction    │  (SHAP)         │
├─────────────────────────────────────────────────────────┤
│            Data Processing Layer                        │
├──────────────────┬──────────────────┬──────────────────┤
│  Data Loading    │  Cleaning &      │  Encoding &     │
│  & Validation    │  Preprocessing   │  Scaling        │
├─────────────────────────────────────────────────────────┤
│              Data & Model Storage                       │
├──────────────────┬──────────────────┬──────────────────┤
│  Training Data   │  Trained Model   │  Artifacts      │
│  (CSV/DB)        │  (joblib)        │  (Scalers, etc) │
└─────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. Data Processing Module (`src/data_processing.py`)
**Responsibilities:**
- Load data from CSV files
- Handle missing values
- Remove duplicates
- Detect and handle outliers
- Data validation and quality checks
- Data splitting (train/test)

**Inputs:** Raw CSV files  
**Outputs:** Clean DataFrame ready for feature engineering

### 2. Feature Engineering Module (`src/feature_engineering.py`)
**Responsibilities:**
- Categorical variable encoding
- Numerical feature scaling
- Feature selection
- Feature transformation
- Feature interaction creation
- Handle skewed distributions

**Inputs:** Clean DataFrame  
**Outputs:** Processed DataFrame with engineered features

### 3. Model Training Module (`src/train_model.py`)
**Responsibilities:**
- Train machine learning models
- Implement cross-validation
- Hyperparameter tuning
- Model evaluation and metrics
- Model persistence (saving)
- Training logging and tracking

**Inputs:** Processed training data  
**Outputs:** Trained model + artifacts (scaler, encoder)

### 4. Model Inference Module (`src/model_inference.py`)
**Responsibilities:**
- Load trained models and artifacts
- Prepare input data
- Generate predictions
- Calculate confidence scores
- Generate explanations (SHAP values)

**Inputs:** New applicant data  
**Outputs:** Risk predictions + confidence + explanations

### 5. Streamlit Application (`app/app.py`)
**Responsibilities:**
- User interface design
- Form handling and validation
- Display predictions
- Visualizations and dashboards
- File upload for batch predictions
- SHAP explanations display

**Features:**
- Single prediction interface
- Batch prediction processing
- Performance dashboard
- Feature importance charts
- Model explainability visualizations

## Data Flow

### Training Pipeline
```
Raw Data (CSV)
    ↓
[Data Processing]
    ↓
Clean Data
    ↓
[Feature Engineering]
    ↓
Engineered Features
    ↓
[Model Training]
    ├→ Train/Test Split
    ├→ Cross-Validation
    ├→ Hyperparameter Tuning
    └→ Model Evaluation
    ↓
Trained Model + Artifacts
    ↓
[Model Persistence]
    ↓
Saved Model Files
```

### Inference Pipeline
```
New Applicant Data
    ↓
[Input Validation]
    ↓
[Feature Engineering]
    (Using saved artifacts)
    ↓
Engineered Features
    ↓
[Model Prediction]
    ↓
Risk Score + Probability
    ↓
[SHAP Explanation]
    ↓
[Format & Display]
    ↓
User Interface
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Web UI and visualization |
| **Backend** | Python | Core logic and processing |
| **ML Framework** | Scikit-learn | Model training and inference |
| **Data Processing** | Pandas, NumPy | Data manipulation |
| **Visualization** | Matplotlib, Seaborn | Charts and plots |
| **Explainability** | SHAP | Model interpretation |
| **Testing** | Pytest | Unit and integration tests |
| **Containerization** | Docker | Deployment and scaling |
| **API** | FastAPI/Flask | Model serving (optional) |

## Configuration Management

Configuration is handled through:
- `config/config.yaml` - Application settings
- `.env` file - Sensitive information (database, API keys)
- `constants.py` - Hard-coded constants

## Error Handling Strategy

1. **Input Validation** - Validate user inputs at UI level
2. **Data Validation** - Check data quality before processing
3. **Error Logging** - Log all errors for debugging
4. **User Feedback** - Display user-friendly error messages
5. **Graceful Degradation** - Fallback mechanisms

## Scalability Considerations

- Model artifacts stored on disk for quick loading
- Batch prediction for processing multiple records
- Caching of preprocessing artifacts
- Potential API layer for external integrations
- Docker containerization for horizontal scaling

## Monitoring & Logging

- **Structured Logging** - Using Python logging module
- **Performance Metrics** - Track prediction latency
- **Model Metrics** - Monitor accuracy over time
- **Audit Trail** - Log all predictions for compliance
- **Health Checks** - Monitor system availability
