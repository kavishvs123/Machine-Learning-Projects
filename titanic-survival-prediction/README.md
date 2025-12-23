# Titanic Survival Prediction

This project implements MLOps principles for the classic Titanic dataset.

## 🏗 System Architecture
```mermaid
graph LR
    subgraph Data_Pipeline
        A[Raw CSV] -->|src.data.make_feast_data| B(Parquet Offline Store)
        B -->|Feast| C[(Feature Store)]
    end

    subgraph Training
        C -->|Get Historical Features| D[Train Model]
        D -->|Stacking Classifier| E[Model Artifact .pkl]
    end
