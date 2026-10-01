#  Factoring Collection Analytics & Prediction Agent

An end-to-end data analytics and predictive machine learning system designed for factoring collection operations. The project models invoice risk scoring, payment delay prediction, and collection forecasting through domain-specific synthetic data generation, feature engineering, and ensemble modeling.

##  Key Highlights
- **Domain-Specific Synthetic Data Pipeline:** Designed and generated realistic, multi-relational factoring datasets (Invoices, Customers, Transactions, and Historical Collections) simulating real-world cash flow and risk dynamics.
- **Feature Engineering & Behavioral Indicators:** Extracted key collection metrics including recency, payment delay ratios, customer default risk scores, and invoice volume indicators.
- **Predictive Machine Learning:** Trained supervised classification and regression pipelines for payment delay forecasting and collection risk assessment.
- **Inference & Analytical API:** Implemented a lightweight Python service (\pi_servis.py\) to serve model predictions and analytical risk reports.

##  Tech Stack & Tools
- **Language:** Python
- **Data & Modeling:** Pandas, NumPy, Scikit-Learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, Python API Service
- **Model Serialization:** Pickle (\.pkl\)

##  Project Structure
- \gent_v*.ipynb\: Iterative feature engineering, model training, and analytical agent workflows.
- \pi_servis.py\: Prediction and inference service.
- \master_dataset_agent.xlsx\ / \*.csv\: Curated synthetic datasets representing factoring domain workflows.
- \*.pkl\: Trained prediction models, scalers, and encoders.
