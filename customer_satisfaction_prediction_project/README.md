# Customer Satisfaction Prediction System
## Predicting Customer Sentiment Before Order Fulfillment

Python Version-->https://www.python.org/downloads/
ZenML-->https://zenml.io/
MLflow-->https://mlflow.org/

---

## Table of Contents
- Overview
- Problem-statement
- Dataset-description
- Architecture & Design
- Technical Stack
- Installation & Setup
- Pipeline Components
- Deployment Strategies
- Running the Project
- Web Application
- Troubleshooting
- Future Improvements

---

## Overview

This project demonstrates an end-to-end machine learning solution for predicting customer satisfaction scores in e-commerce environments. By leveraging historical order data, we can anticipate how customers will rate their experience before the order is even completed, enabling proactive service improvements and better resource allocation.

The system is built using **ZenML** for pipeline orchestration, **MLflow** for experiment tracking and model deployment, and **Streamlit** for the user-facing application. What makes this implementation unique is its flexible deployment architecture that accommodates different operating system constraints while maintaining production-ready standards.

### Key Features
- **Automated ML Pipeline**: Seamless data ingestion, preprocessing, training, and evaluation
- **Continuous Deployment**: Automatic model deployment based on performance thresholds
- **Experiment Tracking**: Complete lineage tracking of models, parameters, and metrics
- **Dual Deployment Strategy**: Flexible deployment options for Windows and Linux environments
- **Interactive Dashboard**: Real-time predictions through a user-friendly interface
- **Production Ready**: Scalable architecture suitable for cloud deployment

---

## Problem Statement

In the competitive e-commerce landscape, customer satisfaction is paramount. However, traditional feedback mechanisms are reactive—we only learn about customer dissatisfaction after the experience is complete. This project addresses this challenge by:

**Primary Objective**: Predict the review score (1-5 stars) a customer will give for their next order based on historical patterns, order characteristics, and transactional features.

**Business Impact**:
- **Proactive Intervention**: Identify potentially negative experiences before they occur
- **Resource Optimization**: Allocate customer service resources to high-risk orders
- **Experience Personalization**: Tailor communication and service levels based on predicted satisfaction
- **Trend Analysis**: Understand which factors most strongly influence customer sentiment

---

## Dataset Description

We utilize the **Brazilian E-Commerce Public Dataset by Olist**, available on Kaggle(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). This comprehensive dataset contains real-world e-commerce transactions with rich feature sets.

### Dataset Characteristics
- **Volume**: 100,000+ orders
- **Time Period**: 2016-2018
- **Geographic Coverage**: Multiple marketplaces across Brazil
- **Granularity**: Order-level data with customer, product, and seller information

### Key Features
The dataset encompasses multiple dimensions of the e-commerce experience:

**Order Information**:
- Order status and timestamps
- Purchase date and delivery dates
- Order value and item quantities

**Payment Details**:
- Payment method (credit card, boleto, voucher, debit card)
- Number of payment installments
- Payment values and dates

**Logistics Metrics**:
- Estimated vs. actual delivery time
- Freight costs
- Shipping performance indicators

**Product Attributes**:
- Product category
- Product dimensions and weight
- Number of photos in listing
- Product description length

**Customer Data**:
- Customer location (state, city)
- Customer unique identifiers
- Purchase history patterns

**Review Scores** (Target Variable):
- Score: 1-5 stars
- Review comments
- Review creation date

---

## Architecture & Design

### System Architecture

The project follows a modular, pipeline-based architecture that separates concerns and enables independent scaling of components:

```
┌─────────────────┐
│   Data Source   │
│  (Olist Data)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│      ZenML Training Pipeline        │
│  ┌─────────────────────────────┐    │
│  │  1. Data Ingestion          │    │
│  │  2. Data Cleaning           │    │
│  │  3. Feature Engineering     │    │
│  │  4. Model Training          │    │
│  │  5. Model Evaluation        │    │
│  └─────────────────────────────┘    │
└──────────────┬──────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│       MLflow Tracking Server         │
│  (Metrics, Parameters, Artifacts)    │
└──────────────┬───────────────────────┘
               │
               ▼
┌──────────────────────────────────────┐
│   Deployment Decision Engine         │
│  (Performance Threshold Validation)  │
└──────────────┬───────────────────────┘
               │
               ├─────────────────────────────┐
               ▼                             ▼
    ┌──────────────────┐         ┌──────────────────────┐
    │  Windows Deploy  │         │   Linux Deploy       │
    │  (Direct Load)   └───┐     │   (MLflow Deployer)  │
    └────────┬─────────┘   │     └──────────┬───────────┘
             │             │                │       
             └──────────┬  └────────────────│
                        ▼                   ▼
              ┌──────────────────┐     ┌────────────────────────┐
              │  Streamlit App   │     │  CLI                   │
              │  (User Interface)│     │(Command Line Interface)│
              └──────────────────┘     └────────────────────────┘
                        
```

### Design Principles

1. **Modularity**: Each pipeline step is self-contained and reusable
2. **Reproducibility**: All experiments are tracked and versioned
3. **Flexibility**: Support for multiple deployment environments
4. **Scalability**: Cloud-ready architecture with minimal modifications
5. **Observability**: Comprehensive logging and monitoring capabilities

---

## Technical Stack

### Core Frameworks
- **ZenML (>=0.20.0)**: ML pipeline orchestration and workflow management
- **MLflow**: Experiment tracking, model registry, and deployment
- **Streamlit**: Interactive web application framework

### Machine Learning
- **scikit-learn**: Model training and evaluation
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations

### Supported Models
The project currently implements and evaluates:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- LightGBM (optional)

### Development Tools
- **Python 3.8+**: Core programming language
- **Git**: Version control
- **WSL (Windows Subsystem for Linux)**: For Windows users requiring full deployment features

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Git
- (Optional) WSL for Windows users wanting full deployment capabilities

### Step 1: Clone the Repository

```bash
git clone https://github.com/zenml-io/zenml-projects.git
cd zenml-projects/customer-satisfaction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install ZenML with Server Support

```bash
pip install "zenml[server]"
```

### Step 5: Install MLflow Integration

```bash
zenml integration install mlflow -y
```

### Step 6: Configure ZenML Stack
First and foremost, you need to initialize a zenml repository by using the command
```bash
# initialize a zenml repo
zenml init
```
A ZenML stack defines the infrastructure and tools your pipelines will use. For this project, we need an MLflow-enabled stack:

```bash
# Register MLflow experiment tracker
zenml experiment-tracker register mlflow_tracker --flavor=mlflow

# Register MLflow model deployer
zenml model-deployer register mlflow_deployer --flavor=mlflow

# Register MLflow model registry (Optional)
zenml model-registry register mlflow_registry --flavor=mlflow

# Create and set the stack
zenml stack register mlflow_stack \
    -a default \
    -o default \
    -d mlflow_deployer \
    -e mlflow_tracker \
    -r mlflow_registry \
    --set
```

### Step 7: Launch ZenML Dashboard

```bash
zenml up
```

This will start the ZenML server and open the dashboard in your browser. The dashboard provides:
- Pipeline run visualization
- Artifact lineage tracking
- Stack component management
- Model registry overview

---

## Pipeline Components

### Training Pipeline

The training pipeline (`run_pipeline.py`) consists of the following steps:

#### 1. Data Ingestion (`ingest_data`)
**Purpose**: Load raw data from the Olist dataset into a structured format

**Process**:
- Reads CSV files from the data directory
- Merges related tables (orders, customers, products, reviews)
- Creates a unified DataFrame for downstream processing
- Validates data integrity and completeness

**Output**: Raw merged DataFrame stored as ZenML artifact

#### 2. Data Cleaning (`clean_data`)
**Purpose**: Prepare data for modeling by handling quality issues

**Process**:
- **Missing Value Treatment**: Imputation strategies based on feature type
- **Data Type Conversion**: Ensure proper types for all columns
- **Column Selection**: Remove irrelevant or redundant features
- **Duplicate Removal**: Eliminate duplicate records

**Output**: Cleaned DataFrame ready for model training
#### 3. Model Training (`train_model`)
**Purpose**: Train machine learning models to predict satisfaction scores

**Process**:
- Splits data into training and validation sets (80/20)
- Trains multiple model types with hyperparameter tuning
- Uses MLflow autologging to capture all training artifacts
- Saves the best-performing model to the model registry

**Models Evaluated**:
- **Linear Regression**: Baseline model
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting with regularization

**MLflow Tracking**:
- Model parameters automatically logged
- Training metrics recorded per epoch
- Model artifacts saved to registry

**Output**: Trained model artifact with metadata

#### 5. Model Evaluation (`evaluation`)
**Purpose**: Assess model performance using multiple metrics

**Metrics Calculated**:
- **Mean Squared Error (MSE)**: Average squared prediction error
- **Root Mean Squared Error (RMSE)**: MSE in original scale
- **Mean Absolute Error (MAE)**: Average absolute error
- **R² Score**: Proportion of variance explained

**Evaluation Process**:
- Predictions on holdout validation set
- Comparison across multiple models
- Residual analysis and error distribution
- Feature importance ranking

**MLflow Logging**:
- All metrics logged to experiment tracker
- Confusion matrix (for classification view of satisfaction levels)
- Prediction vs. actual plots
- Model comparison dashboard

**Output**: Evaluation metrics artifact and performance report

---

### Deployment Pipeline

The deployment pipeline (`run_deployment.py`) extends the training pipeline with continuous deployment capabilities:

#### 6. Deployment Trigger (`deployment_trigger`)
**Purpose**: Determine if the newly trained model should be deployed

**Decision Logic**:
```python
if accuracy >= config.min_accuracy:
    deploy_model = True
else:
    deploy_model = False
```

**Configurable Parameters**:
- `threshold_mse`: Maximum acceptable MSE
- `min_improvement`: Required improvement over current production model

**Process**:
- Compares new model metrics against deployment criteria
- Retrieves current production model performance
- Logs deployment decision with reasoning
- Triggers downstream deployment step if criteria met

**Output**: Boolean deployment decision with metadata

#### 7. Model Deployer (`model_deployer`)
**Purpose**: Deploy the model as a prediction service

This step has **two implementation strategies** based on your operating system:

**Strategy A: Windows Direct Loading** (Your Primary Approach)

Due to Windows limitations with daemon processes, you can implement a direct model loading strategy:

```python
import mlflow.sklearn

# Load the model directly from MLflow registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

# Use in Streamlit app
predictions = model.predict(input_data)
```

**Advantages**:
- ✅ Works on Windows without WSL
- ✅ Simple implementation
- ✅ Direct access to model
- ✅ No background process management

**Limitations**:
- ❌ Requires manual model loading in application
- ❌ No automatic model serving endpoint
- ❌ Less suitable for microservices architecture

**Strategy B: MLflow Model Deployer** (WSL Implementation)

For full deployment capabilities, you used WSL (Windows Subsystem for Linux):

```python
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

# Deploy as MLflow service
either using:
mlflow_model_deployer_step(
     model=model,
     deploy_decision=deployment_condition,
     workers=workers,
     timeout=timeout,
 )
or this(more flexible and control):
deploy_model(
     run_id=mlflow_run_id,
     model_name="model"
)
```

**Advantages**:
- ✅ Full-featured model serving
- ✅ REST API endpoint for predictions
- ✅ Background daemon process
- ✅ Production-grade deployment
- ✅ Automatic model versioning and rollback

**Requirements**:
- Requires Linux environment (WSL on Windows)
- Background process support
- MLflow tracking server

**Deployment Architecture**:

```
┌─────────────────────────────────────┐
│         Windows Environment         │
│                                     │
│  ┌───────────────────────────────┐ │
│  │   Streamlit Application       │ │
│  │   (Direct Model Loading)      │ │
│  │                               │ │
│  │   model = mlflow.sklearn      │ │
│  │            .load_model(uri)   │ │
│  └───────────────────────────────┘ │
│                                     │
└─────────────────────────────────────┘

                  VS

┌─────────────────────────────────────┐
│       WSL/Linux Environment         │
│                                     │
│  ┌───────────────────────────────┐  │
│  │  MLflow Deployment Server     │  │
│  │  (Background Daemon)          │  │
│  │                               │  │
│  │  http://localhost:5000        │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

---

## Deployment Strategies

### Understanding the Implementation

The project cleverly handles the Windows daemon process limitation by implementing two complementary deployment files:

#### File 1: `deployment_pipeline_windows.py`
**Purpose**: Windows-compatible deployment without background services

**Key Components**:
```python
def deploy_model_windows():
    """
    Windows-compatible deployment that saves model
    metadata for direct loading in Streamlit
    """
    # Run training pipeline
    training_pipeline.run()
    
    # If model meets criteria, save deployment metadata
    if should_deploy:
        deployment_info = {
            "model_uri": model_uri,
            "model_version": version,
            "deployed_at": timestamp,
            "metrics": evaluation_metrics
        }
        save_deployment_config(deployment_info)
```

**Streamlit Integration**:
```python
# In streamlit_app.py
def load_model_windows():
    # Read deployment config
    config = read_deployment_config()
    
    # Load model directly
    model = mlflow.sklearn.load_model(config["model_uri"])
    return model

# Make predictions
model = load_model_windows()
prediction = model.predict(input_features)
```

#### File 2: `deployment_pipeline_using_server.py`
**Purpose**: Full-featured deployment using MLflow deployer in WSL

**Key Components**:
```python
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

@pipeline
def continuous_deployment_pipeline():
    # Training steps
    data = ingest_data()
    cleaned = clean_data(data)
    X_train, y_train = engineer_features(cleaned)
    model = train_model(X_train, y_train)
    metrics = evaluation(model, X_train, y_train)
    
    # Deployment decision
    should_deploy = deployment_trigger(metrics)
    
    # Deploy as MLflow service
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=should_deploy
    )
```


### Comparison Matrix

| Feature | Windows Direct Load | WSL MLflow Deployer |
|---------|-------------------|---------------------|
| **Setup Complexity** | Low | Medium |
| **Background Process** | Not Required | Required |
| **REST API** | No | Yes |
| **Automatic Updates** | Manual | Automatic |
| **Scalability** | Limited | High |
| **Production Ready** | Development | Yes |
| **Model Versioning** | Manual Tracking | Built-in |
| **Health Checks** | N/A | Built-in |
| **Load Balancing** | N/A | Supported |
| **Windows Support** | ✅ Native | ⚠️ Via WSL |

### When to Use Each Approach

**Use Windows Direct Loading When**:
- Developing and testing on Windows
- Building proof-of-concept applications
- Single-user applications
- Simplicity is prioritized over features
- No need for REST API endpoints

**Use WSL MLflow Deployer When**:
- Deploying to production
- Building multi-user applications
- Need for API-based predictions
- Require automatic model updates
- Want comprehensive monitoring and logging
- Planning to scale to cloud deployment

---

## Running the Project

### Option 1: Training Pipeline Only

Execute the standard training pipeline without deployment:

```bash
python run_pipeline.py
```

**What Happens**:
1. Data is ingested from the Olist dataset
2. Data cleaning and feature engineering performed
3. Models are trained and evaluated
4. Results logged to MLflow
5. Best model saved to registry

**Expected Output**:
```
Initiating pipeline run...
Step: ingest_data - Status: Completed ✓
Step: clean_data - Status: Completed ✓
Step: train_model - Status: Completed ✓
Step: evaluation - Status: Completed ✓

Pipeline run completed successfully!
MSE: 0.45
RMSE: 0.67
R² Score: 0.78
```

### Option 2: Deployment Pipeline (Windows)

Run deployment with direct model loading:

```bash
python run_deployment_windows.py
```

**What Happens**:
1. Complete training pipeline executes
2. Deployment criteria evaluated
3. If approved, model metadata saved
4. Deployment configuration file created
5. Streamlit app can load model directly

**Configuration File Created** (`deployment_config.json`):
```json
{
  "model_uri": "models:/customer_satisfaction_model/3",
  "model_version": "3",
  "deployed_at": "2025-10-05T14:30:00",
  "metrics": {
    "mse": 0.42,
    "rmse": 0.65,
    "r2": 0.81
  },
  "deployment_method": "direct_load"
}
```

### Option 3: Deployment Pipeline (WSL/Linux)

Run full deployment with MLflow model deployer:

```bash
# In WSL terminal
python run_deployment_wsl.py
```

**What Happens**:
1. Complete training pipeline executes
2. Deployment criteria evaluated
3. If approved, MLflow deployer launches service
4. Background daemon starts serving model
5. REST API endpoint available at `http://localhost:5000`

**Service Status Check**:
```bash
# Check if service is running
zenml model-deployer models list

# Output
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ NAME                 ┃ STATUS  ┃ ENDPOINT    ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━┩
│ customer_sat_model_v3│ RUNNING │ :5000       │
└──────────────────────┴─────────┴─────────────┘
```

### Monitoring Pipeline Runs

#### Via ZenML Dashboard
1. Navigate to `http://localhost:8237` (or your ZenML server URL)
2. Click on "Pipelines" in the left sidebar
3. Select your pipeline run
4. View DAG visualization and step details

#### Via MLflow UI
```bash
mlflow ui
```
Navigate to `http://localhost:5000` to view:
- Experiment runs
- Model parameters and metrics
- Artifact storage
- Model registry

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "No Step found for the name mlflow_deployer"

**Cause**: Artifact store corruption or state mismatch after multiple pipeline runs

**Solution**:
```bash
# 1. Find artifact store location
zenml artifact-store describe

# 2. Backup and delete (CAUTION: Destructive operation)
# Make sure you have the correct path!
rm -rf /path/to/artifact/store

# 3. Re-run pipeline
python run_deployment.py
```

#### Issue 2: "No Environment component with name mlflow is currently registered"

**Cause**: MLflow integration not installed in ZenML

**Solution**:
```bash
zenml integration install mlflow -y
zenml integration list  # Verify installation
```

#### Issue 3: MLflow Service Not Starting on Windows

**Cause**: Windows doesn't support daemon processes required by MLflow deployer

**Solution**: Use the Windows-compatible deployment approach:
```bash
# Use the direct loading deployment
python run_deployment_windows.py

# Or use WSL
wsl
python run_deployment_wsl.py
```

#### Issue 4: Model Loading Error in Streamlit

**Cause**: Model URI or version mismatch

**Solution**:
```python
# In streamlit_app.py, add error handling
try:
    model = mlflow.sklearn.load_model(model_uri)
except Exception as e:
    st.error(f"Model loading failed: {str(e)}")
    st.info("Try running the deployment pipeline again")
```

#### Issue 5: Port Already in Use

**Cause**: Previous MLflow or Streamlit instance still running

**Solution**:
```bash
# Find process using port
# On Windows
netstat -ano | findstr :5000

# On Linux/WSL
lsof -i :5000

# Kill the process
# On Windows (use PID from netstat)
taskkill /PID <PID> /F

# On Linux/WSL
kill -9 <PID>
```

#### Issue 6: WSL Cannot Access Windows Files

**Cause**: Path translation issues between Windows and WSL

**Solution**:
```bash
# In WSL, Windows drives are mounted at /mnt/
cd /mnt/c/Users/YourUsername/path/to/project

# Or set up project directly in WSL filesystem
cp -r /mnt/c/path/to/project ~/customer-satisfaction
cd ~/customer-satisfaction
```

#### Issue 7: Data Not Found

**Cause**: Dataset not properly downloaded or path incorrect

**Solution**:
```bash
# Ensure data is in correct location
ls data/olist_*.csv

# If missing, download from Kaggle
# Place CSV files in data/ directory
```

---

## Future Improvements

### Technical Enhancements

1. **Advanced Feature Engineering**
   - Time-series features from customer purchase patterns
   - Text sentiment analysis from product reviews
   - Collaborative filtering based on similar customers

2. **Model Improvements**
   - Deep learning models (Neural Networks, LSTM for sequential data)
   - Ensemble methods combining multiple models
   - AutoML for hyperparameter optimization
   - Online learning for continuous model updates

3. **Deployment Enhancements**
   - Kubernetes deployment for scalability
   - A/B testing framework for model comparison
   - Blue-green deployment strategy
   - Canary releases for gradual rollout

4. **Monitoring & Observability**
   - Model performance monitoring dashboards
   - Data drift detection
   - Prediction latency tracking
   - Alert system for degraded performance

5. **Infrastructure**
   - Cloud deployment (AWS, GCP, Azure)
   - Containerization with Docker
   - CI/CD pipeline integration
   - Infrastructure as Code (Terraform)

### Business Features

1. **Enhanced UI/UX**
   - Batch prediction capability
   - Historical prediction tracking
   - Exportable reports
   - Mobile-responsive design

2. **Additional Insights**
   - Recommendation engine for improving low-scoring orders
   - Customer segmentation visualization
   - Trend analysis over time
   - ROI calculator for interventions

3. **Integration Capabilities**
   - REST API documentation
   - Webhook notifications for predictions
   - Integration with CRM systems
   - Real-time order processing

---

## Conclusion

This project demonstrates a complete machine learning workflow from data ingestion to model deployment, with a unique approach to handling platform-specific constraints. By implementing both direct model loading for Windows and full MLflow deployment for Linux/WSL, the system maintains flexibility without compromising on functionality.

The modular architecture powered by ZenML ensures that the pipeline is reproducible, scalable, and production-ready. Whether you're running on Windows for development or deploying to Linux-based production environments, this framework provides a solid foundation for building customer satisfaction prediction systems.

### Key Takeaways

- **Flexibility**: Support for multiple deployment strategies based on environment
- **Reproducibility**: Complete experiment tracking and model versioning
- **Scalability**: Cloud-ready architecture with minimal modifications required
- **Usability**: User-friendly interface for non-technical stakeholders
- **Maintainability**: Modular design enabling easy updates and improvements

---

## Resources & References

- https://docs.zenml.io/)
- https://mlflow.org/docs/latest/index.html)
- https://streamlit.io/docs