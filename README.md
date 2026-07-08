## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/lalala6506/MAI201.git
cd MAI201

# 2. Activate your conda environment
conda activate your-env-name

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull data from DVC remote
dvc pull

# 5. Run the full pipeline (prepare, train, evaluate)
dvc repro

# 6. View metrics
dvc metrics show

# 7. Compare experiments in MLflow
mlflow ui
# Open http://localhost:5000
```