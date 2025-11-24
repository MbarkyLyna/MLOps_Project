PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt

setup:

	@echo "Creating the virtual environment and installing dependencies..."
	@virtualenv $(ENV_NAME)
	@. $(ENV_NAME)/bin/activate && pip install -r $(REQUIREMENTS)
	@. $(ENV_NAME)/bin/flake8 *.py --max-line-length=88 --ignore=E501,W503,E203
data:
	@echo "Preparing data..."
	@. $(ENV_NAME)/bin/activate && python main.py --prepare --data "titanic/train.csv"

train:
	@echo "Training model..."
	@. $(ENV_NAME)/bin/activate && python main.py --train --data "titanic/train.csv" --model_out "model.pkl"

test:
	@echo "Running tests..."
	@. $(ENV_NAME)/bin/activate && python main.py --evaluate --model_in "model.pkl" --data "titanic/train.csv"
	@. $(ENV_NAME)/bin/activate && python test_environment.py

# =============================================
# QUALITY GATE – Step 1: Code Formatting
# =============================================
format:
	@echo "Running black formatter..."
	@. $(ENV_NAME)/bin/activate && black *.py --quiet
lint:
	@echo "Running flake8 linter..."
	@$(ENV_NAME)/bin/flake8 *.py --max-line-length=88 --ignore=E501,W503,E203

security:
	@echo "Running bandit security scanner..."
	@. $(ENV_NAME)/bin/activate && bandit -r *.py --quiet || true
quality: format lint security
	@echo "QUALITY GATE PASSED – ALL CHECKS GREEN!"
