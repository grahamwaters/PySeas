#!/bin/bash

# Directories to create
directories=("src" "tests" "models" "data/raw_data" "data/processed_data" "docs" "notebooks" "scripts" "assets" ".github")

# Create directories
for dir in "${directories[@]}"; do
  mkdir -p $dir
done

# Create .gitignore file
echo -e "*.pyc\n__pycache__/\n*.ipynb_checkpoints/\n*.DS_Store\n*.env\n*.venv\n*.egg-info/\ndist/\nbuild/" > .gitignore

# Create README.md
echo "# Project Title" > README.md
echo "## Project Description" >> README.md

# Create requirements.txt
touch requirements.txt

# Create Dockerfile
touch Dockerfile

# Create docker-compose.yml
touch docker-compose.yml
