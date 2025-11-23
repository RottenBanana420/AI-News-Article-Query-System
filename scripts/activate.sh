#!/bin/bash
# Virtual Environment Activation Script for Unix-based Systems
# This script activates the pyenv-virtualenv environment for the AI News Article Query System

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Environment name
ENV_NAME="ai-news-query"

echo -e "${GREEN}AI News Article Query System - Environment Activation${NC}"
echo "========================================================"

# Check if pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo -e "${RED}Error: pyenv is not installed or not in PATH${NC}"
    echo "Please install pyenv first: https://github.com/pyenv/pyenv"
    exit 1
fi

# Check if pyenv-virtualenv is installed
if ! pyenv commands | grep -q virtualenv; then
    echo -e "${RED}Error: pyenv-virtualenv is not installed${NC}"
    echo "Please install pyenv-virtualenv: https://github.com/pyenv/pyenv-virtualenv"
    exit 1
fi

# Check if the virtual environment exists
if ! pyenv virtualenvs | grep -q "$ENV_NAME"; then
    echo -e "${YELLOW}Virtual environment '$ENV_NAME' not found.${NC}"
    echo "Please create it first using:"
    echo "  pyenv virtualenv 3.10.15 $ENV_NAME"
    echo "  pyenv local $ENV_NAME"
    exit 1
fi

# Activate the environment
echo -e "${GREEN}Activating virtual environment: $ENV_NAME${NC}"
pyenv activate "$ENV_NAME"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Environment activated successfully!${NC}"
    echo ""
    echo "Python version: $(python --version)"
    echo "Python path: $(which python)"
    echo ""
    echo "To deactivate, run: pyenv deactivate"
else
    echo -e "${RED}Failed to activate environment${NC}"
    exit 1
fi
