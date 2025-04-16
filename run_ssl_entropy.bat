@echo off
echo Running FAL experiment with SSL-based Entropy Strategy

:: Create distribution directory if it doesn't exist
if not exist "distribution" mkdir distribution

:: Check if checkpoint file already exists in distribution directory
if not exist "distribution\round_99.pt" (
    echo SSL model checkpoint not found in distribution directory.
    echo Please place your SSL model checkpoint as distribution/round_99.pt
    exit /b 1
)

:: Run the FAL experiment with SSL-Entropy strategy
python main.py

echo Experiment completed.
