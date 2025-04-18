@echo off
echo Running FAL experiment with SimCLR-based SSL Entropy Strategy

:: Create SSL_checkpoints directory if it doesn't exist
if not exist "SSL_checkpoints" mkdir SSL_checkpoints

:: Check if SimCLR checkpoint file already exists
if not exist "SSL_checkpoints\final_checkpoint.pt" (
    echo SimCLR model checkpoint not found in SSL_checkpoints directory.
    echo Please place your SimCLR model checkpoint as SSL_checkpoints/final_checkpoint.pt
    exit /b 1
)

:: Run the FAL experiment with SSL-Entropy strategy
python main.py --strategy SSLEntropy

echo Experiment completed.
