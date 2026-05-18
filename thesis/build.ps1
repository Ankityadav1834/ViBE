# Build the ViBE Thesis
# Run this script from c:\Vibe_Battery\thesis\
# Requires: a LaTeX distribution (MiKTeX or TeX Live) in PATH

$ErrorActionPreference = "Continue"
Set-Location "$PSScriptRoot"

Write-Host "=== Building ViBE Thesis ===" -ForegroundColor Cyan

# First pass — generates .aux files
pdflatex -shell-escape -interaction=nonstopmode main.tex

# BibTeX — generates bibliography
bibtex main

# Second pass — resolves citations
pdflatex -shell-escape -interaction=nonstopmode main.tex

# Third pass — resolves cross-references
pdflatex -shell-escape -interaction=nonstopmode main.tex

Write-Host ""
if (Test-Path "main.pdf") {
    Write-Host "SUCCESS: main.pdf generated." -ForegroundColor Green
    Write-Host ("Size: {0:N1} KB" -f ((Get-Item main.pdf).Length / 1KB))
} else {
    Write-Host "FAILED: main.pdf not found. Check main.log for errors." -ForegroundColor Red
}
