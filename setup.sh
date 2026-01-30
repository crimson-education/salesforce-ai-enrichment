#!/bin/bash

echo "Lead Enrichment System - Setup"
echo "==============================="
echo ""

echo "1. Installing Python dependencies..."
pip install pandas numpy requests fuzzywuzzy python-Levenshtein python-dotenv --break-system-packages

echo ""
echo "2. Creating .env file from template..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "   ✓ Created .env file - please edit with your API keys"
else
    echo "   ℹ .env already exists - skipping"
fi

echo ""
echo "3. Checking data directory..."
if [ -d data ]; then
    echo "   ✓ Data directory exists"
else
    echo "   ✗ Data directory not found"
fi

echo ""
echo "==============================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your API keys"
echo "  2. Start School Matcher API: cd school-matcher && npm run dev"
echo "  3. Place CSV files in data/ directory"
echo "  4. Run enrichment: python -m src.main"
echo ""
