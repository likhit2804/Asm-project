# Urban Flood Risk - Real Data Version

This is the cleaned, ready-to-run Python package for real rainfall-driven flood risk analysis.

## Usage
1. Place your Kaggle rainfall dataset (with `date_time` and `precipMM`) inside the `data/` folder.
2. Run:
   ```
   python main.py data/your_rainfall.csv
   ```
3. Outputs (plots, summaries) will appear in `/results`.

## Requirements
- Python 3.10+
- numpy, pandas, scipy, matplotlib, statsmodels
