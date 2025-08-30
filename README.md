# PubChem Temporal Trends Analysis

Interactive Streamlit application analyzing temporal evolution of molecular descriptors in PubChem compounds from 1996-2024.
100+ Molecules random sampled for each decades from 116Mio compounds with MW<1000 from pubchem v18. To be repeated on latest version. 

## Features
- **Temporal Trends**: Track molecular property changes over time
- **Statistical Analysis**: Correlation analysis with significance testing
- **PCA Analysis**: Chemical space evolution visualization
- **Period Comparison**: Statistical comparison between time periods

## Requirements
```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scipy scikit-learn
```

## Usage
```bash
streamlit run pubchem_streamlit_app.py
```

## Data Files Required
- `pubchem_creation_dates.csv` - CID temporal periods
- `molecular_analysis.csv` - Molecular descriptors

Explore how chemical diversity has evolved across decades!
