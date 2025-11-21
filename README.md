# 📊 Automated Data Analysis Platform

A comprehensive, enterprise-grade data analysis and machine learning platform built with Streamlit. This application provides end-to-end capabilities for data scientists and analysts, from data ingestion to model deployment, with an intuitive web interface.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## ✨ Key Features

### 📥 Multi-Source Data Ingestion
- **CSV & Excel Import** - Upload local files with automatic format detection
- **SQL Database Integration** - Connect to any database via SQLAlchemy
- **Persistent Caching** - Save datasets across sessions for seamless workflow

### 🔍 Advanced Filtering System
- **SQL Query Filter** - Execute SELECT queries on in-memory data
- **Date Range Filters** - Temporal data filtering with calendar UI
- **Numeric Range Sliders** - Visual range selection for numeric columns
- **Categorical Multiselect** - Filter by multiple category values
- **Boolean Toggles** - True/False/All filtering options
- **Global Filter Application** - Filters apply across all pages automatically

### 🛠️ Comprehensive Data Cleaning
- **Missing Value Handling** - Multiple strategies (mean, median, mode, constant, drop)
- **Duplicate Detection & Removal** - Intelligent duplicate identification
- **Outlier Management** - IQR-based detection and removal
- **Column Operations** - Rename, change types, drop columns
- **Type Conversion** - Convert between numeric, categorical, datetime, boolean types

### 📊 Rich Data Exploration
- **Statistical Summaries** - Mean, median, std dev, skewness, kurtosis
- **Categorical Analysis** - Unique counts, frequency distributions
- **Correlation Analysis** - Interactive correlation matrices
- **Missing Data Visualization** - Heatmaps showing missing patterns
- **Distribution Analysis** - Histograms, box plots, density plots

### 📈 Interactive Visualizations (11+ Chart Types)
- Histogram
- Box Plot
- Scatter Plot (with trendlines)
- Pair Plot / Scatter Matrix
- Pie Chart
- Bar Chart (with aggregation)
- Line Chart
- Area Chart
- Violin Plot
- Density Heatmap
- Spider/Radar Chart

**Customization Options:**
- Color picker for all charts
- Custom axis labels
- Save charts to dashboard
- Export-ready visualizations

### 🤖 Machine Learning Capabilities

#### Supported Algorithms
- **Linear Regression** - For continuous target variables
- **Logistic Regression** - For binary/multiclass classification
- **Random Forest** - Classifier & Regressor variants
- **XGBoost** - Advanced gradient boosting
- **AutoML** - Automated model selection via PyCaret

#### Feature Engineering
- **Scaling** - StandardScaler, MinMaxScaler
- **Encoding** - One-hot encoding, label encoding
- **Log Transformation** - Handle skewed distributions
- **Correlation-based Feature Selection** - Remove highly correlated features

#### Model Capabilities
- Train/test split configuration
- Automatic problem type detection (classification vs regression)
- Comprehensive metrics (accuracy, F1, precision, recall, RMSE, R²)
- Feature importance visualization
- Prediction interface for new data

### 📆 Time Series Forecasting
- Prophet integration for time series analysis
- Configurable forecast periods
- Automatic trend and seasonality detection
- Visual forecast plots with confidence intervals

### 📊 Dashboard & KPIs
- Configurable Key Performance Indicators
- Display saved visualizations
- Model performance tracking
- Export capabilities

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/automated-data-analysis.git
   cd automated-data-analysis
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload Your Data

Navigate to the **Home** page and:
- Upload a CSV or Excel file, OR
- Connect to a SQL database and execute a query
- Toggle "Keep dataset across sessions" to persist data

### 2. Clean Your Data

Go to the **Data** page to:
- Rename columns
- Change data types
- Handle missing values
- Remove duplicates
- Detect and remove outliers
- Drop unnecessary columns
- Download cleaned dataset

### 3. Apply Global Filters (Optional)

Use the **sidebar** to apply filters that affect all pages:
- SQL queries for complex filtering
- Date ranges for temporal data
- Numeric ranges with sliders
- Category selections
- Boolean toggles

### 4. Explore Your Data

Visit the **Exploration** page for:
- Numeric and categorical summaries
- Missing data analysis
- Correlation matrices
- Statistical distributions

### 5. Create Visualizations

On the **Visualization** page:
- Select from 11+ chart types
- Choose columns for axes
- Customize colors and labels
- Save charts to dashboard

### 6. Train Machine Learning Models

Go to the **ML** page to:
- Apply feature engineering (scaling, encoding, transformations)
- Select target variable
- Choose algorithm
- Configure train/test split
- View metrics and feature importance
- Run AutoML for automatic model selection
- Make predictions on new data

### 7. View Dashboard

The **Dashboard** page provides:
- Customizable KPIs
- All saved visualizations
- Latest model performance metrics
- Quick correlation analysis

## 📁 Project Structure

```
automated-data-ju/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── CHANGELOG.md                # Version history
├── FILTERING_GUIDE.md          # Filtering documentation
├── FILTER_ARCHITECTURE.md      # Filter system architecture
│
├── pages/                      # Streamlit pages
│   ├── __init__.py
│   ├── home.py                 # Data upload & overview
│   ├── data.py                 # Data management & cleaning
│   ├── exploration.py          # EDA & statistics
│   ├── visualization.py        # Interactive charts
│   ├── modeling.py             # ML training & prediction
│   └── dashboard.py            # KPIs & overview
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── data_utils.py           # Data loading & preprocessing
│   ├── filter_utils.py         # Filtering operations
│   ├── ml_utils.py             # Machine learning utilities
│   ├── feature_engineering.py  # Feature transformations
│   └── plotting.py             # Visualization utilities
│
├── models/                     # Model storage & cache
│   └── cached_dataset.pkl      # Cached dataset (auto-generated)
│
├── components/                 # Custom Streamlit components
│   └── __init__.py
│
└── .streamlit/                 # Streamlit configuration
    └── config.toml
```

## 🛠️ Technology Stack

### Core Framework
- **Streamlit** - Web application framework
- **streamlit-option-menu** - Navigation menu component

### Data Processing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Visualization
- **Plotly** - Interactive charts
- **Matplotlib** - Static plotting (backend)
- **Seaborn** - Statistical visualization

### Machine Learning
- **scikit-learn** - ML algorithms and preprocessing
- **XGBoost** - Gradient boosting
- **PyCaret** - AutoML framework
- **Prophet** - Time series forecasting

### Database & Storage
- **SQLAlchemy** - SQL database connectivity
- **SQLite** - In-memory SQL queries
- **Pickle** - Dataset caching

## 🔧 Configuration

### Custom Streamlit Theme

Edit `.streamlit/config.toml` to customize the theme:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Cache Location

By default, datasets are cached in the `models/` directory. To change this, modify `CACHE_FILENAME` in `utils/data_utils.py`.

### Database Connections

The application supports any database compatible with SQLAlchemy. Example connection strings:

```python
# PostgreSQL
postgresql+psycopg2://user:password@host:port/database

# MySQL
mysql+pymysql://user:password@host:port/database

# SQLite
sqlite:///path/to/database.db
```

## 📊 Example Workflows

### Workflow 1: Quick Data Analysis
1. Upload CSV file on Home page
2. Check data summary on Exploration page
3. Create visualizations on Visualization page
4. Save charts to Dashboard

### Workflow 2: ML Model Training
1. Upload dataset on Home page
2. Clean data on Data page (handle missing values, remove outliers)
3. Apply feature engineering on ML page (scale, encode)
4. Train model with selected algorithm
5. View metrics and feature importance
6. Make predictions on new data

### Workflow 3: Advanced Filtering & Analysis
1. Upload large dataset
2. Apply SQL filter in sidebar to subset data
3. Add numeric and categorical filters
4. Explore filtered data on Exploration page
5. Train model on filtered subset

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Update documentation for new features
- Test thoroughly before submitting PRs

## 🐛 Known Issues & Limitations

- **Large Datasets** - Performance may degrade with datasets >1GB due to in-memory processing
- **Browser Compatibility** - Best experience on Chrome/Firefox; Safari may have minor issues
- **AutoML Performance** - PyCaret AutoML can be slow on large datasets
- **Prophet Installation** - May require additional dependencies on Windows

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

## 🔒 Security

- Input validation on all user inputs
- SQL injection prevention via parameterized queries
- No sensitive data stored in cache files
- Session-based data isolation

**Note:** This application is intended for internal/trusted use. Do not expose directly to the internet without proper authentication.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - Initial work

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- scikit-learn contributors for ML tools
- Plotly team for interactive visualizations
- PyCaret team for AutoML capabilities

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: your.email@example.com
- Documentation: [Full Documentation](docs/)

## 🔮 Future Enhancements

- [ ] Add more ML algorithms (CatBoost, LightGBM)
- [ ] Support for more file formats (Parquet, JSON, HDF5)
- [ ] Advanced time series features (ARIMA, SARIMA)
- [ ] Model versioning and experiment tracking
- [ ] Automated report generation (PDF/HTML)
- [ ] Real-time data streaming support
- [ ] Collaborative features (multi-user support)
- [ ] Cloud deployment templates (AWS, Azure, GCP)
- [ ] API endpoint generation for models
- [ ] Integration with data warehouses (Snowflake, BigQuery)

---

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ using Python and Streamlit
