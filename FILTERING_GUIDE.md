# Global Filtering System Guide

## Overview

The application now features an **intelligent global filtering system** in the sidebar that automatically detects column types and provides appropriate filter controls. All filters apply across all pages (Exploration, Visualization, ML, Dashboard).

## Features

### 🔍 Automatic Column Type Detection

The system automatically categorizes columns into:
- **📅 Date/DateTime columns** - Date range pickers
- **🔢 Numeric columns** - Range sliders
- **🏷️ Categorical columns** - Multi-select dropdowns
- **✓ Boolean columns** - True/False/All toggles
- **📝 Text search** - Global search across all columns

### Filter Types

#### 1. Text Search (Global)
- **Location**: Top of sidebar
- **Function**: Searches for text across ALL columns (case-insensitive)
- **Use Case**: Quick search for specific values, IDs, names, etc.
- **Example**: Search "John" to find all rows containing "John" in any column

#### 2. Date Filters
- **Auto-detection**: Automatically appears when datetime columns are found
- **Control**: Date range picker with min/max bounds
- **Features**:
  - Automatically detects date columns
  - Shows one expander per date column
  - Allows selecting start and end dates
  - Filters data inclusively (includes both start and end dates)

#### 3. Numeric Filters
- **Selection**: Choose which numeric columns to filter
- **Control**: Range slider for each selected column
- **Features**:
  - Shows min/max values from the data
  - Allows filtering by range
  - Multiple numeric columns can be filtered simultaneously

#### 4. Categorical Filters
- **Selection**: Choose which categorical columns to filter
- **Control**: Multi-select dropdown
- **Features**:
  - Shows all unique values (up to 50 categories)
  - Select multiple values to include
  - Default: All values selected
  - Values are sorted alphabetically

#### 5. Boolean Filters
- **Auto-detection**: Automatically appears for boolean columns
- **Control**: Radio buttons (All/True/False)
- **Features**:
  - Simple toggle between True, False, or All
  - Horizontal layout for easy selection

## How It Works

### Filter Application Flow

1. **Data Loading**: Load data on the Data page
2. **Filter Detection**: Sidebar automatically detects column types
3. **Filter Selection**: Choose filters from sidebar
4. **Real-time Filtering**: Data is filtered immediately
5. **Cross-page Consistency**: Filtered data is used across all pages

### Session State Management

The system uses Streamlit session state to maintain:
- `clean_df`: Original cleaned dataset
- `filtered_df`: Currently filtered dataset
- `global_search`: Current search term
- Individual filter states for each column

### Filter Summary

At the bottom of the filter section, you'll see:
- **Total rows**: Original dataset size
- **Filtered rows**: Current filtered dataset size
- **Percentage**: How much data remains after filtering

Example:
```
🔍 Filtered: 1,234 / 10,000 rows (12.3%)
```

## Usage Examples

### Example 1: Date Range Analysis
1. Load a dataset with date columns
2. Sidebar automatically shows "📅 Date Filters"
3. Expand the date column expander
4. Select date range (e.g., Jan 2023 - Dec 2023)
5. All pages now show only data from that date range

### Example 2: Numeric Range Filtering
1. In "🔢 Numeric Filters", select columns like "Age" or "Price"
2. Adjust the slider to desired range
3. Visualizations and statistics update automatically

### Example 3: Categorical Filtering
1. In "🏷️ Categorical Filters", select a column like "Category" or "Region"
2. Choose specific values to include
3. Analysis focuses only on selected categories

### Example 4: Combined Filtering
Combine multiple filters for precise analysis:
- Date range: Q1 2023
- Region: ["North", "South"]
- Age: 25-45
- Status: Active only

## Reset Filters

Click the **🔄 Reset All Filters** button at the bottom of the sidebar to:
- Clear all active filters
- Reset search term
- Return to viewing the full dataset

## Integration with Pages

### Exploration Page
- Statistics calculated on filtered data
- Correlation matrices reflect filtered relationships
- Outlier detection on filtered subset

### Visualization Page
- All charts use filtered data
- Legends and axes adjust automatically
- Distribution plots show filtered distributions

### ML Page
- Models train on filtered data
- Feature engineering applied to filtered subset
- Useful for training on specific segments

### Dashboard Page
- Metrics show filtered data statistics
- Quick insights on filtered subset
- Model performance on filtered data

## Performance Notes

- **Large datasets**: Filtering is optimized for performance
- **Categorical limits**: Categories with >50 unique values are not shown in filters
- **Real-time updates**: Filters apply immediately without page refresh
- **Memory efficient**: Only filtered data is processed

## Tips and Best Practices

1. **Start broad, then narrow**: Begin with fewer filters and add more as needed
2. **Check filter summary**: Always verify how many rows remain after filtering
3. **Reset when switching analysis**: Use reset button when starting new analysis
4. **Combine filters strategically**: Use multiple filters for precise segmentation
5. **Date filters first**: Apply date filters before other filters for time-based analysis

## Troubleshooting

### Filters not appearing
- Ensure data is loaded on the Data page
- Check that `clean_df` is not empty
- Verify column types are correctly detected

### No data after filtering
- Check filter summary - may have filtered out all rows
- Use Reset button to start fresh
- Adjust filter ranges to be more inclusive

### Categorical filter missing
- Column may have >50 unique values
- Use text search instead for high-cardinality columns

## Technical Details

### Filter Utilities Module
Location: `utils/filter_utils.py`

Key functions:
- `detect_column_types()`: Automatic type detection
- `apply_numeric_filter()`: Range filtering
- `apply_categorical_filter()`: Value filtering
- `apply_datetime_filter()`: Date range filtering
- `apply_boolean_filter()`: Boolean filtering
- `apply_text_search()`: Global text search
- `get_filter_summary()`: Filter statistics

### Architecture
- **Modular design**: Filters are independent and composable
- **Session state**: Maintains filter state across pages
- **Functional approach**: Pure functions for filtering
- **Type-safe**: Proper type hints throughout
