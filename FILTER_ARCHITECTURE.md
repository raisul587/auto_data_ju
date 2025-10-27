# Global Filtering System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         APP.PY                              │
│                    (Main Entry Point)                       │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
┌───────────────────────────┐   ┌─────────────────────────┐
│   SIDEBAR (Left Panel)    │   │   MAIN CONTENT (Right)  │
│   🔍 Global Filters       │   │   Horizontal Navigation │
└───────────────────────────┘   └─────────────────────────┘
                │                           │
                │                           │
                ▼                           ▼
┌───────────────────────────┐   ┌─────────────────────────┐
│  Filter Detection         │   │   Page Routing          │
│  - detect_column_types()  │   │   - Home                │
│  - Auto-categorize cols   │   │   - Data                │
└───────────────────────────┘   │   - Exploration         │
                │                │   - Visualization       │
                ▼                │   - ML                  │
┌───────────────────────────┐   │   - Dashboard           │
│  Filter UI Components     │   └─────────────────────────┘
│  📝 Text Search           │               │
│  📅 Date Pickers          │               │
│  🔢 Range Sliders         │               ▼
│  🏷️ Multi-select          │   ┌─────────────────────────┐
│  ✓ Boolean Toggles        │   │   Pages Use Filtered    │
└───────────────────────────┘   │   Data (filtered_df)    │
                │                └─────────────────────────┘
                ▼
┌───────────────────────────┐
│  Apply Filters            │
│  - apply_text_search()    │
│  - apply_datetime_filter()│
│  - apply_numeric_filter() │
│  - apply_categorical_...  │
│  - apply_boolean_filter() │
└───────────────────────────┘
                │
                ▼
┌───────────────────────────┐
│  Session State Update     │
│  filtered_df ← result     │
└───────────────────────────┘
```

## Data Flow

```
┌──────────────┐
│  User Loads  │
│  Data (CSV)  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  clean_df        │  ← Original cleaned data
│  (Session State) │
└──────┬───────────┘
       │
       │  ┌─────────────────┐
       │  │  User Applies   │
       └─▶│  Filters in     │
          │  Sidebar        │
          └────────┬────────┘
                   │
                   ▼
          ┌────────────────────┐
          │  filter_utils.py   │
          │  - Detect types    │
          │  - Apply filters   │
          └────────┬───────────┘
                   │
                   ▼
          ┌────────────────────┐
          │  filtered_df       │  ← Filtered subset
          │  (Session State)   │
          └────────┬───────────┘
                   │
       ┌───────────┴───────────────┬──────────────┐
       │                           │              │
       ▼                           ▼              ▼
┌──────────────┐    ┌──────────────────┐   ┌─────────────┐
│ Exploration  │    │  Visualization   │   │  Dashboard  │
│ Page         │    │  Page            │   │  Page       │
└──────────────┘    └──────────────────┘   └─────────────┘
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  All pages use       │
                  │  filtered_df for     │
                  │  analysis & viz      │
                  └──────────────────────┘
```

## Filter Application Sequence

```
1. User Opens Sidebar
        │
        ▼
2. System Detects Column Types
   ┌─────────────────────────────────┐
   │ detect_column_types(clean_df)   │
   └─────────────────────────────────┘
        │
        ▼
3. Display Appropriate Filters
   ┌─────────────────────────────────┐
   │ - Datetime → Date pickers       │
   │ - Numeric → Sliders             │
   │ - Categorical → Multi-select    │
   │ - Boolean → Radio buttons       │
   └─────────────────────────────────┘
        │
        ▼
4. User Adjusts Filters
        │
        ▼
5. Apply Filters Sequentially
   ┌─────────────────────────────────┐
   │ df = clean_df.copy()            │
   │ if text_search:                 │
   │   df = apply_text_search(df)    │
   │ if date_filter:                 │
   │   df = apply_datetime_filter()  │
   │ if numeric_filter:              │
   │   df = apply_numeric_filter()   │
   │ if categorical_filter:          │
   │   df = apply_categorical_...()  │
   │ if boolean_filter:              │
   │   df = apply_boolean_filter()   │
   └─────────────────────────────────┘
        │
        ▼
6. Update Session State
   ┌─────────────────────────────────┐
   │ st.session_state.filtered_df    │
   │ = df                            │
   └─────────────────────────────────┘
        │
        ▼
7. Pages Auto-Update
   ┌─────────────────────────────────┐
   │ All pages read filtered_df      │
   │ and display filtered results    │
   └─────────────────────────────────┘
```

## Module Structure

```
data_analysis_app_relfix/
│
├── app.py                          # Main app with sidebar filtering
│   ├── Imports filter_utils
│   ├── Sidebar: Filter UI
│   └── Routes to pages
│
├── utils/
│   ├── filter_utils.py            # NEW: Filtering logic
│   │   ├── detect_column_types()
│   │   ├── apply_text_search()
│   │   ├── apply_datetime_filter()
│   │   ├── apply_numeric_filter()
│   │   ├── apply_categorical_filter()
│   │   ├── apply_boolean_filter()
│   │   └── get_filter_summary()
│   │
│   ├── data_utils.py              # Data loading & cleaning
│   ├── plotting.py                # Visualization functions
│   ├── feature_engineering.py    # Feature transforms
│   └── ml_utils.py                # ML training
│
└── pages/
    ├── home.py                    # Landing page
    ├── data.py                    # Data management
    ├── exploration.py             # UPDATED: Uses filtered_df
    ├── visualization.py           # UPDATED: Uses filtered_df
    ├── modeling.py                # UPDATED: Uses filtered_df
    └── dashboard.py               # UPDATED: Uses filtered_df
```

## Session State Variables

```python
# Core Data
st.session_state.df              # Original uploaded data
st.session_state.clean_df        # Cleaned data (after preprocessing)
st.session_state.filtered_df     # NEW: Filtered subset of clean_df

# Filter State
st.session_state.global_search   # NEW: Text search term
st.session_state.theme           # UI theme (light/dark/custom)

# Filter-specific states (auto-generated)
st.session_state.date_filter_{col_name}      # Date range per column
st.session_state.numeric_filter_{col_name}   # Numeric range per column
st.session_state.cat_filter_{col_name}       # Selected categories per column
st.session_state.bool_filter_{col_name}      # Boolean value per column

# ML State
st.session_state.model_df        # Feature-engineered data
st.session_state.last_model      # Most recent trained model
st.session_state.last_metrics    # Model performance metrics
```

## Filter Type Detection Logic

```python
def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorizes columns by type:
    
    Priority Order:
    1. Datetime (pd.api.types.is_datetime64_any_dtype)
    2. Boolean (pd.api.types.is_bool_dtype)
    3. Numeric (pd.api.types.is_numeric_dtype)
    4. Categorical (everything else)
    
    Returns:
    {
        'datetime': [col1, col2, ...],
        'boolean': [col3, ...],
        'numeric': [col4, col5, ...],
        'categorical': [col6, col7, ...]
    }
    """
```

## Filter Application Order

Filters are applied in this sequence to maximize efficiency:

1. **Text Search** (if provided)
   - Reduces dataset size early
   - Searches across all columns

2. **Date Filters** (if any)
   - Time-based filtering is common
   - Often reduces data significantly

3. **Numeric Filters** (if any)
   - Range-based filtering
   - Applied to selected columns only

4. **Categorical Filters** (if any)
   - Value-based filtering
   - Applied to selected columns only

5. **Boolean Filters** (if any)
   - Simple True/False filtering
   - Applied last as typically few boolean columns

## Performance Considerations

### Optimization Strategies
- **Lazy Evaluation**: Filters only applied when user interacts
- **Categorical Limit**: Max 50 unique values to prevent UI slowdown
- **Copy-on-Filter**: Original data never modified
- **Sequential Filtering**: Each filter operates on previous result

### Memory Management
```
Original Data (clean_df)     → Stored once
Filtered Data (filtered_df)  → Updated on filter change
Filter States                → Minimal memory (just selections)
```

### Scalability
- **Small datasets (<10K rows)**: Instant filtering
- **Medium datasets (10K-100K)**: Sub-second filtering
- **Large datasets (>100K)**: May take 1-2 seconds
- **Very large (>1M)**: Consider data sampling or pagination

## Error Handling

```python
# Each filter function includes error handling:

try:
    filtered_df = apply_filter(df, params)
except Exception as e:
    st.error(f"Filter error: {e}")
    filtered_df = df  # Fallback to unfiltered data
```

## Extension Points

### Adding New Filter Types

1. **Add detection logic** in `detect_column_types()`
2. **Create filter function** in `filter_utils.py`
3. **Add UI component** in `app.py` sidebar
4. **Apply filter** in filter sequence

Example for adding "Time" filter:
```python
# In filter_utils.py
def apply_time_filter(df, column, start_time, end_time):
    # Implementation
    pass

# In app.py sidebar
if col_types['time']:
    st.subheader("⏰ Time Filters")
    # Add time picker UI
```

## Testing Checklist

- [ ] Load data with various column types
- [ ] Apply each filter type individually
- [ ] Combine multiple filters
- [ ] Reset filters
- [ ] Navigate between pages with filters active
- [ ] Check filter summary accuracy
- [ ] Test with edge cases (empty data, single row, etc.)
- [ ] Verify performance with large datasets
