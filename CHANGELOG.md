# Changelog

## [Unreleased] - 2025-01-25

### Added - Global Filtering System 🎉

#### Major Features
- **Intelligent Sidebar Filtering**: Complete redesign of sidebar with smart filter detection
- **Auto-detection**: Automatically identifies column types and provides appropriate filters
- **Cross-page Filtering**: Filters apply consistently across all pages (Exploration, Visualization, ML, Dashboard)

#### Filter Types
1. **📝 Global Text Search**
   - Search across all columns simultaneously
   - Case-insensitive matching
   - Real-time filtering

2. **📅 Date Range Filters**
   - Automatically detects datetime columns
   - Date picker with min/max bounds
   - Inclusive date range filtering

3. **🔢 Numeric Range Filters**
   - Select which numeric columns to filter
   - Interactive range sliders
   - Shows min/max values from data

4. **🏷️ Categorical Filters**
   - Multi-select dropdowns
   - Supports up to 50 unique values per column
   - Sorted alphabetically

5. **✓ Boolean Filters**
   - Simple True/False/All toggle
   - Horizontal radio buttons
   - Auto-detected for boolean columns

#### User Experience Improvements
- **Filter Summary**: Shows filtered row count vs total (e.g., "1,234 / 10,000 rows (12.3%)")
- **Reset Button**: One-click reset for all filters
- **Filter Status Indicators**: Each page shows when filters are active
- **Responsive Design**: Collapsible expanders for each filter type
- **Theme Selector**: Moved to bottom of sidebar (light/dark/custom)

#### Technical Implementation
- **New Module**: `utils/filter_utils.py` with 7 filtering functions
- **Session State**: Added `filtered_df` and `global_search` tracking
- **Updated Pages**: All 4 analysis pages now use filtered data
- **Documentation**: Added comprehensive `FILTERING_GUIDE.md`

#### Files Modified
- `app.py`: Complete sidebar redesign with intelligent filtering
- `pages/exploration.py`: Uses filtered_df with status indicator
- `pages/visualization.py`: Uses filtered_df with status indicator
- `pages/modeling.py`: Uses filtered_df with status indicator
- `pages/dashboard.py`: Uses filtered_df with status indicator

#### Files Added
- `utils/filter_utils.py`: Core filtering utilities
- `FILTERING_GUIDE.md`: User guide for filtering system
- `CHANGELOG.md`: This file

### Changed
- **Sidebar Purpose**: From simple theme selector to comprehensive filtering system
- **Data Flow**: Pages now check for `filtered_df` before falling back to `clean_df`
- **Theme Selector**: Relocated to bottom of sidebar with collapsed label

### Removed
- Old static sidebar content ("Data analysis made easy" message)
- Standalone theme selector at top of sidebar

---

## Benefits

### For Users
- **Faster Analysis**: Filter data once, analyze everywhere
- **Better Insights**: Focus on specific data segments
- **Intuitive Interface**: Automatic filter type detection
- **Visual Feedback**: Always know what data you're viewing

### For Developers
- **Modular Design**: Clean separation of filtering logic
- **Reusable Functions**: All filters are composable
- **Type Safety**: Full type hints throughout
- **Easy Extension**: Add new filter types easily

---

## Migration Notes

### For Existing Users
- No breaking changes
- Existing functionality preserved
- New filtering is optional
- Reset button returns to full dataset view

### For Developers
- Import new module: `from utils import filter_utils as fu`
- Use `st.session_state.get('filtered_df', st.session_state.clean_df)` in pages
- Filter state persists across page navigation
- See `FILTERING_GUIDE.md` for detailed documentation

---

## Future Enhancements (Potential)

- [ ] Save/load filter configurations
- [ ] Filter presets for common scenarios
- [ ] Advanced text search with regex
- [ ] Filter by multiple date columns simultaneously
- [ ] Export filtered data directly from sidebar
- [ ] Filter history/undo functionality
- [ ] Custom filter combinations with AND/OR logic
- [ ] Filter performance metrics
