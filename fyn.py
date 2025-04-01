import xmltodict
import polars as pl
import ast
from operations import ExtendedOperator  # Assuming this exists for filtering


class DataFilter:
    """Class to handle filtering operations on DataFrames."""
    
    @staticmethod
    def apply_filters(dataset_df, filters):
        """
        Apply filters to a dataframe.
        
        Args:
            dataset_df: DataFrame to filter
            filters: List of filter conditions from XML
            
        Returns:
            Filtered DataFrame
        """
        if not filters:
            return dataset_df
            
        if not isinstance(filters, list):  # Ensure filters is a list
            filters = [filters]
            
        filtered_df = dataset_df
        for flt in filters:
            column, operator, value = flt.get("Column"), flt.get("Operator"), flt.get("Value")
            if column and operator and value:
                operation = getattr(ExtendedOperator, operator, None)
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    value = value
                    
                if operation:
                    filtered_df = filtered_df.filter(operation(filtered_df.select(column).collect()[column], value))
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
                    
        return filtered_df


class DataGrouper:
    """Class to handle grouping operations on DataFrames."""
    
    @staticmethod
    def apply_group_by(dataset_df, group_col, functions, join_key):
        """
        Apply group by operations to a dataframe.
        
        Args:
            dataset_df: DataFrame to group
            group_col: Column to group by
            functions: Comma-separated string of functions to apply
            join_key: Key used for joining
            
        Returns:
            Grouped DataFrame
        """
        # Convert function string to list
        function_list = [func.strip() for func in functions.split(",")]
        
        # Apply group by - directly use the function names with Polars
        agg_exprs = []
        for func in function_list:
            try:
                # Get the method from polars column expression
                agg_method = getattr(pl.col(group_col), func, None)
                if agg_method is not None:
                    agg_exprs.append(agg_method().alias(f"{group_col}_{func}"))
                else:
                    # If method doesn't exist, let Polars raise an appropriate error
                    raise AttributeError(f"Function '{func}' not found in Polars column expressions")
            except Exception as e:
                # Re-raise with more context
                raise ValueError(f"Error applying function '{func}' to column '{group_col}': {str(e)}")
        
        grouped_df = dataset_df.group_by(join_key).agg(agg_exprs)
        return grouped_df.select(list(grouped_df.columns))


class ColumnRenamer:
    """Class to handle column renaming operations."""
    
    @staticmethod
    def rename_columns(df, column_definitions, join_key=None):
        """
        Rename columns according to the XML configuration.
        
        Args:
            df: DataFrame to rename columns
            column_definitions: List of column definitions from XML
            join_key: The join key column name
            
        Returns:
            DataFrame with renamed columns
        """
        rename_dict = {}
        
        # Create mapping of result column names to desired names based on XML
        column_mapping = {}
        for col_def in column_definitions:
            if isinstance(col_def, dict) and "@name" in col_def:
                # The column name that will be used in the final result
                new_name = col_def["@name"]
                # The original column name or pattern that might exist in the dataframe
                # This could be function-specific like "amount_mean", "amount_std", etc.
                if "#text" in col_def:
                    original_pattern = col_def["#text"]
                    column_mapping[original_pattern] = new_name
        
        # Find columns in the dataframe that match our patterns and create renaming dictionary
        for col in df.columns:
            # Check if this is a column we should rename
            for pattern, new_name in column_mapping.items():
                # Handle function-generated column names like "amount_mean", "amount_std"
                if col.endswith(f"_{pattern}") or col == pattern:
                    rename_dict[col] = new_name
                    break
            
            # Special handling for join key if needed
            if join_key and col == join_key:
                # Keep the join key as is or use a mapping if specified
                if join_key in column_mapping:
                    rename_dict[join_key] = column_mapping[join_key]
        
        # Apply renaming if we have any
        if rename_dict:
            df = df.rename(rename_dict)
            
        return df


def dynamic_threshold(xml_data: str, datasets: dict, lazy: bool = False):
    """
    Calculate dynamic thresholds based on XML configuration.
    
    Args:
        xml_data: XML string containing calculation rules
        datasets: Dictionary of datasets {dataset_id: polars_dataframe}
        lazy: Whether to use lazy execution
        
    Returns:
        DataFrame with calculated values
    """
    # Convert XML to dict
    xml_dict = xmltodict.parse(xml_data)
    
    # Get the appropriate root element based on your XML structure
    root_key = next(iter(xml_dict))
    
    # Navigate to DynamicThresholdCalculations
    dynamic_thresholds = xml_dict.get(root_key, {})
    if "DynamicThresholdCalculations" in dynamic_thresholds:
        dynamic_thresholds = dynamic_thresholds["DynamicThresholdCalculations"]
    
    # Extract Primary Key for DYN_CAL_DF
    primary_key = dynamic_thresholds.get("PrimaryKey", {}).get("Key", None)
    
    if not primary_key or not primary_key.strip():
        raise ValueError("PrimaryKey <Key> cannot be null or empty")
    
    # Process each Calculation
    calculations = dynamic_thresholds.get("Calculation", [])
    
    if not isinstance(calculations, list):  # Ensure it's always a list
        calculations = [calculations]
    
    dyn_cal_df = pl.DataFrame()
    initial_flag = True
    
    for calc in calculations:
        dataset_id = calc.get("DatasetId")
        join_key = calc.get("Keys", {}).get("Key")
        
        # Get the dataset
        if dataset_id not in datasets:
            raise ValueError(f"Dataset ID '{dataset_id}' not found in provided datasets")
        
        dataset_df = datasets[dataset_id]
        
        # Ensure lazy execution if needed
        if lazy and not dataset_df.is_lazy():
            dataset_df = dataset_df.lazy()
            if not dyn_cal_df.is_empty() and not dyn_cal_df.is_lazy():
                dyn_cal_df = dyn_cal_df.lazy()
        
        # Apply filters if present
        filters = calc.get("Filters", {}).get("Filter", [])
        if filters:
            dataset_df = DataFilter.apply_filters(dataset_df, filters)
        
        # Apply Group By if present
        value_section = calc.get("Value")
        if value_section:
            group_by_section = value_section.get("GroupBy")
            if group_by_section:
                group_col = group_by_section.get("Column")
                functions = group_by_section.get("Function")
                dataset_df = DataGrouper.apply_group_by(dataset_df, group_col, functions, join_key)
        
        # Get column definitions for renaming
        columns_section = calc.get("Columns", {})
        column_defs = columns_section.get("Column", [])
        if not isinstance(column_defs, list):
            column_defs = [column_defs]
        
        # Apply column renaming for this calculation's result
        dataset_df = ColumnRenamer.rename_columns(dataset_df, column_defs, join_key)
        
        # Merge with dyn_cal_df
        if initial_flag:
            dyn_cal_df = dataset_df
            initial_flag = False
        else:
            dyn_cal_df = dyn_cal_df.join(dataset_df, on=join_key, how="left")
    
    # Return collected result if we used lazy execution
    if lazy:
        return dyn_cal_df.collect()
    return dyn_cal_df


if __name__ == "__main__":
    with open('rule_2.xml', 'r', encoding='utf-8') as file:
        xml_data = file.read()
        
    # For testing, create a simple sample dataset
    sample_data = {
        "account_id": ["A001", "A002", "A003", "A004", "A005"],
        "amount": [100, 200, 300, 400, 500],
        "lookback": [5, 6, 7, 8, 9]
    }
    
    # Create a test dataset
    ds2 = pl.DataFrame(sample_data)
    datasets = {"ds2": ds2}
    
    # Run the calculation
    dyn_cal_df = dynamic_threshold(xml_data, datasets)
    print(dyn_cal_df)