import xmltodict
import polars as pl
from Operations import ExtendedOperator  # Assuming this exists for filtering
import ast

def Dynamic_Threshold(xml_data: str, datasets: dict, lazy: bool = False):
    # Convert XML to dict
    xml_dict = xmltodict.parse(xml_data)
    dynamic_thresholds = xml_dict.get("Rule", {}).get("DynamicThresholdCalculations", {})
    
    # Extract Primary Key for DYN_CAL_DF
    primary_key = dynamic_thresholds.get("PrimaryKey", {}).get("Key")
    if not primary_key or not primary_key.strip():
        raise ValueError("PrimaryKey <Key> cannot be null or empty")
    
    # Process each Calculation
    calculations = dynamic_thresholds.get("Calculation", [])
    if not isinstance(calculations, list):  # Ensure it's always a list
        calculations = [calculations]
    
    DYN_CAL_DF = pl.DataFrame()
    initial_flag = True
    
    for calc in calculations:
        dataset_id = calc.get("DatasetId")
        join_key = calc.get("Keys", {}).get("Key")
        columns = calc.get("Columns", {}).get("Column", [])
        
        # Ensure columns is always a list
        if not isinstance(columns, list):
            columns = [columns]
        
        # Convert column details to a dictionary {name: type}
        column_mapping = {col["@name"]: col["@type"] for col in columns if "@name" in col}
        
        dataset_df = datasets[dataset_id]
        
        # Ensure lazy execution if needed
        if lazy:
            dataset_df = dataset_df.lazy()
            DYN_CAL_DF = DYN_CAL_DF.lazy()
        
        # Apply filters if present
        filters = calc.get("Filters", {}).get("Filter", [])
        if not isinstance(filters, list):
            filters = [filters]
        
        for flt in filters:
            column, operator, value = flt.get("Column"), flt.get("Operator"), flt.get("Value")
            if column and operator and value:
                operation = getattr(ExtendedOperator, operator, None)
                try:
                    value = ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    pass
                if operation:
                    dataset_df = dataset_df.filter(operation(dataset_df.select(column).collect()[column], value))
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
        
        # Apply Group By if present
        value_section = calc.get("Value", {})
        group_by_section = value_section.get("GroupBy")
        if group_by_section:
            group_col = group_by_section.get("Column")
            functions = group_by_section.get("Function", "").split(",")
            
            agg_mapping = {
                "mean": pl.col(group_col).mean(),
                "sum": pl.col(group_col).sum(),
                "std": pl.col(group_col).std(),
                "count": pl.col(group_col).count(),
                "max": pl.col(group_col).max(),
                "min": pl.col(group_col).min()
            }
            
            agg_exprs = [agg_mapping[func.strip()].alias(f"{group_col}_{func}") for func in functions if func.strip() in agg_mapping]
            dataset_df = dataset_df.group_by(join_key).agg(agg_exprs)
        
        dataset_df = dataset_df.select(list(dataset_df.columns))
        
        # Merge with DYN_CAL_DF
        if initial_flag:
            DYN_CAL_DF = dataset_df
            initial_flag = False
        else:
            DYN_CAL_DF = DYN_CAL_DF.join(dataset_df, on=join_key, how="left")
        
        # Rename columns and set data types according to column_mapping
        dtype_map = {
            "string": pl.Utf8,
            "int": pl.Int64, "integer": pl.Int64, "int64": pl.Int64,
            "int32": pl.Int32, "int16": pl.Int16, "int8": pl.Int8,
            "uint32": pl.UInt32, "uint64": pl.UInt64,
            "float": pl.Float64, "double": pl.Float64, "float64": pl.Float64,
            "float32": pl.Float32, "bool": pl.Boolean, "boolean": pl.Boolean,
            "date": pl.Date, "datetime": pl.Datetime
        }
        
        cast_exprs = [pl.col(col_name).cast(dtype_map[col_type.lower()]).alias(col_name) 
                      for col_name, col_type in column_mapping.items() 
                      if col_type.lower() in dtype_map and col_name in DYN_CAL_DF.columns]
        
        if cast_exprs:
            DYN_CAL_DF = DYN_CAL_DF.with_columns(cast_exprs)
    
    return DYN_CAL_DF

if __name__ == "__main__":
    with open('C:/Users/h59257/Downloads/rule_2.xml', 'r', encoding='utf-8') as file:
        xml_data = file.read()
    
    datasets = {"ds2": pl.read_csv("C:/Users/h59257/Downloads/profiles_sgp.csv")}
    
    DYN_CAL_DF = Dynamic_Threshold(xml_data, datasets, lazy=True).collect()
    print(DYN_CAL_DF)
