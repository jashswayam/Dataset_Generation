import xmltodict
import polars as pl
from Operations import ExtendedOperator  # Assuming this exists for filtering
import ast

class AggregatorHelper:
    @staticmethod
    def get_aggregator(group_col, function, alias_name):
        agg_mapping = {
            "mean": pl.col(group_col).mean().alias(alias_name),
            "sum": pl.col(group_col).sum().alias(alias_name),
            "std": pl.col(group_col).std().alias(alias_name),
            "count": pl.col(group_col).count().alias(alias_name),
            "max": pl.col(group_col).max().alias(alias_name),
            "min": pl.col(group_col).min().alias(alias_name)
        }
        return agg_mapping.get(function.strip(), None)

class DataFrameHelper:
    @staticmethod
    def apply_group_by(dataset_df, join_key, group_col, columns, functions):
        function_list = [func.strip() for func in functions.split(",")]
        agg_exprs = []
        for func, col in zip(function_list, columns):
            alias_name = col["@name"]
            agg_expr = AggregatorHelper.get_aggregator(group_col, func, alias_name)
            if agg_expr is not None:
                agg_exprs.append(agg_expr)
        dataset_df = dataset_df.groupby(join_key).agg(agg_exprs)
        return dataset_df

    @staticmethod
    def apply_filters(dataset_df, filters):
        for flt in filters:
            column, operator, value = flt.get("Column"), flt.get("Operator"), flt.get("Value")
            if column and operator and value:
                operation = getattr(ExtendedOperator, operator, None)
                try:
                    value = ast.literal_eval(value)
                except (ValueError):
                    value = value
                if operation:
                    dataset_df = dataset_df.filter(operation(pl.col(column), value))
                else:
                    raise ValueError(f"Unsupported operator: {operator}")
        return dataset_df

def dynamic_threshold(xml_dict: dict, datasets: dict, lazy: bool = False):
    dynamic_thresholds = xml_dict.get("Rule", {}).get("DynamicThresholdCalculations", {})

    primary_key = dynamic_thresholds.get("PrimaryKey", {}).get("Key", None)
    if not primary_key or not primary_key.strip():
        raise ValueError("PrimaryKey <Key> cannot be null or empty")

    calculations = dynamic_thresholds.get("Calculation", [])
    if not isinstance(calculations, list):
        calculations = [calculations]

    DYN_CAL_DF = pl.DataFrame()
    initial_flag = True

    def get_polars_type(type_str):
        type_mapping = {
            'str': pl.Utf8,
            'float32': pl.Float32,
            'float64': pl.Float64,
            'int32': pl.Int32,
            'int64': pl.Int64,
            'bool': pl.Boolean
        }
        return type_mapping.get(type_str, pl.Utf8)

    for calc in calculations:
        dataset_id = calc.get("DatasetId")
        join_key = calc.get("Keys").get("Key")
        columns = calc.get("Columns", {}).get("Column", [])

        if not isinstance(columns, list):
            columns = [columns]

        column_mapping = {col["@name"]: col["@type"] for col in columns if "@name" in col}

        dataset_df = datasets[dataset_id]

        if lazy:
            dataset_df = dataset_df.lazy()

        filters = calc.get("Filters", {}).get("Filter", [])
        if not isinstance(filters, list):
            filters = [filters]

        dataset_df = DataFrameHelper.apply_filters(dataset_df, filters)

        value_section = calc.get("Value")
        group_by_section = value_section.get("GroupBy") if value_section else None

        if group_by_section:
            group_col = group_by_section.get("Column")
            functions = group_by_section.get("Function")
            dataset_df = DataFrameHelper.apply_group_by(dataset_df, join_key, group_col, columns, functions)
            dataset_df = dataset_df.select(list(dataset_df.columns))

        for column_name, column_type in column_mapping.items():
            polars_type = get_polars_type(column_type)
            dataset_df = dataset_df.with_columns([pl.col(column_name).cast(polars_type).alias(column_name)])

        if initial_flag:
            DYN_CAL_DF = pl.concat([DYN_CAL_DF.lazy(), dataset_df.lazy()], how='horizontal')
            initial_flag = False
        else:
            DYN_CAL_DF = DYN_CAL_DF.join(dataset_df, on=join_key, how="left")

    return DYN_CAL_DF

if __name__ == "__main__":
    with open('C:/Users/h59257/Downloads/rule_2.xml', 'r', encoding='utf-8') as file:
        xml_data = file.read()

    xml_dict = xmltodict.parse(xml_data)
    datasets = {"ds2": pl.read_csv("C:/Users/h59257/Downloads/profiles_sgp.csv")}
    DYN_CAL_DF = dynamic_threshold(xml_dict, datasets, lazy=True).collect()

    print(DYN_CAL_DF)