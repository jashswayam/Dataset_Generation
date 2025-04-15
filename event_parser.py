import re
from typing import List, Dict

def utils_eventlevel_parser(pxml_event_level: dict, datasets_dict: dict) -> str:
    def extract_dataset(event_columns: List[str], datasets_dict: dict) -> Dict[str, List[str]]:
        dataset_map = {}
        for item in event_columns:
            ds, col = item.split(":", maxsplit=1)
            keys = datasets_dict.get(ds, {}).get("joining_keys", [])
            dataset_map.setdefault(ds, keys)
        return dataset_map

    def build_expression(dataset_map: dict) -> str:
        datasets = list(dataset_map.keys())
        expr = f"{datasets[0]}"
        for ds in range(1, len(datasets)):
            left_key = dataset_map.get(datasets[ds])
            right_key = dataset_map.get(datasets[ds - 1])
            expr += (
                f".join({datasets[ds]}, left_on={left_key}, right_on={right_key}, "
                f"coalesce=True, suffix='_' + {datasets[ds]})"
            )
        return expr

    # Step 1: Extract columns from XML structure
    columns = pxml_event_level.get("RetainedColumns", {}).get("Column", [])

    # Step 2: Build dataset_map from column references
    dataset_map = extract_dataset(columns, datasets_dict)

    # Step 3: Build join expression string
    expression = build_expression(dataset_map)

    return expression


from utils import utils_eventlevel_parser


def event_level(pxml_event_level: dict, datasets_dict: dict):
    join_expression = utils_eventlevel_parser(pxml_event_level, datasets_dict)

    eval_context = {
        ds_name: datasets_dict[ds_name]["dataframe"]
        for ds_name in datasets_dict
    }

    result_df = eval(join_expression, {}, eval_context)
    return result_df

