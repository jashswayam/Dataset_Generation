import re
from typing import List, Dict

class UtilsEventLevelParser:
    def __init__(self, pxml_event_level: dict, datasets_dict: dict):
        self.pxml_event_level = pxml_event_level
        self.datasets_dict = datasets_dict

    def extract_dataset(self, event_columns: List[str]) -> Dict[str, List[str]]:
        dataset_map = {}
        for item in event_columns:
            ds, col = item.split(":", maxsplit=1)
            keys = self.datasets_dict.get(ds, {}).get("joining_keys", [])
            dataset_map.setdefault(ds, keys)
        return dataset_map

    def extract_ds_columns(self, dataset_map: str) -> Dict[str, set]:
        # Only match ds-prefixed dataset references like ds1.colX
        pattern = r'\b(ds\d+)\.(\w+)\b'
        matches = re.findall(pattern, dataset_map)
        selective_dataset = {}
        for ds, col in matches:
            selective_dataset.setdefault(ds, set()).add(col)
        return selective_dataset

    def build_expression(self, dataset_map: dict) -> str:
        datasets = list(dataset_map.keys())
        expr = f"{datasets[0]}"
        for ds in range(1, len(datasets)):
            left_key = dataset_map.get(datasets[ds])
            right_key = dataset_map.get(datasets[ds - 1])
            expr += f".join({datasets[ds]}, left_on={left_key}, " \
                    f"right_on={right_key}, coalesce=True, suffix='_' + {datasets[ds]})"
        return expr

    def parse(self) -> str:
        columns = self.pxml_event_level.get("RetainedColumns", {}).get("Column", [])
        dataset_map = self.extract_dataset(columns)
        expression = self.build_expression(dataset_map)
        return expression


def event_level(pxml_event_level: dict, datasets_dict: dict):
    # Prepare the parser
    parser = UtilsEventLevelParser(pxml_event_level, datasets_dict)
    
    # Generate the expression string
    join_expression = parser.parse()
    
    # Prepare evaluation context with actual DataFrames
    eval_context = {
        ds_name: datasets_dict[ds_name]["dataframe"]
        for ds_name in datasets_dict
    }
    
    # Evaluate the join expression
    result_df = eval(join_expression, {}, eval_context)
    
    return result_df