import pandas as pd
import numpy as np
import os
import shutil

common_features = [
    "actual_time",
    "alias",
    "assigned_to",
    "assigned_to_detail",
    "blocks",
    "cc",
    "cc_detail",
    "classification",
    "component",
    "creation_time",
    "creator",
    "creator_detail",
    "deadline",
    "depends_on",
    "dupe_of",
    "estimated_time",
    "flags",
    "groups",
    "id",
    "is_cc_accessible",
    "is_confirmed",
    "is_open",
    "is_creator_accessible",
    "keywords",
    "last_change_time",
    "op_sys",
    "platform",
    "priority",
    "product",
    "qa_contact",
    "qa_contact_detail",
    "remaining_time",
    "resolution",
    "see_also",
    "severity",
    "status",
    "summary",
    "target_milestone",
    "update_token",
    "url",
    "version",
    "whiteboard",
]

important_features = [
    "classification",
    "op_sys",
    "platform",
    "severity",
    "status",
    "summary",
]

final_features = [
    "classification",
    "component",
    "depends_on",
    "dupe_of",
    "flags",
    "groups",
    "id",
    "is_open",
    "keywords",
    "op_sys",
    "platform",
    "priority",
    "product",
    "resolution",
    "see_also",
    "severity",
    "status",
    "summary",
    "url",
    "whiteboard",
]

def clean_bugs(dataset_path: str) -> None:
    files = os.listdir(dataset_path)

    final_data = []
    lines_added = 0
    lines_not_added = 0

    for f in files:
        print(f"INFO: processing file {f}")
        df = pd.read_csv(f"{dataset_path}/{f}")

        df_columns = df.columns
        for cc in common_features:
            if cc not in df_columns:
                df[cc] = np.nan

        for i, bug in enumerate(df[common_features].values):
            bug_data = bug.tolist()

            bug_important_values = df[important_features].T[i].tolist()
            if bug_important_values.count(np.nan) > 0 or bug_important_values.count('[]') > 0 or bug_important_values.count('---') > 0:
                lines_not_added += 1
                df.drop([i], axis=0, inplace=True)
                continue

            nan_count = bug_data.count(np.nan)
            empty_array_count = bug_data.count('[]')
            empty_string_count = bug_data.count('---')

            if nan_count + empty_array_count + empty_string_count < bug.size / 2:
                lines_added += 1
            else:
                lines_not_added += 1
                df.drop([i], axis=0, inplace=True)
        
        final_data.append(df)
        
    final_df = pd.concat(final_data)
    final_df[final_features].to_csv("final_dataset.csv", index=False)
    
    print(f"""

Rows added: {lines_added}
Rows cut off: {lines_not_added}
    """)