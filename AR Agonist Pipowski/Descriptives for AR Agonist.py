import os
import matplotlib.pyplot as plt
import pandas as pd

# --- Configuration & File Checking ---
file_path = "LUC_Master_dataset_expanded.csv"
output_image_assay = "LUC_Master_dataset_summary_table.png"
output_image_missing = "LUC_Master_dataset_missing_values.png"
output_csv_both = "LUC_Master_dataset_active_in_both.csv"
output_image_both = "LUC_Master_dataset_active_both_table.png"

if not os.path.exists(file_path):
  print(f"Error: The required file '{file_path}' was not found.")
  print("Please ensure the dataset file is in the correct directory.")
  exit()

# --- Load the Dataset ---
print(f"Loading {file_path}...")
df = pd.read_csv(file_path)
total_rows = len(df)


# =========================================================
# TABLE 1: Chemical Counts & Assay Summary
# =========================================================
print("Generating Assay Summary Table image...")
assays = {
    "Agonist Assay": "HIT CALL_Agonist",
    "Viability Assay": "HIT CALL_Viability",
}

table_data_assay = [
    ["Category", "Chemical Status", "Count", "Percentage"],
    [
        "Dataset Overview",
        "Total Unique Chemicals",
        f"{total_rows:,}",
        "100.00%",
    ],
]

status_order = ["Active", "Inactive", None]

for assay_name, col_name in assays.items():
  if col_name in df.columns:
    counts = df[col_name].value_counts(dropna=False)
    for idx, status in enumerate(status_order):
      count = 0
      if pd.isna(status):
        if pd.isna(counts.index).any():
          count = counts[counts.index.isna()].iloc[0]
      else:
        if status in counts.index:
          count = counts[status]

      status_label = "Missing / NaN" if pd.isna(status) else status
      percentage = (count / total_rows) * 100
      display_category = assay_name if idx == 1 else ""
      table_data_assay.append(
          [display_category, status_label, f"{count:,}", f"{percentage:.2f}%"]
      )
  else:
    table_data_assay.append([assay_name, "Column Not Found", "N/A", "N/A"])

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_position([0.05, 0.05, 0.90, 0.82])
ax.axis("off")
ax.axis("tight")

fig.suptitle(
    "LUC Master Dataset Chemical Counts & Assay Summary",
    fontsize=16,
    fontweight="bold",
    color="#1B365D",
    y=0.93,
)

table1 = ax.table(
    cellText=table_data_assay, colLabels=None, loc="center", cellLoc="center"
)
table1.auto_set_font_size(False)
table1.set_fontsize(11)
table1.scale(1.2, 2.0)

for (row, col), cell in table1.get_celld().items():
  cell.set_edgecolor("#D1D5DB")
  if row == 0:
    cell.set_text_props(fontweight="bold", color="white")
    cell.set_facecolor("#1B365D")
  else:
    cell.set_facecolor("#F4F8FB")
    if col == 0:
      cell.set_facecolor("#EBF3FA")
      cell.set_text_props(fontweight="bold", color="#1B365D")
      if row in [2, 5]:
        cell.visible_edges = "TRL"
      elif row in [3, 6]:
        cell.visible_edges = "RL"
      elif row in [4, 7]:
        cell.visible_edges = "BRL"
    if col == 1:
      text = cell.get_text().get_text()
      if text == "Active":
        cell.set_text_props(color="#0E7490", fontweight="bold")
      elif text == "Missing / NaN":
        cell.set_text_props(color="#6B7280")

plt.savefig(
    output_image_assay, dpi=300, bbox_inches="tight", pad_inches=0.3
)
plt.close(fig)
print(f"✅ Created: '{output_image_assay}'")


# =========================================================
# TABLE 2: Missing Values by Column
# =========================================================
print("Generating Missing Values Summary Table image...")
missing_counts = df.isnull().sum()
missing_percent = (missing_counts / total_rows) * 100

missing_df = pd.DataFrame(
    {"Missing Count": missing_counts, "Missing Percentage": missing_percent}
)
missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
    by="Missing Count", ascending=False
)

table_data_missing = [
    ["Column Name", "Missing Count (N)", "Missing Percentage (%)"]
]
for col_name, row in missing_df.iterrows():
  count = int(row["Missing Count"])
  percentage = row["Missing Percentage"]
  table_data_missing.append([str(col_name), f"{count:,}", f"{percentage:.2f}%"])

fig_height = max(6, len(table_data_missing) * 0.35)
fig, ax = plt.subplots(figsize=(10, fig_height))
ax.set_position([0.05, 0.05, 0.90, 0.82])
ax.axis("off")
ax.axis("tight")

fig.suptitle(
    "LUC Master Dataset — Missing Data Summary by Column",
    fontsize=16,
    fontweight="bold",
    color="#1B365D",
    y=0.93,
)

table2 = ax.table(
    cellText=table_data_missing, colLabels=None, loc="center", cellLoc="center"
)
table2.auto_set_font_size(False)
table2.set_fontsize(10)
table2.scale(1.2, 1.8)

for (row, col), cell in table2.get_celld().items():
  cell.set_edgecolor("#D1D5DB")
  if row == 0:
    cell.set_text_props(fontweight="bold", color="white")
    cell.set_facecolor("#1B365D")
  else:
    cell.set_facecolor("#F4F8FB")
    if col == 0:
      cell.set_text_props(ha="left", fontweight="bold", color="#1B365D")
    else:
      cell.set_text_props(ha="center", color="#333333")

plt.savefig(
    output_image_missing, dpi=300, bbox_inches="tight", pad_inches=0.3
)
plt.close(fig)
print(f"✅ Created: '{output_image_missing}'")


# =========================================================
# PART 3: Filter Chemicals Active in BOTH Assays & Export CSV
# =========================================================
print(
    "Filtering chemicals active in both Agonist and Viability assays..."
)
active_both_df = df[
    (df["HIT CALL_Agonist"] == "Active")
    & (df["HIT CALL_Viability"] == "Active")
].copy()

# Clean DTXSID display for table (remove full URL if desired, or keep as is)
active_both_df.to_csv(output_csv_both, index=False)
print(
    f"✅ Created CSV file for dual-active chemicals: '{output_csv_both}'"
    f" (Total: {len(active_both_df)} chemicals)"
)


# =========================================================
# TABLE 3: Chemicals Active in Both Assays
# =========================================================
print("Generating Dual-Active Chemicals Table image...")

table_data_both = [
    [
        "No.",
        "DTXSID",
        "Preferred Name",
        "CASRN",
        "Molecular Formula",
    ]
]

for idx, (_, row) in enumerate(active_both_df.iterrows(), start=1):
  # Shorten DTXSID URL for cleaner table display
  dtx_raw = str(row["DTXSID"])
  dtx_short = dtx_raw.split("/")[-1] if "/" in dtx_raw else dtx_raw

  name = (
      str(row["PREFERRED NAME"])
      if pd.notna(row["PREFERRED NAME"])
      else "N/A"
  )
  cas = str(row["CASRN"]) if pd.notna(row["CASRN"]) else "N/A"
  formula = (
      str(row["MOLECULAR FORMULA"])
      if pd.notna(row["MOLECULAR FORMULA"])
      else "N/A"
  )

  table_data_both.append([str(idx), dtx_short, name, cas, formula])

fig_height_both = max(6, (len(table_data_both) + 1) * 0.35)
fig, ax = plt.subplots(figsize=(11, fig_height_both))
ax.set_position([0.05, 0.05, 0.90, 0.82])
ax.axis("off")
ax.axis("tight")

fig.suptitle(
    f"Chemicals Active in BOTH Assays (Total: {len(active_both_df)})",
    fontsize=16,
    fontweight="bold",
    color="#1B365D",
    y=0.93,
)

table3 = ax.table(
    cellText=table_data_both, colLabels=None, loc="center", cellLoc="center"
)
table3.auto_set_font_size(False)
table3.set_fontsize(10)
table3.scale(1.2, 1.8)

for (row, col), cell in table3.get_celld().items():
  cell.set_edgecolor("#D1D5DB")
  if row == 0:
    cell.set_text_props(fontweight="bold", color="white")
    cell.set_facecolor("#1B365D")
  else:
    cell.set_facecolor("#F4F8FB")
    if col in [1, 2]:  # DTXSID and Name
      cell.set_text_props(ha="left", fontweight="bold", color="#1B365D")
    else:
      cell.set_text_props(ha="center", color="#333333")

plt.savefig(output_image_both, dpi=300, bbox_inches="tight", pad_inches=0.3)
plt.close(fig)
print(f"✅ Created: '{output_image_both}'")

print("\nProcessing complete! All files and summary table images generated.")