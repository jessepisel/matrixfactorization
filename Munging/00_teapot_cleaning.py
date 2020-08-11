# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Initial Munging
# This notebook takes two `.csv` files and combines the Teapot Dome well headers with the formation top picks `teapot.csv` to create a cleaned file `teapot_clean.csv`

# %%
import pandas as pd

HEADERS = pd.read_csv("TeapotDomeWellHeaders.csv")
HEADERS.rename(columns={"API Number": "API"}, inplace=True)

TOPS = pd.read_csv(r"teapot.csv")

ELEVATIONS = []
GROUND_LEVEL = []
DATUM_TYPE = []
API = []
FORMATION = []
MD = []
for well in range(TOPS.shape[0]):
    well_api = TOPS.iloc[well].API
    row = HEADERS[HEADERS.API == well_api]
    if row.shape[0] > 0:
        ELEVATIONS.append(row["Datum Elevation"].values[0])
        GROUND_LEVEL.append(row["Ground Elevation"].values[0])
        DATUM_TYPE.append(row["Datum Type"].values[0])
        API.append(well_api)
        FORMATION.append(TOPS.iloc[well].Formation)
        MD.append(TOPS.iloc[well].MD)

TEA = pd.DataFrame(
    {
        "API": API,
        "Formation": FORMATION,
        "MD": MD,
        "GL": GROUND_LEVEL,
        "DELEV": ELEVATIONS,
        "DT": DATUM_TYPE,
    }
)

TVDSS = []
for well in range(TEA.shape[0]):
    row = TEA.iloc[well]
    TVDSS.append(row.GL - row.MD - (row.DELEV - row.GL))

TEA["TVDSS"] = TVDSS
TEA.to_csv(r"teapot_clean.csv")

# %%
