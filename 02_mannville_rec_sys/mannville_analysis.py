# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Analysis on the Mannville Group Dataset

import glob
from colorsys import hsv_to_rgb
from random import uniform
import lasio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.preprocessing import binarize
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE


# %matplotlib inline

# +
# well LAS files from https://dataunderground.org/dataset/athabasca
def get_well_data(path):
    """
    This function gets the Mannville data from the LAS files
    Including, UWI, datum, ground elevation, reference elevation, and elevation units
    :param path: path to the unzipped well LAS files
    :return: dataframe with well data
    """
    wells = glob.glob(path)
    uwi = []
    datum = []
    ground = []
    eref = []
    units = []
    failure = 0
    for well in wells[0:]:
        curve = lasio.read(well)
        try:
            da = curve.params["DATM"].value
            gl = curve.params["GL"].value
            u = curve.well["UWI"].value
            units.append(curve.params["GL"].unit)
            eref.append(curve.params["EREF"].value)
            uwi.append(u.replace("W400", "W4/0"))
            datum.append(da)
            ground.append(gl)
        except:
            failure += 1
            print(well)
    print(str(failure) + " logs failed to parse")
    dataframe = pd.DataFrame(
        {"UWI": uwi, "DATUM": datum, "GL": ground, "EREF": eref, "UNIT": units}
    )
    dataframe["datum"] = np.where(
        dataframe.UNIT == "F", dataframe.EREF * 0.304, dataframe.EREF
    )
    # drop wells with no surface datum
    bad_datums = [
        "00/07-11-082-07W4/0",
        "00/10-08-095-21W4/0",
        "00/11-29-094-21W4/0",
        "AA/05-01-096-11W4/0",
        "AA/06-09-095-10W4/0",
        "AA/06-31-096-10W4/0",
        "AA/08-25-096-13W4/0",
        "AA/10-33-097-06W4/0",
        "AA/15-36-096-11W4/0",
        "AB/07-12-093-10W4/0",
        "AB/10-18-096-10W4/0",
    ]
    bad_wells = dataframe[dataframe["UWI"].isin(bad_datums)].index.values
    dataframe.drop(bad_wells, inplace=True)
    dataframe.to_csv("mann_dirty_LAS.csv")
    return dataframe


def clean_well_data():
    """
    This function cleans the output from get_well_data and saves it to a csv
    :param none:
    :return: none
    """
    well_dict = pd.read_csv(r"mann_well_dict.csv")
    dataframe = pd.read_csv("mann_dirty_LAS.csv")
    new_df = pd.merge(
        dataframe, well_dict, how="left", left_on=["UWI"], right_on=["UWI"]
    )
    new_df.dropna(inplace=True)
    tops = pd.read_csv(r"mannvillegrp_picks.csv")  # read in the top data
    tops.rename(columns={"Pick": "MD"}, inplace=True)
    new_df = pd.merge(tops, new_df, how="left", left_on=["SitID"], right_on=["SitID"])
    not_nullDF = new_df.loc[new_df["UWI"].notnull()]
    not_nullDF["TVDSS"] = (not_nullDF.datum - not_nullDF.MD).values
    not_nullDF.to_csv("mannville_cleaned.csv")


################################################


def sample_splitter(dataframe, fraction, randomseed):
    """
    Splits a dataframe into train and test subsets
    :param dataframe: Tops dataframe
    :param fraction: The fraction of tops to use for validation
    :param randomseed: The random seed for random sampling
    :return: training and testing dataframes
    """
    test = dataframe.sample(frac=fraction, random_state=randomseed)
    test_idx = test.index.values
    train = dataframe.drop(test_idx)
    return train, test


# ALS factorization modified from
# https://github.com/mickeykedia/Matrix-Factorization-ALS/blob/master/ALS%20Python%20Implementation.py
# here items are the formation and users are the well
def runALS(A, R, n_factors, n_iterations, lambda_):
    """
    Runs Alternating Least Squares algorithm in order to calculate matrix.
    :param A: User-Item Matrix with ratings
    :param R: User-Item Matrix with 1 if there is a rating or 0 if not
    :param n_factors: How many factors each of user and item matrix will consider
    :param n_iterations: How many times to run algorithm
    :param lambda_: Regularization parameter
    :return: users, items from ALS
    """
    n, m = A.shape
    np.random.seed(86)
    Users = 5 * np.random.rand(n, n_factors,)
    Items = 5 * np.random.rand(n_factors, m)

    def get_error(A, Users, Items, R):
        # This calculates the MSE of nonzero elements
        return np.sum((R * (A - np.dot(Users, Items))) ** 2) / np.sum(R)

    MAE_List = []

    print("Starting Iterations")
    for iteration in range(n_iterations):
        for i, Ri in enumerate(R):
            Users[i] = np.linalg.solve(
                np.dot(Items, np.dot(np.diag(Ri), Items.T))
                + lambda_ * np.eye(n_factors),
                np.dot(Items, np.dot(np.diag(Ri), A[i].T)),
            ).T
        for j, Rj in enumerate(R.T):
            Items[:, j] = np.linalg.solve(
                np.dot(Users.T, np.dot(np.diag(Rj), Users))
                + lambda_ * np.eye(n_factors),
                np.dot(Users.T, np.dot(np.diag(Rj), A[:, j])),
            )
        MAE_List.append(get_error(A, Users, Items, R))
    return Users, Items


# +
def cross_validation_error(dataframe, random_seed, latent_vectors, n_iters, reg):
    """
    Splits a dataframe into 4 different folds for cross validation MAE and RMSE
    :param dataframe: Tops dataframe
    :param random_seed: The random seed for random sampling
    :param latent_vectors: Number of latent vectors for matrix factorization
    :param n_iters: Number of iterations to run ALS
    :param reg: Lamda regularization value
    :return: MAE and RMSE error values for each fold
    """
    np.random.seed(random_seed)
    block_1 = dataframe.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    tops2 = dataframe.drop(block_1)
    block_2 = tops2.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    tops3 = tops2.drop(block_2)
    block_3 = tops3.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    tops4 = tops3.drop(block_3)
    block_4 = tops4.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    blocks = [block_1, block_2, block_3, block_4]
    CV_MAE = []
    CV_MSE = []
    for block in blocks:
        validate = dataframe.loc[block]
        main_group = dataframe.drop(block)
        print(f"Validating on {block.shape[0]} tops")
        D_df = main_group.pivot_table("TVDSS", "Formation", "SitID").fillna(
            0
        )  # pivot table to move into sparse matrix land
        R = D_df.values
        A = binarize(R)

        U, Vt = runALS(R, A, latent_vectors, n_iters, reg)

        recommendations = np.dot(U, Vt)  # get the recommendations

        recsys = pd.DataFrame(
            data=recommendations[0:, 0:], index=D_df.index, columns=D_df.columns
        )  # results

        newDF = recsys.T
        newDF.reset_index(inplace=True)

        flat_preds = pd.DataFrame(recsys.unstack()).reset_index()

        new_df = pd.merge(
            validate,
            flat_preds,
            how="left",
            left_on=["SitID", "Formation"],
            right_on=["SitID", "Formation"],
        )

        new_df.rename(columns={0: "SS_pred"}, inplace=True)

        cleanDF = new_df.dropna()

        cleanDF["signed_error"] = cleanDF["TVDSS"] - cleanDF["SS_pred"]

        CV_MAE.append(MAE(cleanDF.TVDSS.values, cleanDF.SS_pred.values))
        CV_MSE.append(np.sqrt(MSE(cleanDF.TVDSS.values, cleanDF.SS_pred.values)))

    return CV_MAE


def cross_validation(dataframe, random_seed, latent_vectors, n_iters, reg):
    """
    Splits a dataframe into 4 different folds for cross validation
    :param dataframe: Tops dataframe
    :param random_seed: The random seed for random sampling
    :param latent_vectors: Number of latent vectors for matrix factorization
    :param n_iters: Number of iterations to run ALS
    :param reg: Lamda regularization value
    :return: dataframe with predictions and MAE error
    """
    full = []
    block_1 = dataframe.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    tops2 = dataframe.drop(block_1)
    block_2 = tops2.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    tops3 = tops2.drop(block_2)
    block_3 = tops3.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    tops4 = tops3.drop(block_3)
    block_4 = tops4.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    blocks = [block_1, block_2, block_3, block_4]
    CV_MAE = []
    bk = 1
    for block in blocks:
        print(f"Starting Block {bk}")
        validate = dataframe.loc[block]
        main_group = dataframe.drop(block)
        print(f"Validating on {block.shape[0]} tops")
        D_df = main_group.pivot_table("TVDSS", "Formation", "SitID").fillna(
            0
        )  # pivot table to move into sparse matrix land
        R = D_df.values
        A = binarize(R)

        U, Vt = runALS(R, A, latent_vectors, n_iters, reg)

        recommendations = np.dot(U, Vt)  # get the recommendations

        recsys = pd.DataFrame(
            data=recommendations[0:, 0:], index=D_df.index, columns=D_df.columns
        )  # results

        newDF = recsys.T
        newDF.reset_index(inplace=True)

        flat_preds = pd.DataFrame(recsys.unstack()).reset_index()

        newDF = recsys.T
        newDF.reset_index(inplace=True)

        flat_preds = pd.DataFrame(recsys.unstack()).reset_index()

        new_df = pd.merge(
            validate,
            flat_preds,
            how="left",
            left_on=["SitID", "Formation"],
            right_on=["SitID", "Formation"],
        )

        new_df.rename(columns={0: "SS_pred"}, inplace=True)

        cleanDF = new_df.dropna()

        cleanDF["signed_error"] = cleanDF["TVDSS"] - cleanDF["SS_pred"]
        cleanDF["Block"] = [bk] * cleanDF.shape[0]
        well_locs = pd.read_csv(r"well_lat_lng.csv")
        full.append(cleanDF.merge(well_locs[["lat", "lng", "SitID"]], on="SitID"))
        CV_MAE.append(MAE(cleanDF.TVDSS.values, cleanDF.SS_pred.values))
        bk += 1
    output = pd.concat(full)
    return output


def cross_validation_wells(dataframe, random_seed, latent_vectors, n_iters, reg):
    """
    Splits a dataframe into 4 different folds for cross validation on each well
    :param dataframe: Tops dataframe
    :param random_seed: The random seed for random sampling
    :param latent_vectors: Number of latent vectors for matrix factorization
    :param n_iters: Number of iterations to run ALS
    :param reg: Lamda regularization value
    :return: dataframe with error for each wells predictions
    """
    cv_wells = []
    block_1 = dataframe.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    tops2 = dataframe.drop(block_1)
    block_2 = tops2.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    tops3 = tops2.drop(block_2)
    block_3 = tops3.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    tops4 = tops3.drop(block_3)
    block_4 = tops4.sample(
        n=dataframe.shape[0] // 4, random_state=random_seed
    ).index.values
    blocks = [block_1, block_2, block_3, block_4]
    f = 0
    for block in blocks:
        validate = dataframe.loc[block]
        main_group = dataframe.drop(block)
        print(f"Validating on {block.shape[0]} tops")
        D_df = main_group.pivot_table("TVDSS", "Formation", "SitID").fillna(
            0
        )  # pivot table to move into sparse matrix land
        R = D_df.values
        A = binarize(R)

        U, Vt = runALS(R, A, latent_vectors, n_iters, reg)

        recommendations = np.dot(U, Vt)  # get the recommendations

        recsys = pd.DataFrame(
            data=recommendations[0:, 0:], index=D_df.index, columns=D_df.columns
        )  # results

        newDF = recsys.T
        newDF.reset_index(inplace=True)

        flat_preds = pd.DataFrame(recsys.unstack()).reset_index()

        new_df = pd.merge(
            validate,
            flat_preds,
            how="left",
            left_on=["SitID", "Formation"],
            right_on=["SitID", "Formation"],
        )

        new_df.rename(columns={0: "SS_pred"}, inplace=True)

        cleanDF = new_df.dropna()

        cleanDF["signed_error"] = cleanDF["TVDSS"] - cleanDF["SS_pred"]
        well_locs = pd.read_csv(r"well_lat_lng.csv")
        locationDF = cleanDF.merge(well_locs[["lat", "lng", "SitID"]], on="SitID")
        aypi = []
        well_mae = []
        well_rmse = []
        east = []
        north = []
        fold = []

        print(f"foldno is {f}")
        for well in locationDF.SitID.unique():
            aypi.append(well)
            well_mae.append(
                locationDF[locationDF.SitID == well].signed_error.abs().mean()
            )
            well_rmse.append(
                np.sqrt(
                    MSE(
                        locationDF[locationDF.SitID == well].TVDSS,
                        locationDF[locationDF.SitID == well].SS_pred,
                    )
                )
            )
            east.append(locationDF[locationDF.SitID == well].lng.values[0])
            north.append(locationDF[locationDF.SitID == well].lat.values[0])
            fold.append(f)
        by_wellDF = pd.DataFrame(
            {
                "SitID": aypi,
                "Well_MAE": well_mae,
                "well_rmse": well_rmse,
                "Longitude": east,
                "Latitude": north,
                "foldno": fold,
            }
        )
        cv_wells.append(by_wellDF)
        f += 1
    return cv_wells


# -

PATH = "E:/UT Austin/Datasets/LAS_files/mannville_demo_data/*.las"

dirty = get_well_data(PATH)

clean_well_data()

# # Below are the predictions, above is cleaning

tops = pd.read_csv(r"mannville_cleaned.csv", index_col=[0])  # read in the top data
print(tops.shape)
tops.dropna(inplace=True)
tops = tops[tops.Quality >= 0]

# +
training, testing = sample_splitter(tops, 0.1, 86)

print(f"Training size is {len(training)} tops, and test size is {len(testing)} tops")

QC_D_df = training.pivot_table("TVDSS", "Formation", "SitID").fillna(
    0
)  # pivot table to move into sparse matrix land
QC_R = QC_D_df.values
QC_A = binarize(QC_R)
# -

print(
    f"{round(((QC_D_df == 0).astype(int).sum().sum())/((QC_D_df == 0).astype(int).sum().sum()+(QC_D_df != 0).astype(int).sum().sum()),3)*100} percent of the tops are missing"
)

print(
    f"There are {len(tops.UWI.unique())} wells and {len(tops.Formation.unique())} tops"
)

# # Grid Search

# %%capture
grid_search = {}
els = []
nsits = []
regulars = []
for L in range(1, 11):
    for n_it in range(10, 450, 10):
        for reg in [0.001, 0.01, 0.1, 1, 10]:
            grid_search[L, n_it, reg] = np.mean(
                cross_validation_error(tops, 86, L, n_it, reg)
            )
            els.append(L)
            nsits.append(n_it)
            regulars.append(reg)
            print(L, n_it, reg)

# L, its, lbda = min(grid_search, key=grid_search.get)
L, its, lbda = [3, 100, 0.1]

mae = cross_validation_error(tops, 86, 3, 100, 0.1)

cv_DF = cross_validation(tops, 86, L, its, lbda)

# # Adding stratigraphy
# Here we add in the ordered stratigraphy so we can make sense of things

strat_order = [
    1000,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
    9000,
    9500,
    10000,
    11000,
    12000,
    13000,
    14000,
]

# And we assign some colors. The original output from this cell is saved as `mann_color_palette.csv`

# +
color_list = []
for i in range(len(strat_order)):
    if i == 0:  # mannville
        h = uniform(23 / 360, 33 / 360)  # Select random green'ish hue from hue wheel
        s = uniform(0.8, 1)
        v = uniform(0.2, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])
    elif 0 < i <= 4:  # t61 to t31
        h = uniform(83 / 360, 158 / 360)
        s = uniform(0.8, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

    elif 4 < i <= 5:  # clw_wab
        h = uniform(23 / 360, 33 / 360)
        s = uniform(0.8, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

    elif 5 < i <= 6:  # t21
        h = uniform(83 / 360, 158 / 360)
        s = uniform(0.8, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

    elif 6 < i <= 7:  # e20
        h = uniform(23 / 360, 33 / 360)
        s = uniform(0.8, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

    elif 7 < i <= 8:  # t15
        h = uniform(83 / 360, 158 / 360)
        s = uniform(0.8, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

    elif 8 < i <= 9:  # e14
        h = uniform(23 / 360, 33 / 360)
        s = uniform(0.8, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

    elif 9 < i <= 10:  # t11
        h = uniform(83 / 360, 158 / 360)
        s = uniform(0.8, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

    elif 10 < i <= 11:  # t10.5
        h = uniform(83 / 360, 158 / 360)
        s = uniform(0.8, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

    elif 11 < i <= 12:  # e10
        h = uniform(23 / 360, 33 / 360)
        s = uniform(0.8, 1)
        v = uniform(0.2, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

    elif 12 < i <= 13:  # mcmurray
        h = uniform(50 / 360, 60 / 360)
        s = uniform(0.8, 1)
        v = uniform(0.2, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

    else:  # PZ
        h = uniform(300 / 360, 320 / 360)
        s = uniform(0.2, 1)
        v = uniform(0.3, 0.5)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r, g, b])

# colors_to_plot = dict(zip(strat_order, color_list))
# pd.DataFrame(colors_to_plot).to_csv('mann_color_palette.csv', index=False)
colors_to_plot = pd.read_csv("mann_color_palette.csv").to_dict(orient="list")
# -

for block in cv_DF["Block"].unique():
    locationDF = cv_DF[cv_DF["Block"] == block]
    mae_errors = []
    n_holdout = []
    formed = []
    colored = []
    stdev = []
    n_train = []
    rms_errors = []
    for formation in strat_order:
        form = locationDF[locationDF["Formation"] == formation]
        if form.TVDSS.values.shape[0] > 0:
            mae_errors.append(round(MAE(form.TVDSS.values, form.SS_pred.values), 1))
            formed.append(formation)
            stdev.append(np.std(form.signed_error))
            n_holdout.append(form.shape[0])
            n_train.append(tops[tops.Formation == formation].shape[0] - form.shape[0])

            rms_errors.append(np.sqrt(MSE(form.TVDSS.values, form.SS_pred.values)))
        else:
            mae_errors.append(np.nan)
            formed.append(formation)
            stdev.append(np.std(form.signed_error))
            n_train.append(tops[tops.Formation == formation].shape[0] - form.shape[0])
            n_holdout.append(form.shape[0])
            rms_errors.append(np.nan)
    table_1 = pd.DataFrame(
        {
            "Formation": formed,
            "n_train": n_train,
            "n_holdout": n_holdout,
            "MAE": mae_errors,
            "RMSE": rms_errors,
            "Std": stdev,
        }
    )
    nums = dict(zip(formed, n_holdout))
    errers = dict(zip(formed, mae_errors))
    val = dict(zip(formed, n_holdout))
    # table_1.to_csv('mann_Table_1_block_'+str(block)+'.csv')

vc = tops.Formation.value_counts(sort=True)
fm_mapping = {
    1000: "Mannville Group",
    2000: "t61",
    3000: "t51",
    4000: "t41",
    5000: "t31",
    6000: "Clearwater/Wabiskaw",
    7000: "t21",
    8000: "e20",
    9000: "t15",
    9500: "e14",
    10000: "t11",
    11000: "t10.5",
    12000: "e10",
    13000: "McMurray Formation",
    14000: "Paleozoic",
}

fig1 = plt.figure(figsize=(10, 10))
for i in enumerate(strat_order):
    width = 1
    height = 1
    lims = (0, 10)

    ax1 = fig1.add_subplot(111, aspect="equal")
    ax1.add_patch(
        patches.Rectangle(
            (0, i[0] * -1), width, height, color=colors_to_plot[str(i[1])]
        )
    )
    ax1.annotate(fm_mapping[i[1]], (1.5, -1 * i[0] + 0.25), fontsize=12)
    ax1.annotate(vc[vc.index == i[1]].values, (6, -1 * i[0] + 0.25), fontsize=12)
    plt.ylim(-59, 1)
    plt.xlim(lims)
plt.tight_layout()
plt.axis("off")
# plt.savefig('mann_strat_column.pdf')

# +
# this is output from the folds, compressed by hand in excel
errors = pd.read_csv("mann_errors.csv")
grid = plt.GridSpec(2, 1, wspace=0.4, hspace=0.1)
fig = plt.figure(figsize=(2.5, 5))

upper = plt.subplot(grid[0, :])

for formation in strat_order:
    subset = errors[errors.Formation == formation]
    upper.scatter(
        subset["n_holdout"] + subset["n_train"],
        subset["MAE"] * 3.28084,
        color=colors_to_plot[str(formation)],
        alpha=0.75,
        s=15,
    )
    upper.scatter(
        subset["n_holdout.1"] + subset["n_train.1"],
        subset["MAE.1"] * 3.28084,
        color=colors_to_plot[str(formation)],
        alpha=0.75,
        s=15,
    )
    upper.scatter(
        subset["n_holdout.2"] + subset["n_train.2"],
        subset["MAE.2"] * 3.28084,
        color=colors_to_plot[str(formation)],
        alpha=0.75,
        s=15,
    )
    up = upper.scatter(
        subset["n_holdout.3"] + subset["n_train.3"],
        subset["MAE.3"] * 3.28084,
        color=colors_to_plot[str(formation)],
        alpha=0.75,
        s=15,
    )
upper.semilogx()
upper.semilogy()
upper.set_ylabel("MAE (ft)", fontsize=6)
upper.set_title("Error and Number of Picks", fontsize=6)

upper.set_xticklabels(
    [10 ** 1, 10 ** 2, 10 ** 3], fontsize=6,
)
upper.set_yticklabels(
    [10 ^ -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3], fontsize=6
)  # plt.yticks(fontsize=6)
upper.set_xlim(100, 5000)
upper.set_ylim(1, 1500)


lower = plt.subplot(grid[1, :])

for formation in strat_order:
    subset = errors[errors.Formation == formation]
    lower.scatter(
        subset["n_holdout"] + subset["n_train"],
        subset["RMSE"] * 3.28084,
        color=colors_to_plot[str(formation)],
        alpha=0.75,
        s=15,
    )
    lower.scatter(
        subset["n_holdout.1"] + subset["n_train.1"],
        subset["RMSE.1"] * 3.28084,
        color=colors_to_plot[str(formation)],
        alpha=0.75,
        s=15,
    )
    lower.scatter(
        subset["n_holdout.2"] + subset["n_train.2"],
        subset["RMSE.2"] * 3.28084,
        color=colors_to_plot[str(formation)],
        alpha=0.75,
        s=15,
    )
    lo = lower.scatter(
        subset["n_holdout.3"] + subset["n_train.3"],
        subset["RMSE.3"] * 3.28084,
        color=colors_to_plot[str(formation)],
        alpha=0.75,
        s=15,
    )
lower.semilogx()
lower.semilogy()
lower.set_xlabel("Number of top picks of each formation", fontsize=6)
lower.set_ylabel("RMSE (ft)", fontsize=6)
# lower.set_title('Root Mean Squared Error per Formation', fontsize=6)
lower.set_xticklabels(
    [10 ** 1, 10 ** 2, 10 ** 3], fontsize=6,
)
lower.set_yticklabels(
    [10 ^ -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3], fontsize=6
)  # plt.yticks(fontsize=6)
lower.set_xlim(100, 5000)
lower.set_ylim(1, 1500)


# plt.savefig('mann_RMSE_formation.pdf')
# -

# # Error by well

masterDF = pd.concat(cross_validation_wells(tops, 86, L, its, 0.1))
# masterDF.to_csv('mann_error_map.csv')
