# -*- coding: utf-8 -*-
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

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
import seaborn as sns
from colorsys import hsv_to_rgb
from random import randint, uniform, seed
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

# %%
%matplotlib inline


# %% [markdown]
# First we need to define a sample splitter, ALS matrix factorization, and a cross-validation with mean absolute error

# %%
def sample_splitter(dataframe, fraction, randomseed):
    """
    Splits a dataframe into train and test subsets
    :param dataframe: Tops dataframe
    :param fraction: The fraction of tops to use for validation
    :param randomseed: The random seed for random sampling
    """
    test = dataframe.sample(frac=fraction, random_state=randomseed)
    test_idx = test.index.values
    train =  dataframe.drop(test_idx)
    return train, test

#ALS factorization modified from 
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
    :return:
    """
    lambda_ = lambda_
    n_factors = n_factors
    n, m = A.shape
    n_iterations = n_iterations
    np.random.seed(86)
    Users = 5 * np.random.rand(n, n_factors, )
    Items = 5 * np.random.rand(n_factors, m)

    def get_error(A, Users, Items, R):
        # This calculates the MSE of nonzero elements
        return np.sum((R * (A - np.dot(Users, Items))) ** 2) / np.sum(R)

    MAE_List = []

    print("Starting Iterations")
    for iter in range(n_iterations):
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


def cross_validation_MAE(dataframe, random_seed, latent_vectors, n_iters, reg):
    """
    Splits a dataframe into 4 different folds for cross validation with mean absolute error (MAE)
    :param dataframe: Tops dataframe
    :param random_seed: The random seed for random sampling
    :param latent_vectors: Number of latent vectors for matrix factorization
    :param n_iters: Number of iterations to run ALS
    :param reg: Lamda regularization value

    """
    full = []
    block_1  = dataframe.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    tops2 = dataframe.drop(block_1)
    block_2 = tops2.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    tops3 = tops2.drop(block_2)
    block_3 = tops3.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    tops4 = tops3.drop(block_3)
    block_4 = tops4.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    blocks = [block_1, block_2, block_3, block_4]
    CV_MAE = []
    bk = 1
    for block in blocks:
        print(f'Starting Block {bk}')
        validate = dataframe.loc[block]
        main_group = dataframe.drop(block)
        print(f'Validating on {block.shape[0]} tops')
        D_df = main_group.pivot_table("SS", "Formation", "API").fillna(0)#pivot table to move into sparse matrix land
        R = D_df.values
        A = binarize(R) 

        
        U, Vt = runALS(R, A, latent_vectors, n_iters, reg)

        recommendations = np.dot(U, Vt) #get the recommendations

        recsys = pd.DataFrame(
            data=recommendations[0:, 0:], index=D_df.index, columns=D_df.columns
        ) #results

        newDF = recsys.T
        newDF.reset_index(inplace=True)

        flat_preds = pd.DataFrame(recsys.unstack()).reset_index()

        new_df = pd.merge(validate, flat_preds,  how='left', left_on=['API','Formation'], right_on = ['API','Formation'])

        new_df.rename(columns={0:'SS_pred'}, inplace=True)

        cleanDF = new_df.dropna()

        cleanDF['signed_error'] = (cleanDF['SS'] - cleanDF['SS_pred'])
        cleanDF['Block'] = [bk]*cleanDF.shape[0]
        well_locs.rename(columns={'API Number':'API'}, inplace=True)

        full.append(cleanDF.merge(well_locs[['Northing', 'Easting', 'API']], on='API'))
        CV_MAE.append(MAE(cleanDF.SS.values-ssmin, cleanDF.SS_pred.values-ssmin))

        bk += 1
    output = pd.concat(full)
    return CV_MAE


# %% [markdown]
# Here we read in the munged dataset and well locations. We rename a column, and standardize the TVDSS

# %%
tops = pd.read_csv("teapot_clean.csv", index_col=[0]) #read in the top data
tops.rename(columns={'TVDSS':'SS'}, inplace=True)
well_locs = pd.read_csv(r'../Munging/TeapotDomeWellHeaders.csv')
ssmin = tops.SS.min()
tops.SS = tops.SS - ssmin #standardize the subsea values

# %% [markdown]
# Here we run a quick QC, and split the data into test train subsets

# %%
tops.dropna(inplace=True)
train, test = sample_splitter(tops, 0.2, 86)
print(f'Training size is {len(train)} tops, and test size is {len(test)} tops')
#pivot table to move into sparse matrix land
D_df = train.pivot_table("SS", "Formation", "API").fillna(0)
R = D_df.values
A = binarize(R) 
print(f'{round(((D_df == 0).astype(int).sum().sum())/((D_df == 0).astype(int).sum().sum()+(D_df != 0).astype(int).sum().sum()),3)*100} percent of the tops are missing')

# %% [markdown]
# # Grid Search
# This section takes a while to run, could be made more efficient with dask

# %%
%%capture
grid_search = {}
els = []
nsits = []
regulars = []
for L in range(1,11):
    
    for n_it in range(10,450,10):
        for reg in [0.001,0.01,0.1,1,10]:
            grid_search[L,n_it,reg] = np.mean(cross_validation(tops, 86, L, n_it, reg))
            els.append(L)
            nsits.append(n_it)
            regulars.append(reg)
            print(L, n_it, reg)

# %% [markdown]
# Here we assign the optimal number of latent features, iterations, and lambda value. the second line sets it to the searched value from above.

# %%
# uncomment if you ran grid search
#L, its, regs = min(grid_search, key=grid_search.get)
L, its, regs = [2, 290, 0.1]

# %%
print(L, its, regs)

# %% [markdown]
# # delete?

# %%
#errorDF = pd.DataFrame({'L':els, 'iterations':nsits,  'MAE':list(grid_search.values())})
errorDF = pd.read_csv('gridsearch2.csv', index_col=[0])

errorDF.insert(2, 'regularization', [0.001,0.01,0.1,1,10]*440)

# %% jupyter={"outputs_hidden": true}
import seaborn as sns
ax0 = sns.boxplot(x="iterations", y="MAE", data=errorDF, hue='L')
plt.xlabel('Number of Latent Factors')
plt.ylabel('MAE (ft)')
plt.savefig('latent_factors.svg')

# %% jupyter={"outputs_hidden": true}
top100 = errorDF.sort_values("MAE").iloc[:40, :]
#plt.scatter(top100.L, top100.MAE)
#plt.scatter(top100.iterations, top100.MAE)
plt.scatter(top100.regularization, top100.MAE)

# %% jupyter={"outputs_hidden": true}
plt.scatter(errorDF.L, errorDF.MAE)
#plt.scatter(errorDF.regularization, errorDF.MAE)
#plt.scatter(errorDF.iterations, errorDF.MAE)

# %% jupyter={"outputs_hidden": true}
ax1 = sns.boxplot(x="L", y="MAE", data=errorDF)
plt.xlabel('Number of Latent Features')
plt.ylabel('MAE (ft)')
plt.savefig('features.svg')

# %% jupyter={"outputs_hidden": true}
ax1 = sns.boxplot(x="iterations", y="MAE", data=errorDF)
plt.xlabel('Number of iterations')
plt.ylabel('MAE (ft)')
plt.savefig('iterations.svg')

# %% jupyter={"outputs_hidden": true}
ax1 = sns.boxplot(x="regularization", y="MAE", data=errorDF)
plt.xlabel('Regularization Parameter')
plt.ylabel('MAE (ft)')
plt.savefig('regularization.svg')

# %% jupyter={"outputs_hidden": true}
cv_DF = cross_validation_MAE(tops, 86, 2, 290, 0.1)

# %% [markdown]
# # Adding stratigraphy
# Here we add in the ordered stratigraphy so we can make sense of things

# %%
strat_order = ['SSXS', 'SSXbase','SHNNu', 'SHNNl','SHNNbs','StBR','StFT','StGD','StAM','NBRRws','NBRRsg','CRLL',
                'F1WC',
'F1WCBench1Base', 'F1WCBench2Top', 'F1WCBench2Base', 'F1WCBench3Top',
'F1WCbase','B1','F2WC','F2WCbase',
'F3WC',
'F3WCbase',
'MWRY',
'MDDY',
'THRM',
'DKOT',
'LKOT', # base cret
'MRSN',
'SNDCu',# upper sundance
'CNSP', # base jurassic canyon spring
'CRMT', # Crow mountain, Chugwater Group/FM
'ALCV', # Alcova ls, chugwater
'RDPK', # base Triassic chugwater
'ERVY', # goose egg member
'FRLL', # goose egg
'GLND', # goose egg
'MNKT', # goose egg
'OPCH', # goose egg Base Permian
'A Sand', # Tensleep 
'B Dolo',
'B Sand',
'C1 Dolo',
'C1 Sand',
'C2 Dolo',
'C2 Sand',
'C3 Dolo',
'C3 Sand',
'C4 Dolo',
'C4 Sand',
'D Dolo',
'D Sand',
'E Dolo',
'AMSD', # base Pennsylvanian
'MDSN', # base mississippian
'PC'
]

# %% [markdown]
# And we assign some colors. The original output from this cell is saved as `color_palette.csv`

# %%
color_list = []
for i in range(len(strat_order)):
    if i <=29: #cretaceous
        h = uniform(83/360, 158/360) # Select random green'ish hue from hue wheel
        s = uniform(0.8, 1)
        v = uniform(0.2, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r,g,b])
    elif 29<i<=32: #Jurassic
        h = uniform(138/360, 200/360) 
        s = uniform(0.8, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r,g,b])

    elif 32<i<=35: #triassic
        h = uniform(156/360, 180/360) 
        s = uniform(0.2, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r,g,b])

    elif 35<i<= 40: #permian
        h = uniform(190/360, 200/360) 
        s = uniform(0.2, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r,g,b])

    elif 40<i<= 55: #Pennsylvanian
        h = uniform(210/360, 230/360) 
        s = uniform(0.2, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r,g,b])

    elif 55<i<= 56: #missippian
        h = uniform(240/360, 260/360) 
        s = uniform(0.2, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r,g,b])

    else: #PC
        h = uniform(300/360, 320/360) 
        s = uniform(0.2, 1)
        v = uniform(0.3, 1)
        r, g, b = hsv_to_rgb(h, s, v)
        color_list.append([r,g,b])

colors_to_plot = dict(zip(strat_order, color_list))

# %%
# uncomment if you want to create new colors, otherwise just load from csv
#pd.DataFrame(colors_to_plot).to_csv('color_palette.csv', index=False)
colors_to_plot =  pd.read_csv('color_palette.csv').to_dict(orient='list')

# %%
subset = cv_DF[cv_DF.Block == 1]
subset2 = cv_DF[cv_DF.Block == 2]
subset3 = cv_DF[cv_DF.Block == 3]
subset4 = cv_DF[cv_DF.Block == 4]

# %%
## import seaborn as sns
plt.figure(figsize=(2.25,4.25))
sns.set_palette(list(colors_to_plot.values()))

sns.stripplot(x=subset4.signed_error, y=subset4.Formation, order=strat_order, 
              )
plt.xlim(-500,500)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.xlabel('Signed error (ft)', fontsize=6)
plt.xlabel('Formation', fontsize=6)
plt.title('Fold 4', fontsize=10)
#plt.savefig('fold.svg')

# %%
for block in cv_DF['Block'].unique():
    locationDF = cv_DF[cv_DF['Block'] == block]
    mae_errors = []
    n_holdout = []
    formed = []
    colored = []
    stdev = []
    n_train = []
    rms_errors = []
    for formation in strat_order:
        form = locationDF[locationDF['Formation'] == formation]
        if form.SS.values.shape[0] > 0:
            mae_errors.append(round(MAE(form.SS.values-ssmin, form.SS_pred.values-ssmin),1))
            formed.append(formation)
            colored.append(colors_to_plot[formation])
            stdev.append(np.std(form.signed_error))
            n_holdout.append(form.shape[0])
            n_train.append(tops[tops.Formation == formation].shape[0]-form.shape[0])

            rms_errors.append(np.sqrt(MSE(form.SS.values-ssmin, form.SS_pred.values-ssmin)))
        else:
          #  print(str(formation)+' has no holdout samples'+' n picks='+str(tops[tops.Formation == formation].shape[0]))
            mae_errors.append(np.nan)
            formed.append(formation)
            colored.append(colors_to_plot[formation])
            stdev.append(np.std(form.signed_error))
            n_train.append(tops[tops.Formation == formation].shape[0]-form.shape[0])
            n_holdout.append(form.shape[0])
            rms_errors.append(np.nan)
    table_1 = pd.DataFrame({'Formation':formed, 'n_train': n_train, 'n_holdout':n_holdout, 'MAE':mae_errors,
                 'RMSE':rms_errors, 'Std':stdev})
    nums = dict(zip(formed, n_holdout))
    errers = dict(zip(formed, mae_errors))
    val = dict(zip(formed, n_holdout))
    table_1.to_csv('Table_1_block_'+str(block)+'.csv')


# %% [markdown]
# The file output of the cell above is manually combined into `errors.csv` and read in to the next cell

# %%
errors = pd.read_csv('errors.csv')

grid = plt.GridSpec(2, 1, wspace=0.4, hspace=0.1)
fig = plt.figure(figsize=(2.5,5))

upper = plt.subplot(grid[0, :])

for formation in strat_order:
    subset = errors[errors.Formation == formation]
    upper.scatter(subset['n_holdout']+subset['n_train'], subset['MAE']*3.28084, 
                color=colors_to_plot[str(formation)], alpha=0.75, s=15)
    upper.scatter(subset['n_holdout.1']+subset['n_train.1'], subset['MAE.1']*3.28084,
                color=colors_to_plot[str(formation)], alpha=0.75, s=15)
    upper.scatter(subset['n_holdout.2']+subset['n_train.2'], subset['MAE.2']*3.28084,
                color=colors_to_plot[str(formation)], alpha=0.75, s=15)
    up = upper.scatter(subset['n_holdout.3']+subset['n_train.3'], subset['MAE.3']*3.28084,
                color=colors_to_plot[str(formation)], alpha=0.75, s=15)
upper.semilogx()
upper.semilogy()
#upper.set_xlabel('Number of top picks of each formation', fontsize=6)
upper.set_ylabel('MAE (ft)', fontsize=6)
upper.set_title('Error and Number of Picks', fontsize=6)

upper.set_xticklabels([10**-2,10**-1,10**1, 10**3],fontsize=6,)
upper.set_yticklabels([10^-1,10**0, 10**1, 10**2, 10**3],fontsize=6)#plt.yticks(fontsize=6)
upper.set_xlim(0.1,5000)
upper.set_ylim(1,2000)


lower = plt.subplot(grid[1, :])

for formation in strat_order:
    subset = errors[errors.Formation == formation]
    lower.scatter(subset['n_holdout']+subset['n_train'], subset['RMSE']*3.28084, 
                color=colors_to_plot[str(formation)], alpha=0.75, s=15)
    lower.scatter(subset['n_holdout.1']+subset['n_train.1'], subset['RMSE.1']*3.28084,
                color=colors_to_plot[str(formation)], alpha=0.75, s=15)
    lower.scatter(subset['n_holdout.2']+subset['n_train.2'], subset['RMSE.2']*3.28084,
                color=colors_to_plot[str(formation)], alpha=0.75, s=15)
    lo = lower.scatter(subset['n_holdout.3']+subset['n_train.3'], subset['RMSE.3']*3.28084,
                color=colors_to_plot[str(formation)], alpha=0.75, s=15)
lower.semilogx()
lower.semilogy()
lower.set_xlabel('Number of top picks of each formation', fontsize=6)
lower.set_ylabel('RMSE (ft)', fontsize=6)
#lower.set_title('Root Mean Squared Error per Formation', fontsize=6)
lower.set_xticklabels([10**-2,10**-1,10**1, 10**3],fontsize=6,)
lower.set_yticklabels([10^-1,10**0, 10**1, 10**2, 10**3],fontsize=6)#plt.yticks(fontsize=6)
lower.set_xlim(0.1,5000)
lower.set_ylim(1,2000)


#plt.savefig('RMSE_formation.pdf')

# %%
U, Vt = runALS(R, A, L, its, regs)

recommendations = np.dot(U, Vt) #get the recommendations

recsys = pd.DataFrame(
    data=recommendations[0:, 0:], index=D_df.index, columns=D_df.columns
) #results

newDF = recsys.T
newDF.reset_index(inplace=True)

flat_preds = pd.DataFrame(recsys.unstack()).reset_index()

new_df = pd.merge(test, flat_preds,  how='left', left_on=['API','Formation'], right_on = ['API','Formation'])

new_df.rename(columns={0:'SS_pred'}, inplace=True)

cleanDF = new_df.dropna()

cleanDF['signed_error'] = (cleanDF['SS'] - cleanDF['SS_pred'])

# %%
well_locs = pd.read_csv(r'Munging/TeapotDomeWellHeaders_outliers_JRP.csv')
well_locs.rename(columns={'API Number':'API'}, inplace=True)

locationDF = cleanDF.merge(well_locs[['Northing', 'Easting', 'API']], on='API')
flat_preds.rename(columns={0:'SS_pred'}, inplace=True)
flat_preds['SS_pred'] = flat_preds['SS_pred'] + ssmin

# %%
flat_preds[flat_preds.API == 490251103200].to_csv('upper_well.csv') # upper well 
flat_preds[flat_preds.API == 490252319500].to_csv('middle_well.csv') # middle well
flat_preds[flat_preds.API == 490250633800].to_csv('lower_well.csv') # lower well

# %%
fig, ax = plt.subplots(1,1, figsize=(2,2))

for formation in strat_order:
    form = locationDF[locationDF['Formation'] == formation]
    plt.hist(form.signed_error, range=(-200,200), bins=40, stacked=True, color=colors_to_plot[formation])
plt.vlines(np.std(locationDF.signed_error), 0,90, linestyle='dashed') #95% of the picks are within these
plt.vlines(-1*np.std(locationDF.signed_error), 0,90, linestyle='dashed', label='1 σ')
plt.xlabel('Signed Error', fontsize=6)
plt.ylabel('Number of counts', fontsize=6)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)
plt.ylim(0,50)
plt.legend(fontsize=6)
plt.tight_layout()
plt.savefig('Figure 5.svg')

# %%
# TODO: recreate the number of picks vs error plot for each block 
nums = dict(zip(formed, n_picks))
errers = dict(zip(formed, mae_errors))
val = dict(zip(formed, n_holdout))
fig, ax = plt.subplots(1,1, figsize=(10,10))

for formation in strat_order:
        plt.scatter(nums[formation], errers[formation], color=colors_to_plot[formation])
plt.semilogx()
plt.semilogy()
#plt.xlabel('Number of top picks of each formation', fontsize=6)
#plt.ylabel('MAE (ft)', fontsize=6)
#plt.xticks(fontsize=6)
#plt.yticks(fontsize=6)
plt.xlim(0.5, 2000)
plt.ylim()
plt.tight_layout()

# %%
locationDF_full = tops.merge(well_locs[['Northing', 'Easting', 'API']], on='API')
predictions = flat_preds.merge(well_locs[['Northing', 'Easting', 'API']], on='API')

signed = cleanDF.merge(well_locs[['Northing', 'Easting', 'API']], on='API')

# %% [markdown]
# This is test size error calculation

# %%
%%capture
big_de_error = []
for ran in range(100):
    de_error = []
    for i in np.arange(0.99, 0, -0.1):
        train, test = sample_splitter(tops, i, ran)
        print(f'Training size is {len(train)} tops, and test size is {len(test)} tops')
        D_df = train.pivot_table("SS", "Formation", "API").fillna(0)#pivot table to move into sparse matrix land
        R = D_df.values
        from sklearn.preprocessing import binarize
        A = binarize(R) 
        U, Vt = runALS(R, A, 3, 20, 0.1)
        recommendations = np.dot(U, Vt) #get the recommendations
        recsys = pd.DataFrame(
            data=recommendations[0:, 0:], index=D_df.index, columns=D_df.columns
        ) #results
        newDF = recsys.T
        newDF.reset_index(inplace=True)
        #test = newDF[["FORT UNION", "API"]]
        flat_preds = pd.DataFrame(recsys.unstack()).reset_index()
        new_df = pd.merge(test, flat_preds,  how='left', left_on=['API','Formation'], right_on = ['API','Formation'])
        new_df.rename(columns={0:'SS_pred'}, inplace=True)
        cleanDF = new_df.dropna()
        cleanDF['signed_error'] = (cleanDF['SS'] - cleanDF['SS_pred'])
        de_error.append(np.sqrt(MSE(cleanDF.SS.values-ssmin, cleanDF.SS_pred.values-ssmin)))

        locationDF = cleanDF.merge(well_locs[['Northing', 'Easting', 'API']], on='API')
    big_de_error.append(de_error)
    
'''
        for formation in strat_order:
            form = locationDF[locationDF['Formation'] == formation]
            plt.hist(form.signed_error, range=(-200,200), bins=40, stacked=True, color=colors_to_plot[formation])
        plt.vlines(locationDF.quantile(0.05).signed_error, 0,80, linestyle='dashed') #95% of the picks are within these
        plt.vlines(locationDF.quantile(0.95).signed_error, 0,80, linestyle='dashed', label='95% Quantile')
        plt.xlabel('Signed Error')
        plt.ylabel('Number of counts')
        plt.legend()
        plt.show()
        plt.clf()
'''

# %%
for i in range(len(big_de_error)):
    
    plt.plot(np.arange(0.01,1,0.1), big_de_error[i], c='k', marker='o', markersize=1, alpha=0.1)
    plt.ylim(1,10000)
    plt.semilogy()
    plt.ylabel('Root Mean Squared Error (ft)')
    plt.xlabel('Percent of dataset used to train')
    plt.title('Root Mean Squared Error for 100 random train-test splits')
    
    plt.savefig(r'E:\UT Austin\Manuscripts\RecSys\Figures\working\testvstrainRMSE.pdf')

# %% [markdown]
# ### Now let's hold out one well and slowly add in tops for it

# %%
ssmin = tops.SS.min()
tops.SS = tops.SS - ssmin #standardize the subsea values
from sklearn.metrics import mean_absolute_error as MAE

tops.dropna(inplace=True)

def well_sample_splitter(dataframe, master_dataframe, number_of_tops, randomseed):
    test = dataframe.sample(n=number_of_tops, random_state=randomseed)
    test_idx = test.index.values
    train =  master_dataframe.drop(test_idx)
    return train, test

well_errors = []
for well in range(100):
    moving_well = tops[tops.API == tops.API.unique()[well]]
    errors = []
    for i in range(1, moving_well.shape[0], 1):

        train, test = well_sample_splitter(moving_well, tops, i, 86)

        #print(f'Training size is {len(train)} tops, and test size is {len(test)} tops')

        D_df = train.pivot_table("SS", "Formation", "API").fillna(0)#pivot table to move into sparse matrix land
        R = D_df.values
        from sklearn.preprocessing import binarize
        A = binarize(R) 
        U, Vt = runALS(R, A, 3, 20, 0.1)

        recommendations = np.dot(U, Vt) #get the recommendations

        recsys = pd.DataFrame(
            data=recommendations[0:, 0:], index=D_df.index, columns=D_df.columns
        ) #results

        newDF = recsys.T
        newDF.reset_index(inplace=True)

        flat_preds = pd.DataFrame(recsys.unstack()).reset_index()

        new_df = pd.merge(test, flat_preds,  how='left', left_on=['API','Formation'], right_on = ['API','Formation'])

        new_df.rename(columns={0:'SS_pred'}, inplace=True)

        cleanDF = new_df.dropna()

        cleanDF['signed_error'] = (cleanDF['SS'] - cleanDF['SS_pred'])

        errors.append(MAE(cleanDF.SS.values-ssmin, cleanDF.SS_pred.values-ssmin))
    print(f'Done with {well+1} wells')
    well_errors.append(errors)

# %% [markdown]
# # Error by well

# %%
cv_wells = []
def cross_validation_wells(dataframe, random_seed, latent_vectors, n_iters, reg):
    block_1  = dataframe.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    tops2 = dataframe.drop(block_1)
    block_2 = tops2.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    tops3 = tops2.drop(block_2)
    block_3 = tops3.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    tops4 = tops3.drop(block_3)
    block_4 = tops4.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    blocks = [block_1, block_2, block_3, block_4]
    CV_MAE = []
    f = 0
    for block in blocks:
        validate = dataframe.loc[block]
        main_group = dataframe.drop(block)
        print(f'Validating on {block.shape[0]} tops')
        D_df = main_group.pivot_table("SS", "Formation", "API").fillna(0)#pivot table to move into sparse matrix land
        R = D_df.values
        A = binarize(R) 

        
        U, Vt = runALS(R, A, latent_vectors, n_iters, reg)

        recommendations = np.dot(U, Vt) #get the recommendations

        recsys = pd.DataFrame(
            data=recommendations[0:, 0:], index=D_df.index, columns=D_df.columns
        ) #results

        newDF = recsys.T
        newDF.reset_index(inplace=True)

        flat_preds = pd.DataFrame(recsys.unstack()).reset_index()

        new_df = pd.merge(validate, flat_preds,  how='left', left_on=['API','Formation'], right_on = ['API','Formation'])

        new_df.rename(columns={0:'SS_pred'}, inplace=True)

        cleanDF = new_df.dropna()

        cleanDF['signed_error'] = (cleanDF['SS'] - cleanDF['SS_pred'])
        well_locs = pd.read_csv(r'Munging/TeapotDomeWellHeaders_outliers_JRP.csv')
        well_locs.rename(columns={'API Number':'API'}, inplace=True)

        locationDF = cleanDF.merge(well_locs[['Northing', 'Easting', 'API']], on='API')
        aypi = []
        well_mae = []
        well_rmse = []
        east = []
        north = []
        fold = []
        
        print(f'foldno is {f}')
        for well in locationDF.API.unique():
            aypi.append(well)
            well_mae.append(locationDF[locationDF.API == well].signed_error.abs().mean())
            well_rmse.append(np.sqrt(MSE(locationDF[locationDF.API == well].SS, locationDF[locationDF.API == well].SS_pred)))
            east.append(locationDF[locationDF.API == well].Easting.values[0])
            north.append(locationDF[locationDF.API == well].Northing.values[0])
            fold.append(f)
        by_wellDF = pd.DataFrame({'API':aypi, 'Well_MAE':well_mae, 'well_rmse':well_rmse, 'Easting':east, 'Northing':north,
                                 'foldno':fold})
        cv_wells.append(by_wellDF)
        f+=1
        #CV_MAE.append(MAE(cleanDF.SS.values-ssmin, cleanDF.SS_pred.values-ssmin))

    return cv_wells

# %%
masterDF = pd.concat(cross_validation_wells(tops, 86, 2, 290, 0.1))
masterDF.to_csv('error_map.csv')

# %%
plt.scatter(masterDF[masterDF.foldno == 3].Easting, masterDF[masterDF.foldno == 3].Northing, 
            c=masterDF[masterDF.foldno == 3].Well_MAE)
plt.colorbar()

# %%
masterDF[masterDF.foldno ==2].sort_values('Well_MAE')

# %%
plt.scatter(masterDF[masterDF.foldno == 1].Easting, masterDF[masterDF.foldno == 1].Northing, 
            c=masterDF[masterDF.foldno == 1].Well_MAE)

# %%
well_locs = pd.read_csv(r'TeapotDomeWellHeaders.csv')
well_locs.rename(columns={'API Number':'API'}, inplace=True)

what = tops.merge(well_locs[['Northing', 'Easting', 'API']], on='API')
plt.scatter(what.Easting, what.Northing)

# %% [markdown]
# ### How about Pyvista?

# %%
import pyvista as pv
import panel
from itkwidgets import view
panel.extension("vtk")
pv.set_plot_theme("document")

# %%
gl = tops
surface = gl.merge(well_locs[['Northing', 'Easting', 'API']], on='API')
cloud = pv.PolyData(np.array(list(zip(surface.Easting, surface.Northing, surface.GL))))
surf = cloud.delaunay_2d()

# %%
plotter = pv.Plotter(notebook=False)
for formation in strat_order:
    subpreds = predictions[predictions['Formation'] == formation]
    points = np.array(list(zip(subpreds.Easting, subpreds.Northing, subpreds['SS_pred'])))
    if points.shape[0] > 0:
        pc = pv.PolyData(points)
    #data = tops_predicted[i][:,-1]
    #pc['elevation'] = data
    plotter.add_points(pc, color=colors_to_plot[formation])#color=[r,g,b])
plotter.add_mesh(surf)
plotter.set_scale(1,1,2, 0) #V.E.
plotter.show()

# %%
