import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
from random import randint, uniform, seed
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

def sample_splitter(dataframe, fraction, randomseed):
    test = dataframe.sample(frac=fraction, random_state=randomseed)
    test_idx = test.index.values
    train =  dataframe.drop(test_idx)
    return train, test

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
    #print("Initiating ")
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

def cross_validation(dataframe, random_seed, latent_vectors, n_iters, reg):
    block_1  = dataframe.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    tops2 = dataframe.drop(block_1)
    block_2 = tops2.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    tops3 = tops2.drop(block_2)
    block_3 = tops3.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    tops4 = tops3.drop(block_3)
    block_4 = tops4.sample(n=dataframe.shape[0]//4, random_state=random_seed).index.values
    blocks = [block_1, block_2, block_3, block_4]
    CV_MAE = []
    for block in blocks:
        validate = dataframe.loc[block]
        main_group = dataframe.drop(block)
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

        CV_MAE.append(MAE(cleanDF.SS.values-ssmin, cleanDF.SS_pred.values-ssmin))

    return CV_MAE

if __name__=='__main__':
    tops = pd.read_csv(r"teapot_clean_JRP.csv", index_col=[0]) #read in the top data
    tops.rename(columns={'TVDSS':'SS'}, inplace=True)

    ssmin = tops.SS.min()
    tops.SS = tops.SS - ssmin #standardize the subsea values

    tops.dropna(inplace=True)


    train, test = sample_splitter(tops, 0.01, 86)

    #print(f'Training size is {len(train)} tops, and test size is {len(test)} tops')

    D_df = train.pivot_table("SS", "Formation", "API").fillna(0)#pivot table to move into sparse matrix land
    R = D_df.values
    A = binarize(R)

    #print(f'{round(((D_df == 0).astype(int).sum().sum())/((D_df == 0).astype(int).sum().sum()+(D_df != 0).astype(int).sum().sum()),3)*100} percent of the tops are missing')
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

    L, its, regs = min(grid_search, key=grid_search.get)
    errorDF = pd.DataFrame({'L':els, 'iterations':nsits,  'MAE':list(grid_search.values())})
    errorDF.to_csv('gridsearch2.csv')

    import seaborn as sns
    ax0 = sns.boxplot(x="L", y="MAE", data=errorDF)
    plt.xlabel('Number of Latent Factors')
    plt.ylabel('MAE (ft)')
    plt.savefig('latent_factors.svg')
    ax1 = sns.boxplot(x="iterations", y="MAE", data=errorDF)
    plt.xlabel('Number of iterations')
    plt.ylabel('MAE (ft)')
    plt.savefig('iterations.svg')
    ax1 = sns.boxplot(x="regularization", y="MAE", data=errorDF)
    plt.xlabel('Regularization Parameter')
    plt.ylabel('MAE (ft)')
    plt.savefig('regularization.svg')
