import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gp
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import mean_absolute_error as MSE
from sklearn.preprocessing import binarize

#%matplotlib inline
tops = pd.read_csv(r"jonah_tops.csv")  # read in the top data

ssmin = tops.SS.min()
tops.SS = tops.SS - ssmin  # standardize the subsea values

tops.dropna(inplace=True)


def sample_splitter(dataframe, fraction, randomseed):
    test = dataframe.sample(frac=fraction, random_state=randomseed)
    test_idx = test.index.values
    train = dataframe.drop(test_idx)
    return train, test


train, test = sample_splitter(tops, 0.1, 86)

D_df = train.pivot_table("SS", "Formation", "API").fillna(
    0
)  # pivot table to move into sparse matrix land

R = D_df.values

A = binarize(R)

# ALS factorization from
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
    print("Initiating ")
    lambda_ = lambda_
    n_factors = n_factors
    n, m = A.shape
    n_iterations = n_iterations
    Users = 5 * np.random.rand(n, n_factors)
    Items = 5 * np.random.rand(n_factors, m)

    def get_error(A, Users, Items, R):
        # This calculates the MSE of nonzero elements
        return np.sum((R * (A - np.dot(Users, Items))) ** 2) / np.sum(R)

    MSE_List = []

    # print("Starting Iterations")
    for iter in range(n_iterations):
        for i, Ri in enumerate(R):
            Users[i] = np.linalg.solve(
                np.dot(Items, np.dot(np.diag(Ri), Items.T))
                + lambda_ * np.eye(n_factors),
                np.dot(Items, np.dot(np.diag(Ri), A[i].T)),
            ).T
        # print(
        #    "Error after solving for User Matrix:",
        #    get_error(A, Users, Items, R),
        # )

        for j, Rj in enumerate(R.T):
            Items[:, j] = np.linalg.solve(
                np.dot(Users.T, np.dot(np.diag(Rj), Users))
                + lambda_ * np.eye(n_factors),
                np.dot(Users.T, np.dot(np.diag(Rj), A[:, j])),
            )
        # print(
        #    "Error after solving for Item Matrix:",
        #    get_error(A, Users, Items, R),
        # )

        MSE_List.append(get_error(A, Users, Items, R))
        # print("%sth iteration is complete..." % iter)
    return Users, Items
    # print(MSE_List)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(range(1, len(MSE_List) + 1), MSE_List); plt.ylabel('Error'); plt.xlabel('Iteration')
    # plt.title('Python Implementation MSE by Iteration \n with %d formations and %d wells' % A.shape);
    # plt.savefig('Python MSE Graph.pdf', format='pdf')
    # plt.show()


MAE_iter = []
for iterations in range(1, 20):
    print(f"Starting for {iterations} iterations")
    for factors in range(20):
        print(f"Starting for {factors} factors")
        U, Vt = runALS(R, A, factors, iterations, 0.1)

        recommendations = np.dot(U, Vt)  # get the recommendations
        recsys = pd.DataFrame(
            data=recommendations[0:, 0:], index=D_df.index, columns=D_df.columns
        )  # results
        newDF = recsys.T
        newDF.reset_index(inplace=True)
        # test = newDF[["FORT UNION", "API"]]
        flat_preds = pd.DataFrame(recsys.unstack()).reset_index()
        new_df = pd.merge(
            test,
            flat_preds,
            how="left",
            left_on=["API", "Formation"],
            right_on=["API", "Formation"],
        )
        new_df.rename(columns={0: "SS_pred"}, inplace=True)
        cleanDF = new_df.dropna()
        cleanDF["signed_error"] = cleanDF["SS"] - cleanDF["SS_pred"]
        mae = MSE(cleanDF.SS.values - ssmin, cleanDF.SS_pred.values - ssmin)
        MAE.append(mae)
        print(f"Mean Absolute Error is {mae}")
    MAE_iter.append(MAE)

    plt.plot(range(20), MAE)
    plt.xlabel("No of factors")
    plt.ylabel("Mean Absolute Error")
    plt.show()
print(MAE_iter)
