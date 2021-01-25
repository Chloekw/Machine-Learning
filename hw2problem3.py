import numpy as np
import pandas as pd
from sklearn import linear_model

def binary_classifier(A, w, b):
    b_predict = np.dot(A, w)
    b_predict = b_predict.T[0]
    b = b.T[0]
    n = A.shape[0]

    false = 0
    false_positive = 0
    true_negative = 0
    for p in range(n):
        if b_predict[p] > 1/2:
            if b[p] == 0:
                false += 1
                false_positive += 1
        else:
            if b[p] == 1:
                false += 1
            else:
                true_negative += 1
    return false/n, false_positive/(false_positive + true_negative)


if __name__ == '__main__':
    print()
    df_train = pd.read_csv('compas-train.csv')
    df_test = pd.read_csv('compas-test.csv')

    n = df_train.shape[0]
    features = ["sex","age","race","juv_fel_count","juv_misd_count","juv_other_count","priors_count","c_charge_degree"]
    A_comb = np.concatenate(
        (np.ones((df_train.shape[0], 1)), df_train[features]), axis=1)
    b = df_train[["two_year_recid"]]

    # compute affine function of training data (for checking), got the same weight vector as above
    # x = pd.DataFrame(df_train[features]).to_numpy()
    # y = pd.DataFrame(b).to_numpy()
    # reg = linear_model.LinearRegression(fit_intercept=True)
    # reg.fit(x, y)
    # print("intercept (training): ", reg.intercept_)
    # print("weight vector (training): ", reg.coef_)


    pinverse_A_comb = np.linalg.pinv(A_comb)
    coeff_comb = np.dot(pinverse_A_comb, b)
    print("Weight vector (training): ", coeff_comb.T)
    print("MSE (training): ",(np.linalg.norm(np.dot(A_comb, coeff_comb) - b)) ** 2 / n)

    A_comb_test = np.concatenate(
        (np.ones((df_test.shape[0], 1)), df_test[features]), axis=1)
    b_test = df_test[["two_year_recid"]]
    print("MSE (test): ", (np.linalg.norm(np.dot(A_comb_test, coeff_comb) - b_test)) ** 2 / df_test.shape[0])

    # compute the error rate and false positive rate of f_hat binary classifier
    err_rate, false_pos_rate = binary_classifier(A_comb_test, coeff_comb, pd.DataFrame(b_test).to_numpy())
    print("\nError rate (test): ", err_rate)
    print("False positive rate (test): ", false_pos_rate)

    # choose sex as our feature A to separate test dataset
    A_test_0 = df_test.loc[df_test['sex'] == 0]
    test_0 = np.concatenate(
        (np.ones((A_test_0.shape[0], 1)), A_test_0[features]), axis=1)
    test_b_0 = A_test_0[["two_year_recid"]]

    A_test_1 = df_test.loc[df_test['sex'] == 1]
    test_1 = np.concatenate(
        (np.ones((A_test_1.shape[0], 1)), A_test_1[features]), axis=1)
    test_b_1 = A_test_1[["two_year_recid"]]

    err_rate0, false_pos_rate0 = binary_classifier(test_0, coeff_comb, pd.DataFrame(test_b_0).to_numpy())
    print("\nError rate (test sex = 0): ", err_rate0)
    print("False positive rate (test sex = 0): ", false_pos_rate0)

    err_rate1, false_pos_rate1 = binary_classifier(test_1, coeff_comb, pd.DataFrame(test_b_1).to_numpy())
    print("\nError rate (test sex = 1): ", err_rate1)
    print("False positive rate (test sex = 1): ", false_pos_rate1)


    # separate training data in two groups, ie sex = 0 and sex = 1
    print("\nSeparate training data into two groups: ")
    train_0 = df_train.loc[df_train['sex'] == 0]
    A_train_0 = np.concatenate(
        (np.ones((train_0.shape[0], 1)), train_0[features]), axis=1)
    b_train_0 = train_0[["two_year_recid"]]
    coeff_0 = np.dot(np.linalg.pinv(A_train_0), b_train_0)
    print("Weight vector (training sex = 0): ", coeff_0.T)
    err_rate0, false_pos_rate0 = binary_classifier(test_0, coeff_0, pd.DataFrame(test_b_0).to_numpy())
    print("Error rate (test sex = 0): ", err_rate0)
    print("False positive rate (test sex = 0): ", false_pos_rate0)


    train_1 = df_train.loc[df_train['sex'] == 1]
    A_train_1 = np.concatenate(
        (np.ones((train_1.shape[0], 1)), train_1[features]), axis=1)
    b_train_1 = train_1[["two_year_recid"]]
    coeff_1 = np.dot(np.linalg.pinv(A_train_1), b_train_1)
    print("\nWeight vector (training sex = 1): ", coeff_1.T)
    err_rate1, false_pos_rate1 = binary_classifier(test_1, coeff_1, pd.DataFrame(test_b_1).to_numpy())
    print("Error rate (test sex = 1): ", err_rate1)
    print("False positive rate (test sex = 1): ", false_pos_rate1)
