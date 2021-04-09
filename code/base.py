import os
import math
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict
from sklearn import preprocessing

GD = "gd-"
NE = "normal_equation-"
MSE = "MSE"
RMSE = "RMSE"


def create_outputs_folder(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)


def normalize(df):
    new_df = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    new_df_scaled = min_max_scaler.fit_transform(new_df)
    return pd.DataFrame(new_df_scaled)


def read_shuffle_normalize_data():
    data = pd.read_csv("./data1_Signal.csv", names=["x", "y"], header=1)
    data = data.sample(frac=1).reset_index(drop=True)
    print("head of shuffled data:", data.head())
    X_df = pd.DataFrame(data.x)
    X_df = normalize(X_df)
    print("head of normalized shuffled data:", X_df.head())
    y_df = pd.DataFrame(data.y)
    return X_df, y_df


def add_bias(X_df):
    X = np.array(X_df)
    m, _ = X_df.shape
    bias = np.ones(m)
    return np.insert(X, 0, values=bias, axis=1)


def flatten(y_df):
    return np.array(y_df).flatten()


def get_test_and_train_datasets(X_df, y_df, r=0.7, include_bias=True):
    m, _ = X_df.shape
    if include_bias:
        X = add_bias(X_df)
    else:
        X = X_df.values  # TODO test
    y = flatten(y_df)
    p = int(r * m)
    X_train = X[:p, :]
    X_test = X[p:, :]
    y_train = y[:p]
    y_test = y[p:]
    return X_train, X_test, y_train, y_test


def copy_column_src_as_polynomial_to_dest_with_degree_d(X, src, dest, d):
    return np.insert(X, dest, X[:, src] ** d, axis=1)


def extend_X(X, d):
    src = 1
    dest = X.shape[1]
    return copy_column_src_as_polynomial_to_dest_with_degree_d(X, src, dest, d)


def train(
    algorithm,
    function,
    X_train,
    y_train,
    X_test,
    y_test,
    alpha,
    lambda_,
    iterations,
    cf,
    degree,
):
    theta, train_cost_history, test_cost_history, step_sizes = function(
        X_train, y_train, X_test, y_test, alpha, lambda_, iterations, cf
    )

    return {
        "algorithm": algorithm,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "theta": theta,
        "train_cost_history": train_cost_history,
        "test_cost_history": test_cost_history,
        "step_sizes": step_sizes,
        "alpha": alpha,
        "lambda_": lambda_,
        "iterations": iterations,
        "cf": cf,
        "degree": degree,
    }


def predict(X, theta):
    return X.dot(theta)


def cost_function(X, y, theta, lambda_, cf):
    m, _ = X.shape
    if lambda_ is None:
        J = np.sum((X.dot(theta) - y) ** 2) / (2 * m)
    else:
        l = lambda_ * np.sum(theta ** 2)
        l = l - lambda_ * (theta[0] ** 2)
        J = (np.sum((X.dot(theta) - y) ** 2) + l) / (2 * m)
    if cf == MSE:
        return J
    elif cf == RMSE:
        return math.sqrt(J)
    raise Exception("unknown cost function: " + cf)


def gradient(X_train, y_train, theta, loss, lambda_, cf):
    m, n = X_train.shape
    if cf == MSE:
        if lambda_ is None:
            return X_train.T.dot(loss) / m
        else:
            t = lambda_ * theta
            t[0] = 0
            return (X_train.T.dot(loss) + t) / m
    elif cf == RMSE:
        cost = cost_function(X_train, y_train, theta, lambda_, cf)
        if lambda_ is None:
            return (X_train.T.dot(loss) / m) * (1 / (2 * cost))
        else:
            t = lambda_ * theta
            t[0] = 0
            return ((X_train.T.dot(loss) + t) / m) * (1 / (2 * cost))
    raise Exception("unknown cost function: " + cf)


def gradient_descent(X_train, y_train, X_test, y_test, alpha, lambda_, iterations, cf):
    m, n = X_train.shape
    theta = np.random.uniform(low=-1, high=1, size=(n,))
    # theta = np.zeros(n)
    train_cost_history = np.zeros(iterations)
    test_cost_history = np.zeros(iterations)
    step_sizes = np.zeros((iterations, n))
    for iteration in range(iterations):
        hypothesis = X_train.dot(theta)
        loss = hypothesis - y_train
        grad = gradient(X_train, y_train, theta, loss, lambda_, cf)
        step_size = alpha * grad
        theta = theta - step_size
        train_cost_history[iteration] = cost_function(
            X_train, y_train, theta, None, cf
        )  # note: always pass lambda as None
        test_cost_history[iteration] = cost_function(
            X_test, y_test, theta, None, cf
        )  # note: always pass lambda as None

        step_sizes[iteration] = step_size
    return theta, train_cost_history, test_cost_history, step_sizes


def normal_equation(X_train, y_train, X_test, y_test, alpha, lambda_, iterations, cf):
    assert iterations == 1
    _, n = X_train.shape
    if lambda_ is None:
        theta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T.dot(y_train))
    else:
        ones = np.ones((n, n))
        ones[0, 0] = 0
        theta = np.linalg.inv(X_train.T.dot(X_train) + lambda_ * ones).dot(
            X_train.T.dot(y_train)
        )
    return (
        theta,
        [
            cost_function(X_train, y_train, theta, None, cf)
        ],  # note: always pass lambda as None
        [
            cost_function(X_test, y_test, theta, None, cf)
        ],  # note: always pass lambda as None
        np.zeros((1, n)),
    )


def next_color():
    colors = {
        0: "red",
        1: "blue",
        2: "green",
        3: "yellow",
        4: "pink",
        5: "brown",
        6: "purple",
        7: "gray",
        8: "black",
        9: "orange",
        10: "DarkRed",
        11: "DeepPink",
        12: "DarkKhaki",
        13: "RebeccaPurple",
        14: "DeepSkyBlue",
    }
    for i in range(1000):
        yield colors[i % 15]


def plot_2d_data(X_df, y_df, path, title=""):
    plt.figure(figsize=(20, 15))
    plt.plot(X_df.values, y_df.values, ".")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.savefig(path + "plot-data.png")
    plt.show()


def plot_model(
    model,
    path,
    plot_best_fit=False,
    plot_cost_per_iteration=True,
    plot_step_size_per_iteration=True,
    test_or_validation="test",
):
    algorithm = model["algorithm"]
    X_train = model["X_train"]
    y_train = model["y_train"]
    X_test = model["X_test"]
    y_test = model["y_test"]
    theta = model["theta"]
    train_cost_history = model["train_cost_history"]
    test_cost_history = model["test_cost_history"]
    step_sizes = model["step_sizes"]
    cf = model["cf"]
    iterations = model["iterations"]
    alpha = model["alpha"]
    lambda_ = model["lambda_"]
    degree = model["degree"]
    print(
        "model:" "algorithm:",
        algorithm,
        "train_cost:",
        train_cost_history[-1],
        test_or_validation + "_cost:",
        test_cost_history[-1],
        "cf:",
        cf,
        "iterations:",
        iterations,
        "alpha:",
        alpha,
        "lambda:",
        lambda_,
        "degree:",
        degree,
    )
    print("theta:", theta)

    if plot_best_fit:
        y_test_predicted = predict(X_test, theta)
        y_best_fit = predict(X_train, theta)
        plt.figure(figsize=(20, 15))
        plt.plot(X_train[:, 1], y_train, ".", color="r", label="X_train, y_train")
        plt.plot(X_train[:, 1], y_best_fit, ".", color="y", label="best fit")
        plt.plot(
            X_test[:, 1],
            y_test,
            ".",
            color="b",
            label="X_" + test_or_validation + ", y_" + test_or_validation,
        )
        plt.plot(
            X_test[:, 1],
            y_test_predicted,
            ".",
            color="g",
            label="X_"
            + test_or_validation
            + ", y_"
            + test_or_validation
            + "_predicted",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        title = "best_fit-algorithm_{}-degree_{}-iterations_{}-alpha_{}-lambda_{}-cf_{}".format(
            algorithm, degree, iterations, alpha, lambda_, cf
        )
        plt.title(title)
        plt.legend()
        plt.savefig(path + title + ".png")
        plt.show()

    # ----

    if plot_cost_per_iteration:
        plt.figure(figsize=(20, 15))
        plt.plot(
            range(1, iterations + 1),
            train_cost_history,
            ".",
            color="r",
            label="train cost hisotry",
        )
        plt.plot(
            range(1, iterations + 1),
            test_cost_history,
            ".",
            color="b",
            label=test_or_validation + " cost hisotry",
        )
        plt.xlabel("iteration")
        plt.ylabel("cost")
        title = "cost_per_iteration-algorithm_{}-degree_{}-iterations_{}-alpha_{}-lambda_{}-cf_{}".format(
            algorithm, degree, iterations, alpha, lambda_, cf
        )
        plt.title(title)
        plt.legend()
        plt.savefig(path + title + ".png")
        plt.show()

    # ----

    if plot_step_size_per_iteration:
        plt.figure(figsize=(20, 15))
        _, n = step_sizes.shape
        get_next_color = next_color()
        for i in range(n):
            plt.plot(
                range(1, iterations + 1),
                step_sizes[:, i],
                ".",
                color=next(get_next_color),
                label="theta_" + str(i),
            )
        plt.xlabel("iteration")
        plt.ylabel("step size")
        title = "step_size_per_iteration-algorithm_{}-degree_{}-iterations_{}-alpha_{}-lambda_{}-cf_{}".format(
            algorithm, degree, iterations, alpha, lambda_, cf
        )
        plt.title(title)
        plt.legend()
        plt.savefig(path + title + ".png")
        plt.show()


# TODO algorithm
def _plot_models(
    all_models,
    index_type,
    path,
    plot_best_fit=False,
    plot_cost_per_iteration=True,
    plot_step_size_per_iteration=True,
    test_or_validation="test",
):
    if plot_best_fit:
        for (
            index,
            models,
        ) in all_models.items():  # index is either iteration, degree, or lambda
            plt.figure(figsize=(20, 15))
            get_next_color = next_color()
            plt.plot(
                models[0]["X_train"][:, 1],
                models[0]["y_train"],
                ".",
                color=next(get_next_color),
                label="train data",
            )
            plt.plot(
                models[0]["X_test"][:, 1],
                models[0]["y_test"],
                ".",
                color=next(get_next_color),
                label=test_or_validation + " data",
            )
            for model in models:
                plt.plot(
                    model["X_train"][:, 1],
                    predict(model["X_train"], model["theta"]),
                    ".",
                    color=next(get_next_color),
                    label="best_fit-degree_{}-alpha_{}-lambda_{}-cf_{}-iterations_{}".format(
                        model["degree"],
                        model["alpha"],
                        model["lambda_"],
                        model["cf"],
                        model["iterations"],
                    ),
                )
                plt.plot(
                    model["X_test"][:, 1],
                    predict(model["X_test"], model["theta"]),
                    ".",
                    color=next(get_next_color),
                    label=test_or_validation
                    + "_data_prediction-degree_{}-alpha_{}-lambda_{}-cf_{}-iterations_{}".format(
                        model["degree"],
                        model["alpha"],
                        model["lambda_"],
                        model["cf"],
                        model["iterations"],
                    ),
                )
            plt.xlabel("x")
            plt.ylabel("y")
            title = "{}_{}_best_fits".format(index_type, index)
            plt.title(title)
            plt.legend()
            plt.savefig(path + title + ".png")
            plt.show()

            # ----
    if plot_cost_per_iteration:
        for index, models in all_models.items():
            plt.figure(figsize=(20, 15))
            get_next_color = next_color()
            if index_type == "iterations" and index == 1:
                for model in models:
                    plt.plot(
                        [model["degree"]],
                        model["train_cost_history"],
                        ".",
                        color=next(get_next_color),
                        label="train_cost-degree_{}-alpha_{}-lambda_{}-cf_{}-iterations_{}".format(
                            model["degree"],
                            model["alpha"],
                            model["lambda_"],
                            model["cf"],
                            model["iterations"],
                        ),
                    )
                    plt.plot(
                        [model["degree"]],
                        model["test_cost_history"],
                        ".",
                        color=next(get_next_color),
                        label=test_or_validation
                        + "_cost-degree_{}-alpha_{}-lambda_{}-cf_{}-iterations_{}".format(
                            model["degree"],
                            model["alpha"],
                            model["lambda_"],
                            model["cf"],
                            model["iterations"],
                        ),
                    )
                plt.xlabel("degree")
                plt.ylabel("cost")
                title = "{}_{}_cost_per_degree".format(index_type, index)
                plt.title(title)
                plt.legend()
                plt.savefig(path + title + ".png")
                plt.show()

                # ----

                for model in models:
                    plt.plot(
                        [model["lambda_"]],
                        model["train_cost_history"],
                        ".",
                        color=next(get_next_color),
                        label="train_cost-degree_{}-alpha_{}-lambda_{}-cf_{}-iterations_{}".format(
                            model["degree"],
                            model["alpha"],
                            model["lambda_"],
                            model["cf"],
                            model["iterations"],
                        ),
                    )
                    plt.plot(
                        [model["lambda_"]],
                        model["test_cost_history"],
                        ".",
                        color=next(get_next_color),
                        label=test_or_validation
                        + "_cost-degree_{}-alpha_{}-lambda_{}-cf_{}-iterations_{}".format(
                            model["degree"],
                            model["alpha"],
                            model["lambda_"],
                            model["cf"],
                            model["iterations"],
                        ),
                    )
                plt.xlabel("lambda")
                plt.ylabel("cost")
                title = "{}_{}_cost_per_lambda".format(index_type, index)
                plt.title(title)
                plt.legend()
                plt.savefig(path + title + ".png")
                plt.show()
            else:
                for model in models:
                    plt.plot(
                        range(1, model["iterations"] + 1),
                        model["train_cost_history"],
                        ".",
                        color=next(get_next_color),
                        label="train_cost_history-degree_{}-alpha_{}-lambda_{}-cf_{}-iterations_{}".format(
                            model["degree"],
                            model["alpha"],
                            model["lambda_"],
                            model["cf"],
                            model["iterations"],
                        ),
                    )
                    plt.plot(
                        range(1, model["iterations"] + 1),
                        model["test_cost_history"],
                        ".",
                        color=next(get_next_color),
                        label=test_or_validation
                        + "_cost_history-degree_{}-alpha_{}-lambda_{}-cf_{}-iterations_{}".format(
                            model["degree"],
                            model["alpha"],
                            model["lambda_"],
                            model["cf"],
                            model["iterations"],
                        ),
                    )
                plt.xlabel("iteration")
                plt.ylabel("cost")
                title = "{}_{}_cost_per_iteration".format(index_type, index)
                plt.title(title)
                plt.legend()
                plt.savefig(path + title + ".png")
                plt.show()

            # ----
    if plot_step_size_per_iteration:
        for index, models in all_models.items():
            plt.figure(figsize=(20, 15))
            get_next_color = next_color()
            for model in models:
                plt.plot(
                    range(1, model["iterations"] + 1),
                    model["step_sizes"][:, 0],
                    ".",
                    color=next(get_next_color),
                    label="step_sizes_theta0-degree_{}-alpha_{}-lambda_{}-cf_{}-iterations_{}".format(
                        model["degree"],
                        model["alpha"],
                        model["lambda_"],
                        model["cf"],
                        model["iterations"],
                    ),
                )
                plt.plot(
                    range(1, model["iterations"] + 1),
                    model["step_sizes"][:, 1],
                    ".",
                    color=next(get_next_color),
                    label="step_sizes_theta1-degree_{}-alpha_{}-lambda_{}-cf_{}-iterations_{}".format(
                        model["degree"],
                        model["alpha"],
                        model["lambda_"],
                        model["cf"],
                        model["iterations"],
                    ),
                )
            plt.xlabel("iteration")
            plt.ylabel("step size")
            title = "{}_{}_step_size_per_iteration".format(index_type, index)
            plt.title(title)
            plt.legend()
            plt.savefig(path + title + ".png")
            plt.show()


def plot_models(
    all_models,
    path,
    plot_per_iteration=True,
    plot_per_degree=True,
    plot_per_lambda=True,
    plot_best_fit=False,
    plot_cost_per_iteration=True,
    plot_step_size_per_iteration=True,
    test_or_validation="test",
):
    if plot_per_iteration:
        models_per_iteration = defaultdict(list)
        for model in all_models:
            models_per_iteration[model["iterations"]].append(model)
        _plot_models(
            models_per_iteration,
            "iterations",
            path,
            plot_best_fit=plot_best_fit,
            plot_cost_per_iteration=plot_cost_per_iteration,
            plot_step_size_per_iteration=plot_step_size_per_iteration,
            test_or_validation=test_or_validation,
        )

    if plot_per_degree:
        models_per_degree = defaultdict(list)
        for model in all_models:
            models_per_degree[model["degree"]].append(model)
        _plot_models(
            models_per_degree,
            "degree",
            path,
            plot_best_fit=plot_best_fit,
            plot_cost_per_iteration=plot_cost_per_iteration,
            plot_step_size_per_iteration=plot_step_size_per_iteration,
            test_or_validation=test_or_validation,
        )

    if plot_per_lambda:
        models_per_lambda = defaultdict(list)
        for model in all_models:
            models_per_lambda[model["lambda_"]].append(model)
        _plot_models(
            models_per_lambda,
            "lambda",
            path,
            plot_best_fit=plot_best_fit,
            plot_cost_per_iteration=plot_cost_per_iteration,
            plot_step_size_per_iteration=plot_step_size_per_iteration,
            test_or_validation=test_or_validation,
        )


def run(
    X_train,
    X_test,
    y_train,
    y_test,
    algorithm,
    start_degree,
    alpha,
    degrees,
    cost_functions,
    lambdas,
    iterations,
):
    m_X_train, _ = X_train.shape
    m_X_test, _ = X_test.shape
    print("train data = {} rows, test data = {} rows".format(m_X_train, m_X_test))
    models = []
    degrees.sort()

    for deg in degrees:
        for d in range(start_degree, deg + 1):
            X_train = extend_X(X_train, d)
            X_test = extend_X(X_test, d)
        start_degree = deg + 1
        for cf in cost_functions:
            for lambda_ in lambdas:
                if algorithm == GD:
                    for i in iterations:
                        models.append(
                            train(
                                algorithm,
                                gradient_descent,
                                X_train,
                                y_train,
                                X_test,
                                y_test,
                                alpha,
                                lambda_,
                                i,
                                cf,
                                deg,
                            )
                        )
                elif algorithm == NE:
                    models.append(
                        train(
                            algorithm,
                            normal_equation,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            alpha,
                            lambda_,
                            1,
                            cf,
                            deg,
                        )
                    )
                else:
                    raise Exception("unknown algorithm: " + ALGORITHM)

    return models
