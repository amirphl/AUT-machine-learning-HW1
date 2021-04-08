import pandas as pd
from base import *


def read_shuffle_normalize_data():
    data = pd.read_csv("./data1_Signal.csv", names=["x", "y"], header=1)
    data = data.sample(frac=1).reset_index(drop=True)
    print("head of shuffled data:", data.head())
    X_df = pd.DataFrame(data.x)
    X_df = normalize(X_df)
    print("head of normalized shuffled data:", X_df.head())
    y_df = pd.DataFrame(data.y)
    return X_df, y_df


if __name__ == "__main__":
    path = "./q1-outputs/"
    algorithm = NE
    start_degree = 2
    alpha = 0.1
    degrees = [5, 7]
    cost_functions = [MSE, RMSE]
    lambdas = [None, 5]
    iterations = [1000, 10000]
    create_outputs_folder(path)
    X_df, y_df = read_shuffle_normalize_data()
    plot_2d_data(X_df, y_df, path, title="dataset")
    X_train, X_test, y_train, y_test = get_test_and_train_datasets(
        X_df, y_df, r=0.7, include_bias=True
    )
    models = run(
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
    )

    # for model in models:
    #    plot_model(
    #        model,
    #        path,
    #        plot_best_fit=True,
    #        plot_cost_per_iteration=True,
    #        plot_step_size_per_iteration=True,
    #    ) TODO uncomment
    plot_models(
        models,
        path,
        plot_per_iteration=True,
        plot_per_degree=True,
        plot_per_lambda=True,
        plot_best_fit=True,
        plot_cost_per_iteration=True,
        plot_step_size_per_iteration=False,  # TODO make it True
    )
