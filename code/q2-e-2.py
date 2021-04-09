import seaborn as sns
from base import *


def read_shuffle_data():
    names = [
        "id",
        "date",
        "price",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "zipcode",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
    ]
    data = pd.read_csv("./data2_house_data.csv", names=names, header=1)
    # data.assign(date=pd.to_datetime(data.date))
    data["date"] = pd.to_datetime(data["date"]).astype(int) / 10 ** 9
    data = data.sample(frac=1).reset_index(drop=True)
    return data


def extract_X_and_y(data):
    y = data.iloc[:, 2]  # price column
    X = data.drop(columns=data.columns[[2]])
    return X, y


def remove_features(data):
    return data.drop(["id", "date", "zipcode", "sqft_above"], axis=1)


def get_test_and_validation_and_train_datasets(
    X_df, y_df, r=0.5, p=0.3, include_bias=True
):
    assert r + p < 1
    m, _ = X_df.shape
    if include_bias:
        X = add_bias(X_df)
    else:
        X = X_df.values  # TODO test
    y = flatten(y_df)
    s = int(r * m)
    u = int(p * m)
    X_train = X[:s, :]
    y_train = y[:s]
    X_validation = X[s : s + u, :]
    y_validation = y[s : s + u]
    X_test = X[s + u :, :]
    y_test = y[s + u :]
    return X_train, X_validation, X_test, y_train, y_validation, y_test


if __name__ == "__main__":
    algorithm = NE
    alpha = None
    cost_functions = [MSE]
    lambdas = [0.5, 5]
    iterations = [1]
    start_degree = 3
    degrees = [
        2
    ]  # 1, 2 # which means no degree is applied since it is applied manually blow
    path = "./q2-section_e-outputs_removed_features/"
    create_outputs_folder(path)
    path = path + "alpha_" + str(alpha) + "-degree_" + str(start_degree - 1) + "/"
    create_outputs_folder(path)
    data = read_shuffle_data()
    print("head of shuffled data:")
    print(data.head())
    data = remove_features(data)
    data = normalize(data)
    print("head of normalized data:")
    print(data.head())
    X_df, y_df = extract_X_and_y(data)
    print("head of X_df:")
    print(X_df.head())
    print("head of y_df:")
    print(y_df.head())
    X_df = pd.concat([X_df, X_df ** 2], axis=1)  # other forms
    print("head of data with quadratic polynomial features:")
    print(X_df.head())
    # TODO why too much zero values?

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = get_test_and_validation_and_train_datasets(
        X_df, y_df, r=0.5, p=0.3, include_bias=True
    )

    models = run(
        X_train,
        X_validation,
        y_train,
        y_validation,
        algorithm,
        start_degree,
        alpha,
        degrees,
        cost_functions,
        lambdas,
        iterations,
    )

    for model in models:
        plot_model(
            model,
            path,
            plot_best_fit=False,
            plot_cost_per_iteration=True,
            plot_step_size_per_iteration=False,
            test_or_validation="validation",
        )

    plot_models(
        models,
        path,
        plot_per_iteration=True,
        plot_per_degree=True,
        plot_per_lambda=True,
        plot_best_fit=False,
        plot_cost_per_iteration=True,
        plot_step_size_per_iteration=False,
        test_or_validation="validation",
    )

    best_model = models[0]
    for model in models:
        if model["test_cost_history"][-1] < best_model["test_cost_history"][-1]:
            best_model = model

    print("best model:")
    print(
        "model:" "algorithm:",
        best_model["algorithm"],
        "train_cost:",
        best_model["train_cost_history"][-1],
        "validation_cost:",
        best_model["test_cost_history"][-1],
        "cf:",
        best_model["cf"],
        "iterations:",
        best_model["iterations"],
        "alpha:",
        best_model["alpha"],
        "lambda:",
        best_model["lambda_"],
        "degree:",
        best_model["degree"],
    )

    print("test cost on best model:")
    print(cost_function(X_test, y_test, best_model["theta"], None, model["cf"]))
