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
    print("head of shuffled data:")
    print(data.head())
    return data


def plot_correlation_hitmap(data, path):
    corr = data.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.savefig(path + "hitmap.png")
    plt.title("dataset features correlations")
    plt.show()


if __name__ == "__main__":
    path = "./q2-section_a_b-outputs/"
    create_outputs_folder(path)
    data = read_shuffle_data()
    plot_2d_data(data.lat, data.long, path, title="dataset")
    sns.set_theme(style="white")
    plot_correlation_hitmap(data, path)
