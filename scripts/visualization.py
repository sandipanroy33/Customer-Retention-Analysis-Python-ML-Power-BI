"""
visualization.py
------------------
Reusable plotting utilities used throughout the EDA notebook:
- histogram_boxplot: combined boxplot + histogram for a numerical feature
- labeled_barplot: countplot/barplot with count or percentage labels
- stacked_barplot: crosstab + 100%-stacked bar chart of predictor vs target
- distribution_plot_wrt_target: distribution/boxplots of a feature split by target
- plot_churn_rate: churn rate (%) by a categorical column
- find_highest_risk_combinations: ranks combinations of categorical features
  by churn rate to surface the highest-risk customer segments

Usage (import into a notebook or another script):
    from scripts.visualization import histogram_boxplot, labeled_barplot, ...
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def histogram_boxplot(data, feature, figsize=(10, 7), kde=False, bins=None):
    """
    Boxplot and histogram combined for a numerical feature.

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (10,7))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,
        sharex=True,
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )
    sns.boxplot(data=data, x=feature, ax=ax_box2, showmeans=True, color="violet")
    if bins:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins)
    else:
        sns.histplot(data=data, x=feature, kde=kde, ax=ax_hist2)
    ax_hist2.axvline(data[feature].mean(), color="green", linestyle="--")
    ax_hist2.axvline(data[feature].median(), color="black", linestyle="-")
    plt.show()


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with count or percentage labels on top of each bar.

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of raw counts (default False)
    n: display only the top n category levels (default None, i.e. all levels)
    """
    total = len(data[feature])
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=10)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc:
            label = "{:.1f}%".format(100 * p.get_height() / total)
        else:
            label = p.get_height()

        x = p.get_x() + p.get_width() / 2
        y = p.get_height()

        ax.annotate(
            label, (x, y), ha="center", va="center", size=12,
            xytext=(0, 5), textcoords="offset points",
        )

    plt.show()


def stacked_barplot(data, predictor, target):
    """
    Print the category cross-tab counts and plot a 100%-stacked bar chart.

    data: dataframe
    predictor: independent variable (categorical column)
    target: target variable (e.g. Churn_New)
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()


def distribution_plot_wrt_target(data, predictor, target):
    """
    Plot histograms and boxplots of a numerical predictor split by each
    class of the target variable.
    """
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]], x=predictor, kde=True,
        ax=axs[0, 0], color="teal", stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[1]))
    sns.histplot(
        data=data[data[target] == target_uniq[1]], x=predictor, kde=True,
        ax=axs[0, 1], color="orange", stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data, x=target, y=predictor, ax=axs[1, 1],
        showfliers=False, palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()


def plot_churn_rate(df, column):
    """Plot churn rate (%) grouped by a categorical column."""
    churn_data = df.groupby(column)["Churn_New"].value_counts(normalize=True).unstack()
    churn_data = churn_data * 100

    churn_data.plot(kind="bar", stacked=False, figsize=(8, 5))
    plt.title(f"Churn Rate by {column}")
    plt.ylabel("Percentage (%)")
    plt.xlabel(column)
    plt.legend(title="Churn")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def find_highest_risk_combinations(df, features, max_combinations=20):
    """
    Rank combinations of categorical `features` by churn rate to surface
    the highest-risk customer segments.

    df: cleaned dataframe (must contain 'Churn_New' and a count column, e.g. 'gender')
    features: list of categorical column names to group by (e.g.
        ['Contract', 'PaymentMethod_New', 'TechSupport'])
    max_combinations: number of top rows to return
    """
    grouped = df.groupby(features).agg(
        Churn_Rate_pct=("Churn_New", lambda x: (x == "Yes").sum() / len(x) * 100),
        Customer_Count=("gender", "count"),
    )
    grouped = grouped.round(2).sort_values("Churn_Rate_pct", ascending=False)
    return grouped.head(max_combinations)
