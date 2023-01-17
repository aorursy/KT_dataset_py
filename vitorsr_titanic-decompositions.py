from titanic_utility_script import *



train, test = read_data_()

train, test = get_title_feature_(train, test)

train, test = get_surname_(train, test)

train, test = get_first_names_(train, test)

train, test = get_married_feature_(train, test)

train, test = get_family_counts_(train, test)

train, test = extract_ticket_number_(train, test)

train, test = get_ticket_length_(train, test)

train, test = get_ticket_counts_(train, test)

train, test = drop_unused_(train, test)

train, test = clean_data_(train, test)

train, test = format_categories_(train, test)
def plot_decompositions_(train, test):

    from pandas import concat, DataFrame

    from sklearn.decomposition import FastICA, KernelPCA, PCA

    from sklearn.manifold import Isomap

    from numpy import hstack, ones

    from umap import UMAP

    from matplotlib.pyplot import figure, scatter, subplot, title, xticks, yticks

    from itertools import combinations



    X, X_submit, y = get_X_y_(train, test)

    X, X_submit = encode_categories_(X, X_submit, y)

    X, X_submit = impute_missing_(X, X_submit)

    df = concat([X, X_submit])



    n_c, r_s = {"n_components": 4}, {"random_state": 0}



    pca = PCA(**n_c, **r_s).fit(df)

    kpc = KernelPCA(**n_c, kernel="rbf", gamma=0.4, **r_s, n_jobs=-1).fit(df)

    ica = FastICA(**n_c, max_iter=2000, tol=1e-3, **r_s).fit(df)

    iso = Isomap(**n_c).fit(df)

    ump = UMAP(**n_c, **r_s).fit(df, hstack([y, -ones(X_submit.shape[0])]))



    df_pca = DataFrame(pca.transform(df)).iloc[: X.shape[0]]

    df_kpc = DataFrame(kpc.transform(df)).iloc[: X.shape[0]]

    df_ica = DataFrame(ica.transform(df)).iloc[: X.shape[0]]

    df_iso = DataFrame(iso.transform(df)).iloc[: X.shape[0]]

    df_ump = DataFrame(ump.transform(df)).iloc[: X.shape[0]]



    del df



    n_decompositions = 5

    df_list = [df_pca, df_kpc, df_ica, df_iso, df_ump]

    title_list = ["PCA", "KernelPCA", "ICA", "Isomap", "UMAP"]



    _, s = figure(num=0, figsize=(15, n_decompositions * 2 + 1)), 1

    for df, t in zip(df_list, title_list):

        for i, j in combinations([0, 1, 2, 3], 2):

            _, s = subplot(n_decompositions, 6, s), s + 1

            _ = (

                scatter(df[i], df[j], s=2, c=y),

                title("{} ({}, {})".format(t, i, j)),

                xticks([]),

                yticks([]),

            )
plot_decompositions_(train, test)