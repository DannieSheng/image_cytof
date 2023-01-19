import os
import pickle as pkl
import skimage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import scipy
from typing import Union, Optional, Type, Tuple, List


def load_CytofImage(savename):
    cytof_img = pkl.load(open(savename, "rb"))
    return cytof_img

def load_CytofCohort(savename):
    cytof_cohort = pkl.load(open(savename, "rb"))
    return cytof_cohort

def save_multi_channel_img(img, savename):
    """
    A helper function to save multi-channel images
    """
    skimage.io.imsave(savename, img)


def generate_color_dict(names: List,
                        sort_names: bool = True,
                       ):
    """
    Randomly generate a dictionary of colors based on provided "names"
    """
    if sort_names:
        names.sort()

    color_dict = dict((n, plt.cm.get_cmap('tab20').colors[i]) for (i, n) in enumerate(names))
    return color_dict

def show_color_table(color_dict: dict, # = None,
                   # names: List = ['1'],
                   title: str = "",
                   maxcols: int = 4,
                   emptycols: int = 0,
                   # sort_names: bool = True,
                   dpi: int = 72,
                   cell_width: int = 212,
                   cell_height: int = 22,
                   swatch_width: int = 48,
                   margin: int = 12,
                   topmargin: int = 40,
                   show: bool = True
                   ):
    """
    Show color dictionary
    Generate the color table for visualization.
    If "color_dict" is provided, show color_dict;
    otherwise, randomly generate color_dict based on "names"
    reference: https://matplotlib.org/stable/gallery/color/named_colors.html
    args:
        color_dict (optional) = a dictionary of colors. key: color legend name - value: RGB representation of color
        names (optional) = names for each color legend (default=["1"])
        title (optional) = title for the color table (default="")
        maxcols = maximum number of columns in visualization
        emptycols (optional) = number of empty columns for a maxcols-column figure,
            i.e. maxcols=4 and emptycols=3 means presenting single column plot (default=0)
        sort_names (optional) = a flag indicating whether sort colors based on names (default=True)
    """

#     if sort_names:
#         names.sort()

#     if color_pool is None:
#         color_pool = dict((n, plt.cm.get_cmap('tab20').colors[i]) for (i, n) in enumerate(names))
#     else:
    names = color_dict.keys()

    n = len(names)
    ncols = maxcols - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    #     width  = cell_width * 4 + 2 * margin
    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + margin + topmargin

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - topmargin) / height)
    #     ax.set_xlim(0, cell_width * 4)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, loc="left", pad=10)

    for i, n in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, n, fontsize=12,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y - 9), width=swatch_width,
                      height=18, facecolor=color_dict[n], edgecolor='0.7')
        )


# def visualize_scatter(data, communities, n_community, title, figsize=(4,4), savename=None, show=False):
#     """
#     data = data to visualize (N, 2)
#     communities = group indices correspond to each sample in data (N, 1) or (N, )
#     n_community = total number of groups in the cohort (n_community >= unique number of communities)
#     """
#     fig, ax = plt.subplots(1,1, figsize=figsize)
#     ax.set_title(title)
#     sns.scatterplot(x=data[:,0], y=data[:,1], hue=communities, palette='tab20',
#                     hue_order=np.arange(n_community))
#                     #                 legend=legend,
#                     # hue_order=np.arange(n_community))
#     plt.axis('tight')
#     plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
#     if savename is not None:
#         print("saving plot to {}".format(savename))
#         plt.savefig(savename)
#     if show:
#         plt.show()
#         return None
#     return fig

def visualize_scatter(data, communities, n_community, title, figsize=(4,4), savename=None, show=False, ax=None):
    """
    data = data to visualize (N, 2)
    communities = group indices correspond to each sample in data (N, 1) or (N, )
    n_community = total number of groups in the cohort (n_community >= unique number of communities)
    """
    clos = not show and ax is None
    show = show and ax is None
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = None
    ax.set_title(title)
    sns.scatterplot(x=data[:,0], y=data[:,1], hue=communities, palette='tab20',
                    hue_order=np.arange(n_community), ax=ax)
                    #                 legend=legend,
                    # hue_order=np.arange(n_community))
    
    ax.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    # plt.axis('tight')
    if savename is not None:
        print("saving plot to {}".format(savename))
        plt.savefig(savename)
    if show:
        plt.show()
    if clos:
        plt.close('all')
    return fig

def visualize_expression(data, markers, group_ids, title, figsize=(2,2), savename=None, show=False, ax=None):
    clos = not show and ax is None
    show = show and ax is None
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
    else:
        fig = None

    sns.heatmap(data,
                cmap='magma',
                xticklabels=markers,
                yticklabels=group_ids,
                ax=ax
               )
    ax.set_xlabel("Markers")
    ax.set_ylabel("Phenograph clusters")
    ax.set_title("normalized expression - {}".format(title))
    # plt.axis('tight')
    if savename is not None:
        plt.savefig(savename)
    if show:
        plt.show()
    if clos:
        plt.close('all')
    return fig

def _get_thresholds(df_feature: pd.DataFrame,
                    features: List[str],
                    thres_bg: float = 0.3,
                    visualize: bool = True,
                    verbose: bool = False):
    """Calculate thresholds for each feature by assuming a Gaussian Mixture Model
    Inputs:
        df_feature = dataframe of extracted feature summary
        features   = a list of features to calculate thresholds from
        thres_bg   = a threshold such that the component with the mixing weight greater than the threshold would
                            be considered as background. (Default=0.3)
        visualize  = a flag indicating whether to visualize the feature distributions and thresholds or not.
                            (Default=True)
        verbose    = a flag indicating whether to print calculated values on screen or not. (Default=False)
    Outputs:
        thresholds = a dictionary of calculated threshold values
    :param df_feature: pandas.core.frame.DataFrame
    :param features: list
    :param visualize: bool
    :param verbose: bool
    :return thresholds: dict
    """
    thresholds = {}
    for f, feat_name in enumerate(features):
        X = df_feature[feat_name].values.reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=0, n_init=2).fit(X)
        mu = np.min(gm.means_[gm.weights_ > thres_bg])
        which_component = np.argmax(gm.means_ == mu)

        if verbose:
            print(f"GMM mean values: {gm.means_}")
            print(f"GMM weights: {gm.weights_}")
            print(f"GMM covariances: {gm.covariances_}")

        X     = df_feature[feat_name].values
        hist  = np.histogram(X, 150)
        sigma = np.sqrt(gm.covariances_[which_component, 0, 0])
        background_ratio = gm.weights_[which_component]
        thres = sigma * 2.5 + mu
        thresholds[feat_name] = thres

        n = sum(X > thres)
        percentage = n / len(X)

        ## visualize
        if visualize:
            fig, ax = plt.subplots(1, 1)
            ax.hist(X, 150, density=True)
            ax.set_xlabel("log2({})".format(feat_name))
            ax.plot(hist[1], scipy.stats.norm.pdf(hist[1], mu, sigma) * background_ratio, c='red')
            ax.axvline(x=thres, c='red')
            ax.text(0.7, 0.9, "n={}, percentage={}".format(n, np.round(percentage, 3)), ha='center', va='center',
                    transform=ax.transAxes)
            ax.text(0.3, 0.9, "mu={}, sigma={}".format(np.round(mu, 2), np.round(sigma, 2)), ha='center', va='center',
                    transform=ax.transAxes)
            ax.text(0.3, 0.8, "background ratio={}".format(np.round(background_ratio, 2)), ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(feat_name)
            plt.show()
    return thresholds

def _generate_summary(df_feature: pd.DataFrame, features: List[str], thresholds: dict) -> pd.DataFrame:
    """Generate (cell level) summary table for each feature in features: feature name, total number (of cells),
        calculated GMM threshold for this feature, number of individuals (cells) with greater than threshold values,
        ratio of individuals (cells) with greater than threshold values
    Inputs:
        df_feature = dataframe of extracted feature summary
        features   = a list of features to generate summary table
        thresholds = (calculated GMM-based) thresholds for each feature
    Outputs:
        df_info    = summary table for each feature

    :param df_feature: pandas.core.frame.DataFrame
    :param features: list
    :param thresholds: dict
    :return df_info: pandas.core.frame.DataFrame
    """

    df_info = pd.DataFrame(columns=['feature', 'total number', 'threshold', 'positive counts', 'positive ratio'])

    for feature in features:  # loop over each feature
        thres = thresholds[feature]  # fetch threshold for the feature
        X = df_feature[feature].values
        n = sum(X > thres)
        N = len(X)

        df_new_row = pd.DataFrame({'feature': feature, 'total number': N, 'threshold': thres,
                                   'positive counts': n, 'positive ratio': n / N}, index=[0])
        df_info = pd.concat([df_info, df_new_row])
    return df_info.reset_index(drop=True)