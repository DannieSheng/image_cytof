import os
import re
import glob
import pickle as pkl
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import warnings
from tqdm import tqdm
import skimage
from sklearn.mixture import GaussianMixture
import phenograph
import umap
import seaborn as sns

import hyperion_preprocess as pre
import hyperion_segmentation as seg


def _longest_substring(str1, str2):
    ans = ""
    len1, len2 = len(str1), len(str2)
    for i in range(len1):
        for j in range(len2):
            match = ""
            _len = 0
            while ((i+_len < len1) and (j+_len < len2) and str1[i+_len] == str2[j+_len]):
                match += str1[i+_len]
                _len += 1
                if len(match) > len(ans):
                    ans = match
    return ans

def extract_feature(channels, raw_image, nuclei_seg, cell_seg, filename, show_head=False):
    """ Extract nuclei and cell level feature from cytof image based on nuclei segmentation and cell segmentation
        results
    Inputs:
        channels   = channels to extract feature from
        raw_image  = raw cytof image
        nuclei_seg = nuclei segmentation result
        cell_seg   = cell segmentation result
        filename   = filename of current cytof image
    Returns:
        feature_summary_df = a dataframe containing summary of extracted features
        morphology         = names of morphology features extracted

    :param channels: list
    :param raw_image: numpy.ndarray
    :param nuclei_seg: numpy.ndarray
    :param cell_seg: numpy.ndarray
    :param filename: string
    :param morpholoty: list
    :return feature_summary_df: pandas.core.frame.DataFrame
    """
    assert (len(channels) == raw_image.shape[-1])

    # morphology features to be extracted
    morphology = ["area", "convex_area", "eccentricity", "extent",
                "filled_area", "major_axis_length", "minor_axis_length",
                "orientation", "perimeter", "solidity", "pa_ratio"]

    ## morphology features
    nuclei_morphology = [_ + '_nuclei' for _ in morphology]  # morphology - nuclei level
    cell_morphology = [_ + '_cell' for _ in morphology]  # morphology - cell level

    ## single cell features
    # nuclei level
    sum_exp_nuclei = [_ + '_nuclei_sum' for _ in channels]  # sum expression over nuclei
    ave_exp_nuclei = [_ + '_nuclei_ave' for _ in channels]  # average expression over nuclei

    # cell level
    sum_exp_cell   = [_ + '_cell_sum' for _ in channels]  # sum expression over cell
    ave_exp_cell   = [_ + '_cell_ave' for _ in channels]  # average expression over cell

    # column names of final result dataframe
    column_names       = ["filename", "id", "coordinate_x", "coordinate_y"] + \
                         sum_exp_nuclei + ave_exp_nuclei + nuclei_morphology + \
                         sum_exp_cell + ave_exp_cell + cell_morphology

    # Initiate
    res = dict()
    for column_name in column_names:
        res[column_name] = []

    n_nuclei = np.max(nuclei_seg)
    for nuclei_id in tqdm(range(2, n_nuclei + 1), position=0, leave=True):
        res["filename"].append(filename)
        res["id"].append(nuclei_id)
        regions = skimage.measure.regionprops((nuclei_seg == nuclei_id) * 1)  # , coordinates='xy') (deprecated)
        if len(regions) >= 1:
            this_nucleus = regions[0]
        else:
            continue
        regions = skimage.measure.regionprops((cell_seg == nuclei_id) * 1)  # , coordinates='xy') (deprecated)
        if len(regions) >= 1:
            this_cell = regions[0]
        else:
            continue
        centroid_y, centroid_x = this_nucleus.centroid  # y: rows; x: columns
        res['coordinate_x'].append(centroid_x)
        res['coordinate_y'].append(centroid_y)

        # morphology
        for i, feature in enumerate(morphology[:-1]):
            res[nuclei_morphology[i]].append(getattr(this_nucleus, feature))
            res[cell_morphology[i]].append(getattr(this_cell, feature))
        res[nuclei_morphology[-1]].append(1.0 * this_nucleus.perimeter ** 2 / this_nucleus.filled_area)
        res[cell_morphology[-1]].append(1.0 * this_cell.perimeter ** 2 / this_cell.filled_area)

        # markers
        for i, marker in enumerate(channels):
            ch = i
            res[sum_exp_nuclei[i]].append(np.sum(raw_image[nuclei_seg == nuclei_id, ch]))
            res[ave_exp_nuclei[i]].append(np.average(raw_image[nuclei_seg == nuclei_id, ch]))
            res[sum_exp_cell[i]].append(np.sum(raw_image[cell_seg == nuclei_id, ch]))
            res[ave_exp_cell[i]].append(np.average(raw_image[cell_seg == nuclei_id, ch]))

    feature_summary_df = pd.DataFrame(res)
    if show_head:
        print(feature_summary_df.head())
    return feature_summary_df


###############################################################################
def check_feature_distribution(feature_summary_df, features):
    """ Visualize feature distribution for each feature
    Inputs:
        feature_summary_df = dataframe of extracted feature summary
        features           = features to check distribution
    Returns:
        None

    :param feature_summary_df: pandas.core.frame.DataFrame
    :param features: list
    """

    for feature in features:
        fig, ax = plt.subplots(1, 1, figsize=(3,2))
        print(feature)
        ax.hist(np.log2(feature_summary_df[feature] + 0.0001), 100)
        ax.set_xlim(-15, 15)
        plt.show()



def feature_quantile_normalization(feature_summary_df, features, qs=[75,99]):
    """ Calculate the q-quantiles of selected features given quantile q values. Then perform q-quantile normalization
     on these features using calculated quantile values. The feature_summary_df will be updated in-place with new
     columns "feature_qnormed" generated and added. Meanwhile, visualize distribution of log2 features before and after
     q-normalization
    Inputs:
        feature_summary_df = dataframe of extracted feature summary
        features           = features to be normalized
        qs                 = quantile q values (default=[75,99])
    Returns:
        quantiles          = quantile values for each q
    :param feature_summary_df: pandas.core.frame.DataFrame
    :param features: list
    :param qs: list
    :return quantiles: dict
    """
    expressions = []
    expressions_normed = dict((key, []) for key in qs)
    quantiles   = {}
    colors = cm.rainbow(np.linspace(0, 1, len(qs)))
    for feat in features:
        quantiles[feat] = {}
        expressions.extend(feature_summary_df[feat])

        plt.hist(np.log2(np.array(expressions) + 0.0001), 100, density=True)
        for q, c in zip(qs, colors):
            quantile_val = np.quantile(expressions, q/100)
            quantiles[feat][q] = quantile_val
            plt.axvline(np.log2(quantile_val), label=f"{q}th quantile", c=c)
            print(f"{q}th quantile: {quantile_val}")

            # log-quantile normalization
            normed = np.log2(feature_summary_df.loc[:, feat] / quantile_val + 0.0001)
            feature_summary_df.loc[:, f"{feat}_{q}normed"] = normed
            expressions_normed[q].extend(normed)
        plt.xlim(-15, 15)
        plt.xlabel("log2(expression of all markers)")
        plt.legend()
        plt.show()

    # visualize before & after quantile normalization
    '''N = len(qs)+1 # (len(qs)+1) // 2 + (len(qs)+1) %2'''
    log_expressions = tuple([np.log2(np.array(expressions) + 0.0001)] + [expressions_normed[q] for q in qs])
    labels = ["before normalization"] + [f"after {q} normalization" for q in qs]
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.hist(log_expressions, 100, density=True, label=labels)
    ax.set_xlabel("log2(expressions for all markers)")
    plt.legend()
    plt.show()
    return quantiles


def feature_scaling(feature_summary_df, features, inplace=False):
    """Perform in-place mean-std scaling on selected features. Normally, do not scale nuclei sum feature
    Inputs:
        feature_summary_df = dataframe of extracted feature summary
        features           = features to perform scaling on
        inplace            = an indicator of whether perform the scaling in-place (Default=False)
    Returns:

    :param feature_summary_df: pandas.core.frame.DataFrame
    :param features: list
    :param inplace: bool
    """

    scaled_feature_summary_df = feature_summary_df if inplace else feature_summary_df.copy()

    for feat in features:
        if feat not in feature_summary_df.columns:
            print(f"Warning: {feat} not available!")
            continue
        scaled_feature_summary_df[feat] = \
            (scaled_feature_summary_df[feat] - np.average(scaled_feature_summary_df[feat])) \
            / np.std(scaled_feature_summary_df[feat])
    if not inplace:
        return scaled_feature_summary_df


import scipy
def _get_thresholds(feature_summary_df, features, visualize=True, verbose=False):
    """Calculate thresholds for each feature by assuming a Gaussian Mixture Model
    Inputs:
        feature_summary_df = dataframe of extracted feature summary
        features           = a list of features to calculate thresholds from
        visualize          = a flag indicating whether or not visualize the feature distributions and thresholds.
                            (Default=True)
        verbose            = a flag indicating whether or not print calculated values on screen. (Default=False)
    Outputs:
        thresholds         = a dictionary of calculated threshold values
    :param feature_summary_df: pandas.core.frame.DataFrame
    :param features: list
    :param visualize: bool
    :param verbose: bool
    :return thresholds: dict
    """
    thresholds = {}
    for f, feat_name in enumerate(features):
        X = feature_summary_df[feat_name].values.reshape(-1, 1)
        gm = GaussianMixture(n_components=2, random_state=0, n_init=2).fit(X)
        mu = np.min(gm.means_[gm.weights_ > 0.3])
        which_component = np.argmax(gm.means_ == mu)

        if verbose:
            print(f"GMM mean values: {gm.means_}")
            print(f"GMM weights: {gm.weights_}")
            print(f"GMM covariances: {gm.covariances_}")

        X = feature_summary_df[feat_name].values
        hist = np.histogram(X, 150)
        sigma = np.sqrt(gm.covariances_[which_component, 0, 0])
        background_ratio = gm.weights_[which_component]
        thres = sigma * 2.5 + mu
        thresholds[feat_name] = thres

        n = sum(X > thres)
        percentage = n / len(X)

        # visualize
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


def generate_summary(feature_summary_df, features, vis_thres=False):
    """Generate (cell level) summary table for each feature in features: feature name, total number (of cells),
        calculated GMM threshold for this feature, number of individuals (cells) with greater than threshold values,
        ratio of individuals (cells) with greater than threshold values
    Inputs:
        feature_summary_df = dataframe of extracted feature summary
        features           = a list of features to generate summary table
        vis_thres          = a flag indicating whether or not visualize the process of calculating thresholds
                            (Default=False)
    Outputs:
        df_info    = summary table for each feature
        thresholds = calculated GMM-based thresholds for each feature
    :param feature_summary_df: pandas.core.frame.DataFrame
    :param features: list
    :param visualize: bool
    :return df_info: pandas.core.frame.DataFrame
    :return thresholds: dict
    """

    df_info = pd.DataFrame(columns=['feature', 'total number', 'threshold', 'positive counts', 'positive ratio'])
    thresholds = _get_thresholds(feature_summary_df, features, visualize=vis_thres)
    for feature in features:
        # calculate threshold
        thres = thresholds[feature]
        X = feature_summary_df[feature].values
        n = sum(X > thres)
        N = len(X)

        df_new_row = pd.DataFrame({'feature': feature,'total number':N, 'threshold':thres,
                                  'positive counts':n, 'positive ratio': n/N}, index=[0])
        df_info = pd.concat([df_info, df_new_row])

        # df_info = df_info.append({'feature': feature,'total number':N, 'threshold':thres,
        #                           'positive counts':n, 'positive ratio': n/N}, ignore_index=True)
    return df_info, thresholds


def visualize_thresholding_outcome(feat,
                                   feature_summary_df,
                                   raw_image,
                                   channel_names,
                                   thres,
                                   nuclei_seg,
                                   cell_seg,
                                   vis_quantile_q=0.9, savepath=None):
    """ Visualize calculated threshold for a feature by mapping back to nuclei and cell segmentation outputs - showing
        greater than threshold pixels in red color, others with blue color.
        Meanwhile, visualize the original image with red color indicating the channel correspond to the feature.
    Inputs:
        feat               = name of the feature to visualize
        feature_summary_df = dataframe of extracted feature summary
        raw_image          = raw cytof image
        channel_names       = a list of marker names, which is consistent with each channel in the raw_image
        thres              = threshold value for feature "feat"
        nuclei_seg         = nuclei segmentation output
        cell_seg           = cell segmentation output
    Outputs:
        stain_nuclei       = nuclei segmentation output stained with threshold information
        stain_cell         = cell segmentation output stained with threshold information
    :param feat: string
    :param feature_summary_df: pandas.core.frame.DataFrame
    :param raw_image: numpy.ndarray
    :param channel_names: list
    :param thres: float
    :param nuclei_seg: numpy.ndarray
    :param cell_seg: numpy.ndarray
    :return stain_nuclei: numpy.ndarray
    :return stain_cell: numpy.ndarray
    """
    col_name = channel_names[np.argmax([len(_longest_substring(feat, x)) for x in channel_names])]
    col_id   = channel_names.index(col_name)
    df_temp = pd.DataFrame(columns=[f"{feat}_overthres"], data=np.zeros(len(feature_summary_df), dtype=np.int32))
    df_temp.loc[feature_summary_df[feat] > thres, f"{feat}_overthres"] = 1
    feature_summary_df = pd.concat([feature_summary_df, df_temp], axis=1)
    # feature_summary_df.loc[:, f"{feat}_overthres"] = 0
    # feature_summary_df.loc[feature_summary_df[feat] > thres, f"{feat}_overthres"] = 1

    '''rgba_color = [plt.cm.get_cmap('tab20').colors[_ % 20] for _ in feature_summary_df.loc[:, f"{feat}_overthres"]]'''
    color_ids  = []

    # stained Nuclei image
    stain_nuclei = np.zeros((nuclei_seg.shape[0], nuclei_seg.shape[1], 3)) + 1
    for i in range(2, np.max(nuclei_seg) + 1):
        color_id = feature_summary_df[f"{feat}_overthres"][feature_summary_df['id'] == i].values[0] * 2
        if color_id not in color_ids:
            color_ids.append(color_id)
        stain_nuclei[nuclei_seg == i] = plt.cm.get_cmap('tab20').colors[color_id][:3]

    # stained Cell image
    stain_cell = np.zeros((cell_seg.shape[0], cell_seg.shape[1], 3)) + 1
    for i in range(2, np.max(cell_seg) + 1):
        color_id = feature_summary_df[f"{feat}_overthres"][feature_summary_df['id'] == i].values[0] * 2
        stain_cell[cell_seg == i] = plt.cm.get_cmap('tab20').colors[color_id][:3]

    fig, axs = plt.subplots(1,3,figsize=(16, 8))
    channel_ids = (col_id, 0)
    '''print(channel_ids)'''
    quantiles = [np.quantile(raw_image[..., _], vis_quantile_q) for _ in channel_ids]
    vis_img, _ = pre.cytof_merge_channels(raw_image, channel_names=channel_names,
                                          channel_ids=channel_ids, quantiles=quantiles)
    marker = feat.split("(")[0]
    print(f"Nuclei and cell with high {marker} expression shown in orange, low in blue.")

    axs[0].imshow(vis_img)
    axs[1].imshow(stain_nuclei)
    axs[2].imshow(stain_cell)
    axs[0].set_title("pseudo-colored original image")
    axs[1].set_title(f"{marker} expression shown in nuclei")
    axs[2].set_title(f"{marker} expression shown in cell")
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()
    return stain_nuclei, stain_cell, vis_img


########################################################################################################################
############################################### batch functions ########################################################
########################################################################################################################
def batch_extract_feature(files, markers, nuclei_markers, membrane_markers=None, show_vis=False):
    """Extract features for cytof images from a list of files. Normally this list contains ROIs of the same slide
    Inputs:
        files            = a list of files to be processed
        markers          = a list of marker names used when generating the image
        nuclei_markers   = a list of markers define the nuclei channel (used for nuclei segmentation)
        membrane_markers = a list of markers define the membrane channel (used for cell segmentation) (Default=None)
        show_vis         = an indicator of showing visualization during process
    Outputs:
        file_features    = a dictionary contains extracted features for each file

    :param files: list
    :param markers: list
    :param nuclei_markers: list
    :param membrane_markers: list
    :param show_vis: bool
    :return file_features: dict
    """
    file_features = {}
    for f in tqdm(files):
        # read data
        df = pre.cytof_read_data(f)
        # preprocess
        df_ = pre.cytof_preprocess(df)
        column_names = markers[:]
        df_output = pre.define_special_channel(df_, 'nuclei', markers=nuclei_markers)
        column_names.insert(0, 'nuclei')
        if membrane_markers is not None:
            df_output = pre.define_special_channel(df_output, 'membrane', markers=membrane_markers)
            column_names.append('membrane')
        raw_image = pre.cytof_txt2img(df_output, marker_names=column_names)

        if show_vis:
            merged_im, _ = pre.cytof_merge_channels(raw_image, channel_ids=[0, -1], quantiles=None, visualize=False)
            plt.imshow(merged_im[0:200, 200:400, ...])
            plt.title('Selected region of raw cytof image')
            plt.show()

        # nuclei and cell segmentation
        nuclei_img = raw_image[..., column_names.index('nuclei')]
        nuclei_seg, color_dict = seg.cytof_nuclei_segmentation(nuclei_img, show_process=False)
        if membrane_markers is not None:
            membrane_img = raw_image[..., column_names.index('membrane')]
            cell_seg, _ = seg.cytof_cell_segmentation(nuclei_seg, membrane_channel=membrane_img, show_process=False)
        else:
            cell_seg, _ = seg.cytof_cell_segmentation(nuclei_seg, show_process=False)
        if show_vis:
            marked_image_nuclei = seg.visualize_segmentation(raw_image, nuclei_seg, channel_ids=(0, -1), show=False)
            marked_image_cell = seg.visualize_segmentation(raw_image, cell_seg, channel_ids=(-1, 0), show=False)
            fig, axs = plt.subplots(1,2,figsize=(10,6))
            axs[0].imshow(marked_image_nuclei[0:200, 200:400, :]), axs[0].set_title('nuclei segmentation')
            axs[1].imshow(marked_image_cell[0:200, 200:400, :]), axs[1].set_title('cell segmentation')
            plt.show()

        # feature extraction
        feat_names = markers[:]
        feat_names.insert(0, 'nuclei')
        df_feat_sum = extract_feature(feat_names, raw_image, nuclei_seg, cell_seg, filename=f)
        file_features[f] = df_feat_sum
    return file_features



def batch_norm_scale(file_features, column_names, qs=[75,99]):
    """Perform feature log transform, quantile normalization and scaling in a batch
    Inputs:
        file_features = A dictionary of dataframes containing extracted features. key - file name, item - feature table
        column_names  = A list of markers. Should be consistent with column names in dataframe of features
        qs            = quantile q values (Default=[75,99])
    Outputs:
        file_features_out = log transformed, quantile normalized and scaled features for each file in the batch
        quantiles         = a dictionary of quantile values for each file in the batch

    :param file_features: dict
    :param column_names: list
    :param qs: list
    :return file_features_out: dict
    :return quantiles: dict
    """
    file_features_out = copy.deepcopy(file_features) # maintain a copy of original file_features

    # marker features
    cell_markers_sum   = [_ + '_cell_sum' for _ in column_names]
    cell_markers_ave   = [_ + '_cell_ave' for _ in column_names]
    nuclei_markers_sum = [_ + '_nuclei_sum' for _ in column_names]
    nuclei_markers_ave = [_ + '_nuclei_ave' for _ in column_names]

    # morphology features
    morphology = ["area", "convex_area", "eccentricity", "extent",
                  "filled_area", "major_axis_length", "minor_axis_length",
                  "orientation", "perimeter", "solidity", "pa_ratio"]
    nuclei_morphology = [_ + '_nuclei' for _ in morphology]  # morphology - nuclei level
    cell_morphology   = [_ + '_cell' for _ in morphology]  # morphology - cell level

    # features to be normalized
    features_to_norm = [x for x in nuclei_markers_sum + nuclei_markers_ave + cell_markers_sum + cell_markers_ave \
                        if not x.startswith('nuclei')]

    # features to be scaled
    scale_features = []
    for feature_name in nuclei_morphology + cell_morphology + nuclei_markers_sum + nuclei_markers_ave + \
                        cell_markers_sum + cell_markers_ave:
        '''if feature_name not in nuclei_morphology + cell_morphology and not feature_name.startswith('nuclei'):
            scale_features += [feature_name, f"{feature_name}_75normed", f"{feature_name}_99normed"]
        else:
            scale_features += [feature_name]'''
        temp = [feature_name]
        if feature_name not in nuclei_morphology + cell_morphology and not feature_name.startswith('nuclei'):
            for q in qs:
                temp += [f"{feature_name}_{q}normed"]
        scale_features += temp

    quantiles = {}
    for f, df in file_features_out.items():
        print(f)
        quantiles[f] = feature_quantile_normalization(df, features=features_to_norm, qs=qs)
        feature_scaling(df, features=scale_features, inplace=True)
    return file_features_out, quantiles



def batch_scale_feature(outdir, file_scale, normq):
    """
    Inputs:
        outdir     = output saving directory, which contains the scale file generated previously,
                     the input_output_csv file with the list of available cytof_img class instances in the batch,
                     as well as previously saved cytof_img class instances in .pkl files
        file_scale = full file name of the scaling information
        normq      = q value of quantile normalization
    Outputs: None
        Scaled feature are saved as .csv files in subfolder "feature_qnormed_scaled" in outdir
        A new attribute will be added to cytof_img class instance, and the update class instance is saved in outdir
    """
    n_attr = f"df_feature_{normq}normed"
    n_attr_scaled = f"{n_attr}_scaled"

    # saving directory of scaled normed feature
    dirq = os.path.join(outdir, f"feature_{normq}normed_scaled")
    if not os.path.exists(dirq):
        os.makedirs(dirq)

    # load scaling parameters
    df_scale = pd.read_csv(os.path.join(outdir, file_scale), index_col=False)
    m = df_scale[df_scale.columns].iloc[0]
    s = df_scale[df_scale.columns].iloc[1]

    dfs = {}
    cytofs = {}
    cytofs_ = pd.read_csv(os.path.join(outdir, "input_output.csv"))['output_file'].tolist()

    # loop over all image_cytof classs instances, scale attribute "df_feature", add attribute "df_feature_scaled" and
    # save scaled feature
    for f in cytofs_:
        f_roi = os.path.basename(f).split(".pkl")[0]
        cytof_img = pkl.load(open(f, "rb"))
        assert hasattr(cytof_img, n_attr), f"attribute {n_attr} not exist"
        df_feat = copy.deepcopy(getattr(cytof_img, n_attr))

        assert len([x for x in df_scale.columns if x not in df_feat.columns]) == 0

        # scale
        df_feat[df_scale.columns] = (df_feat[df_scale.columns] - m) / s

        # save scaled feature to csv
        df_feat.to_csv(os.path.join(dirq, f"{f_roi}.csv"), index=False)

        # add attribute "df_feature_scaled"
        setattr(cytof_img, n_attr_scaled, df_feat)

        # save updated cytof_img class instance
        pkl.dump(cytof_img, open(f, "wb"))

        dfs[f_roi] = df_feat
        cytofs[f_roi] = cytof_img

#     return dfs, cytofs

#     # loop over all normalized feature files to scale features
#     for f in glob.glob(os.path.join(outdir, "*{}".format(suffix))):
#         print(f)
#         f_roi_  = os.path.basename(f).split(suffix)[0]
#         print(f_roi_)
#         df_feat = pd.read_csv(f, index_col=False)
#         assert len([x for x in df_scale.columns if x not in df_feat.columns]) == 0
#         # load corresponding image_cytof class instance, add attribute "df_feature_scaled" and save
#         cytof_img = pkl.load(open(glob.glob(os.path.join(outdir, f"{f_roi_}*.pkl"))[0], "rb"))

#         # scale
#         df_feat[df_scale.columns] = (df_feat[df_scale.columns]-m)/s

#         # save scaled feature
#         df_feat.to_csv(os.path.join(params.outdir, suffix.replace(".csv", "_scaled.csv")), index=False)

#         cytof_img.df_feature_scaled = df_feat

# dfs, cytofs =


def batch_generate_summary(outdir, feature_type="normed", normq=75, scaled=True):
    """
    Inputs:
        outdir       = output saving directory, which contains the scale file generated previously, as well as previously saved
                     cytof_img class instances in .pkl files
        normq        = q value of quantile normalization
        feature_type = type of feature to be used, available choices: "original", "normed", "scaled"
    Outputs: None
        Two .csv files, one for cell sum and the other for cell average features, are saved for each ROI, containing the
        threshold and cell count information of each feature, in the subfolder "marker_summary" under outdir
    """
    assert feature_type in ["original", "normed", "scaled"], 'accepted feature types are "original", "normed", "scaled"'
    if feature_type == "original":
        feat_name = ""
    elif feature_type == "normed":
        feat_name = f"{normq}normed"
    else:
        feat_name = f"{normq}normed_scaled"

    n_attr = f"df_feature_{feat_name}"

    dir_sum = os.path.join(outdir, "marker_summary", feat_name)
    print(dir_sum)
    if not os.path.exists(dir_sum):
        os.makedirs(dir_sum)

    seen = 0
    for f in pd.read_csv(os.path.join(outdir, "input_output.csv"))['output_file'].tolist():
        f_roi = os.path.basename(f).split(".pkl")[0]
        cytof_img = pkl.load(open(f, "rb"))
        df_feat = getattr(cytof_img, n_attr)
        if seen == 0:
            feat_cell_sum = cytof_img.features['cell_sum']
            feat_cell_ave = cytof_img.features['cell_ave']
        # summary for cell sum features
        df_info_cell_sum, thresholds_cell_sum = generate_summary(df_feat, features=feat_cell_sum, vis_thres=True)
        # summary for cell average features
        df_info_cell_ave, thresholds_cell_sum = generate_summary(df_feat, features=feat_cell_ave, vis_thres=True)

        # Attach summary to cytof_img class instance
        setattr(cytof_img, f"cell_count_{feat_name}_sum", df_info_cell_sum)
        setattr(cytof_img, f"cell_count_{feat_name}_ave", df_info_cell_ave)
        df_info_cell_sum.to_csv(os.path.join(dir_sum, f"{f_roi}_cell_count_sum.csv"), index=False)
        df_info_cell_ave.to_csv(os.path.join(dir_sum, f"{f_roi}_cell_count_ave.csv"), index=False)
        pkl.dump(cytof_img, open(f, "wb"))
        seen += 1
    return dir_sum


def _vis_cell_phenotypes(df_feat, communities, n_community, markers, list_features, accumul_type="sum", savedir=None):
    """ Visualize cell phenotypes for a given dataframe of feature
    Args:
        df_feat: a dataframe of features
        communities: a list of communities (can be a subset of the cohort communities, but should be consistent with df_feat)
        n_community: number of communities in the cohort (n_community >= number of unique values in communities)
        markers: a list of markers used in CyTOF image (to be present in the heatmap visualization)
        list_features: a list of feature names (consistent with columns in df_feat)
        accumul_type: feature aggregation type, choose from "sum" and "ave" (default="sum")
        savedir: results saving directory. If not None, visualization plots will be saved in the desired directory (default=None)
    Returns:
        cell_cluster: a (N, M) matrix, where N = # of clustered communities, and M = # of markers

        cell_cluster_norm: the normalized form of cell_cluster (normalized by subtracting the median value)
    """
    assert accumul_type in ["sum", "ave"], "Wrong accumulation type! Choose from 'sum' and 'ave'!"
    cell_cluster= np.zeros((n_community, len(markers)))
    for cluster in range(len(np.unique(communities))):
        df_sub = df_feat[communities == cluster]
        if df_sub.shape[0] == 0:
            continue
        # for each feature in the list of features
        for i, feat in enumerate(list_features):
            cell_cluster[cluster, i] = np.average(df_sub[feat])
    cell_cluster_norm = cell_cluster - np.median(cell_cluster, axis=0)
    sns.heatmap(cell_cluster - np.median(cell_cluster, axis=0),#cell_cluster_,
                cmap='magma',
                xticklabels=markers,
                yticklabels=np.arange(len(np.unique(communities)))
               )
    plt.xlabel("Markers")
    plt.ylabel("Phenotype")
    fname = "phenotypes"
    if accumul_type == "sum":
        plt.title("normalized expression - cell sum")
        fname += "-cell_sum.png"
    elif accumul_type == "ave":
        plt.title("normalized expression - cell average")
        fname += "-cell_ave.png"
    if savedir is not None:
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, fname))
    plt.show()
    return cell_cluster, cell_cluster_norm

def batch_co_expression_analysis(outdir, feature_type, accumul_type, normq=75):
    """ Perform co-expression analysis in a batch
    Inputs:
        outdir       = output saving directory, previously saved cytof_img class instances in .pkl files
        feature_type = type of feature to be used, available choices: "original", "normed", "scaled"
        accumul_type = type of marker expression accumulation, acceptable choices: "sum", "ave"
        normq        = q value of quantile normalization
    Outputs: None
        Two .csv files, one for cell sum and the other for cell average features, are saved for each ROI, containing the
        threshold and cell count information of each feature, in the subfolder "marker_summary" under outdir
    """
    assert feature_type in ["original", "normed", "scaled"]
    if feature_type == "original":
        feat_name = ""
    elif feature_type == "normed":
        feat_name = f"_{normq}normed"
    else:
        feat_name = f"_{normq}normed_scaled"
    n_attr = f"df_feature{feat_name}"

    clustered = False
    for f_cytof_im in glob.glob(os.path.join(outdir, "*.pkl")):
        f_roi = os.path.basename(f_cytof_im).split("_cytof_img.pkl")[0]
        print(f_roi)
        cytof_im = pkl.load(open(f_cytof_im, "rb"))

        df_feat = getattr(cytof_im, n_attr)

        # all gene (marker) columns
        marker_col_all = [x for x in df_feat.columns if "cell_{}".format(accumul_type) in x]
        marker_all = [x.split('(')[0] for x in marker_col_all]

        n_marker = len(marker_col_all)
        n_cell = len(df_feat)

        # corresponding thresholding info file
        f_sum = glob.glob(os.path.join(outdir, "marker_summary", f"{f_roi}_*{accumul_type}.csv"))[0]
        df_info_cell = pd.read_csv(f_sum, index_col=False)
        pos_nums = df_info_cell["positive counts"]
        pos_ratios = df_info_cell["positive ratio"]

        # expected_percentage
        # an N by N matrix, where N represent for the number of total gene (marker)
        # each ij-th element represents for the percentage that both the i-th and the j-th gene is "positive"
        # based on the threshold defined previously
        expected_percentage = np.zeros((n_marker, n_marker))
        for ii in range(n_marker):
            for jj in range(n_marker):
                expected_percentage[ii, jj] = pos_ratios[ii] * pos_ratios[jj]
        thresholds = df_info_cell["threshold"]

        # Co-expression
        # an N by N matrix, where N represent for the number of gene (marker)
        # each ij-th element represents for the percentage of cells that show positive in both i-th and j-th gene
        edge_nums = np.zeros_like(expected_percentage)
        for ii in range(n_marker):
            _x = df_feat[marker_col_all[ii]].values > thresholds[ii]
            for jj in range(n_marker):
                _y = df_feat[marker_col_all[jj]].values > thresholds[jj]
                edge_nums[ii, jj] = np.sum(np.all([_x, _y], axis=0)) / n_cell
        edge_percentages = edge_nums

        # Normalize
        edge_percentages_norm = np.log10(edge_percentages / expected_percentage + 0.1)

        # Fix Nan
        edge_percentages_norm[np.isnan(edge_percentages_norm)] = np.log10(1 + 0.1)

        # Plot
        plt.figure(figsize=(6, 4))
        ax = sns.heatmap(edge_percentages_norm, center=np.log10(1 + 0.1),
                         cmap='RdBu_r', vmin=-1, vmax=3,
                         xticklabels=marker_all, yticklabels=marker_all)
        plt.show()

        if not clustered:
            plt.figure(figsize=(6, 4))
            clustergrid = sns.clustermap(edge_percentages_norm,
                                         center=np.log10(1 + 0.1), cmap='RdBu_r', vmin=-1, vmax=3,
                                         xticklabels=marker_all, yticklabels=marker_all, figsize=(6, 6))
            plt.title(f_roi)
            plt.show()
            clustered = True
        else:
            plt.figure(figsize=(6, 4))
            sns.clustermap(edge_percentages_norm[clustergrid.dendrogram_row.reordered_ind, :] \
                               [:, clustergrid.dendrogram_row.reordered_ind],
                           center=np.log10(1 + 0.1), cmap='RdBu_r', vmin=-1, vmax=3,
                           xticklabels=np.array(marker_all)[clustergrid.dendrogram_row.reordered_ind],
                           yticklabels=np.array(marker_all)[clustergrid.dendrogram_row.reordered_ind],
                           figsize=(6, 6), row_cluster=False, col_cluster=False)
            plt.title(f_roi)
            plt.show()


def clustering_phenograph(cohort_file, outdir, normq=75, feat_comb="all", k=None, save_vis=False):
    """Perform Pheno-graph clustering for the cohort
        Inputs:
            cohort_file  = a .csv file include the whole cohort
            outdir       = output saving directory, previously saved cytof_img class instances in .pkl files
            normq        = q value for quantile normalization
            feat_comb    = desired feature combination to be used for phenograph clustering, acceptable choices: "all",
                        "cell_sum", "cell_ave", "cell_sum_only", "cell_ave_only" (Default="all")
            k            = number of initial neighbors to run Pheno-graph (Default=None)
                        If k is not provided, k is set to N / 100, where N is the total number of single cells
            save_vis     = a flag indicating whether to save the visualization output (Default=False)
    Outputs:
        df_all = a dataframe of features for all cells in the cohort, with the clustering output saved in the column
        'phenotype_total{n_community}', where n_community stands for the total number of communities defined by the cohort
        Also, each individual cytof_img class instances will be updated with 2 new attributes:
        1)"num phenotypes ({feat_comb}_{normq}normed_{k})"
        2)"phenotypes ({feat_comb}_{normq}normed_{k})"
    """
    feat_groups = {
        "all": ["cell_sum", "cell_ave", "cell_morphology"],
        "cell_sum": ["cell_sum", "cell_morphology"],
        "cell_ave": ["cell_ave", "cell_morphology"],
        "cell_sum_only": ["cell_sum"],
        "cell_ave_only": ["cell_ave"]
    }
    assert feat_comb in feat_groups.keys(), f"{feat_comb} not supported!"

    feat_name = f"_{normq}normed_scaled"
    n_attr    = f"df_feature{feat_name}"

    dfs = {}
    cytof_ims = {}

    df_io = pd.read_csv(os.path.join(outdir, "input_output.csv"))
    df_slide_roi = pd.read_csv(cohort_file)

    df_slide_roi["input_file"] = df_slide_roi[["path", "ROI"]].apply(lambda row: '/'.join(row.values.astype(str)),
                                                                     axis=1)
    df_slide_roi["input_file"] = df_slide_roi["input_file"] + ".txt"

    df_info = df_slide_roi.merge(df_io, on="input_file")

    # load all scaled feature in the cohort
    for i in df_info.index:
        f_in = df_info.loc[i, "input_file"]
        f_out = df_info.loc[i, "output_file"]
        f_roi = f_in.split('/')[-1].split('.txt')[0]
        if not os.path.isfile(f_out):
            continue

        cytof_img = pkl.load(open(f_out, "rb"))
        if i == 0:
            dict_feat = cytof_img.features
            markers = cytof_img.markers
        cytof_ims[f_roi] = cytof_img
        dfs[f_roi] = getattr(cytof_img, n_attr)
    feat_names = [x for y in feat_groups[feat_comb] for x in dict_feat[y]]
    # concatenate feature dataframes of all rois in the cohort
    df_all = pd.concat([_ for key, _ in dfs.items()])

    # set number of nearest neighbors k and run PhenoGraph for phenotype clustering
    k = k if k else int(df_all.shape[0] / 100)  # 100
    communities, graph, Q = phenograph.cluster(df_all[feat_names], k=k, n_jobs=-1)  # run PhenoGraph
    n_community = len(np.unique(communities))

    # Visualize
    ## project to 2D
    umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
    proj_2d = umap_2d.fit_transform(df_all[feat_names])

    # plot together
    print("Visualization in 2d - cohort")
    plt.figure(figsize=(4, 4))
    plt.title("cohort")
    sns.scatterplot(x=proj_2d[:, 0], y=proj_2d[:, 1], hue=communities, palette='tab20',
                    #                 legend=legend,
                    hue_order=np.arange(n_community))
    plt.axis('tight')
    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    if save_vis:
        vis_savedir = os.path.join(outdir, "phenograph_{}_{}normed_{}".format(feat_comb, normq, k))
        if not os.path.exists(vis_savedir):
            os.makedirs(vis_savedir)
        plt.savefig(os.path.join(vis_savedir, "cluster_scatter.png"))
    plt.show()

    # cohort
    print("Showing cell phenotypes for the cohort")
    for fname in feat_groups[feat_comb]:

        if "morphology" in fname:
            continue
        if "sum" in fname:
            accumul_type = "sum"
        elif "ave" in fname:
            accumul_type = "ave"
        # print(f"Accumulation type: {accumul_type}")
        list_features = dict_feat[fname]

        cell_cluster, cell_cluster_ = _vis_cell_phenotypes(df_all, communities,
                                                           n_community, markers,
                                                           list_features, accumul_type,
                                                           savedir=vis_savedir)

    # attach clustering output to df_all
    df_all[f'phenotype_total{n_community}'] = communities

    # split df_all to single ROIs
    # load all scaled feature in the cohort
    for i in df_info.index:
        f_in = df_info.loc[i, "input_file"]
        f_out = df_info.loc[i, "output_file"]
        f_roi = f_in.split('/')[-1].split('.txt')[0]
        cond = df_all["filename"] == f_in
        '''if not os.path.isfile(f_out):
            continue'''
        setattr(cytof_ims[f_roi], "phenograph", {})
        cytof_ims[f_roi].phenograph[f"{feat_comb}_{normq}normed_{k}"] = {
            "feat_name": n_attr,
            "num_community": n_community,
            "phenotypes": df_all.loc[cond, f'phenotype_total{n_community}']
        }
        # setattr(cytof_ims[f_roi], f"num phenotypes ({feat_comb}_{normq}normed_{k})", n_community)
        # setattr(cytof_ims[f_roi], f"phenotypes ({feat_comb}_{normq}normed_{k})", communities[cond])

        # save updated CyTOF image class instances
        pkl.dump(cytof_ims[f_roi], open(f_out, "wb"))

    return df_all, k#, cytof_ims



