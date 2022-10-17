import itertools
import re
import warnings
import os
import copy
import pickle as pkl
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm
from skimage.segmentation import mark_boundaries
from hyperion_segmentation import cytof_nuclei_segmentation, cytof_cell_segmentation, visualize_segmentation


# def _get_colors(n):
#     base_colors = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1),
#                             (0, 1, 1), (1, 0, 1), (1, 1, 0),
#                             (1, 1, 1)])

#     n0 = len(base_colors)
#     if n <= n0:
#         colours = base_colors[:n]
#     else:
#         colours = np.vstack((base_colors, cm.rainbow(np.linspace(0, 1, n-n0))[:,:-1]))
#     return colours


def _save_multi_channel_img(img, savename):
    """
    A helper function to save multi-channel images
    """
    skimage.io.imsave(savename, img)


def _get_colortable(color_pool=None, names=['1'], title="", emptycols=0, sort_names=True):
    """
    Generate the color table for visualization.
    If "color_pool" is provided, use color_pool; otherwise, randomly generate colors based on "names"
    reference: https://matplotlib.org/stable/gallery/color/named_colors.html
    args:
        color_pool (optional) = a dictionary of colors. key: color legend name - value: RGB representation of color
        names (optional) = names for each color legend (default=["1"])
        title (optional) = title for the color table (default="")
        emptycols (optional) = number of empty columns for a four-column figure,
            i.e. emptycols=3 means presenting single column plot (default=0)
        sort_names (optional) = a flag indicating whether sort colors based on names (default=True)
    """
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    if sort_names:
        names.sort()

    if color_pool is None:
        color_pool = dict((n, plt.cm.get_cmap('tab20').colors[i]) for (i, n) in enumerate(names))
    else:
        names = color_pool.keys()

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin / width, margin / height,
                        (width - margin) / width, (height - topmargin) / height)
    ax.set_xlim(0, cell_width * 4)
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
                      height=18, facecolor=color_pool[n], edgecolor='0.7')
        )
    return fig, color_pool


def load_saved_CytofImage(savename):
    cytof_img = pkl.load(open(savename, "rb"))
    return cytof_img

class CytofImage():
    def __init__(self, df=None, slide="", roi="", filename=""):
        self.df       = df
        self.slide    = slide
        self.roi      = roi
        self.filename = filename
        self.columns  = None # column names in original cytof data (dataframe)
        self.markers  = None # protein markers
        self.labels   = None # metal isotopes used to tag protein

        self.image    = None
        self.channels = None # channel names correspond to each channel of self.image

        self.features = None



    def save_cytof(self, savename):
        pkl.dump(self, open(savename, "wb"))

    def get_markers(self, imarker0=None):
        """ Get (1) the channel names correspond to each image channel
                (2) a list of protein markers used to obtain the CyTOF image
                (3) a list of labels tagged to each of the protein markers
        """
        self.columns = list(self.df.columns)
        if imarker0 is not None:  # if the index of the 1st marker provided
            self.raw_channels = self.columns[imarker0:]
        else:  # assumption: channel names have the common expression: marker(label*)
            pattern = "\w+.*\(\w+\)"
            self.raw_channels = [re.findall(pattern, t)[0] for t in self.columns if len(re.findall(pattern, t)) > 0]

        self.raw_markers = [x.split('(')[0] for x in self.raw_channels]
        self.raw_labels = [x.split('(')[-1].split(')')[0] for x in self.raw_channels]

        self.channels = self.raw_channels.copy()
        self.markers  = self.raw_markers.copy()
        self.labels   = self.raw_labels.copy()


    def preprocess(self):
        nrow = max(self.df['Y'].values) + 1
        ncol = max(self.df['X'].values) + 1
        n = len(self.df)
        if nrow * ncol > n:
            df2 = pd.DataFrame(np.zeros((nrow * ncol - n, len(self.df.columns)), dtype=int),
                               columns=self.df.columns)
            self.df = pd.concat([self.df, df2])

    def check_channels(self, channels=None, xlim=None, ylim=None, ncols=5, vis_q=0.9, colorbar=False):
        """
        xlim = a list of 2 numbers indicating the ylimits to show image (default=None)
        ylim = a list of 2 numbers indicating the ylimits to show image (default=None)
        ncols = number of subplots per row (default=5)
        vis_q = percentile q used to normalize image before visualization  (default=0.9)
        """
        if channels is not None:
            if not all([cl.lower() in self.channels for cl in channels]):
                print("At least one of the channels not available, visualizing all channels instead!")
                channels = None
        if channels is None:  # if no desired channels specified, check all channels
            channels = self.channels
        nrow = max(self.df['Y'].values) + 1
        ncol = max(self.df['X'].values) + 1
        if len(channels) <= ncols:
            ax_nrow = 1
            ax_ncol = len(channels)
        else:
            ax_ncol = ncols
            ax_nrow = int(np.ceil(len(channels) / ncols))
        fig, axes = plt.subplots(ax_nrow, ax_ncol, figsize=(3 * ax_ncol, 3 * ax_nrow))
        if ax_nrow == 1:
            axes = np.array([axes])
            if ax_ncol == 1:
                axes = np.expand_dims(axes, axis=1)
        for i, _ in enumerate(channels):
            _ax_nrow = int(np.floor(i / ax_ncol))
            _ax_ncol = i % ax_ncol
            image = self.df[_].values.reshape(nrow, ncol)
            percentile_q = np.quantile(image, vis_q) if np.quantile(image, vis_q)!= 0 else 1
            image = np.clip(image / percentile_q, 0, 1)
            axes[_ax_nrow, _ax_ncol].set_title(_)
            if xlim is not None:
                image = image[:, xlim[0]:xlim[1]]
            if ylim is not None:
                image = image[ylim[0]:ylim[1], :]
            im = axes[_ax_nrow, _ax_ncol].imshow(image, cmap="gray")
            if colorbar:
                fig.colorbar(im, ax=axes[_ax_nrow, _ax_ncol])
        plt.tight_layout()
        plt.show()

    def remove_special_channels(self, channels):
        for channel in channels:
            if channel not in self.channels:
                print("Channel {} not available, escaping...".format(channel))
                continue
            idx = self.channels.index(channel)
            self.channels.pop(idx)
            self.markers.pop(idx)
            self.labels.pop(idx)
            self.df.drop(columns=channel, inplace=True)

    def define_special_channels(self, channels_dict):
        # create a copy of original dataframe
        self.df_orig = self.df.copy()
        channels_rm = []
        for new_name, old_names in channels_dict.items():

            if len(old_names) == 0:
                continue

            old_nms = []
            for i, old_name in enumerate(old_names):
                if old_name not in self.channels:
                    warnings.warn('{} is not available!'.format(old_name['marker_name']))
                    continue
                old_nms.append(old_name)
            print("Defining channel '{}' by summing up channels: {}.".format(new_name, ', '.join(old_nms)))
            if len(old_nms) > 0:
                channels_rm += old_nms
                for i, old_name in enumerate(old_nms):
                    if i == 0:
                        self.df[new_name] = self.df[old_name]
                    else:
                        self.df[new_name] += self.df[old_name]
                self.channels.append(new_name)
        return channels_rm

    def get_image(self, channels=None, inplace=True):
        """ Get channel images based on provided channels. By default, get channel images correspond to all channels
        """
        if channels is not None:
            if not all([cl in self.channels for cl in channels]):
                print("At least one of the channels not available, using default all channels instead!")
                channels = self.channels
                inplace = True
        else:
            channels = self.channels
            inplace = True
        nc = len(channels)
        nrow = max(self.df['Y'].values) + 1
        ncol = max(self.df['X'].values) + 1
        print("Output image shape: [{}, {}, {}]".format(nrow, ncol, nc))

        target_image = np.zeros([nrow, ncol, nc], dtype=float)
        for _nc in range(nc):
            target_image[..., _nc] = self.df[channels[_nc]].values.reshape(nrow, ncol)
        if inplace:
            self.image = target_image
        else:
            return target_image

    def visualize_channels(self, channel_ids=None, channel_names=None, quantiles=None, visualize=False):
        assert channel_ids or channel_names, 'At least one should be provided, either "channel_ids" or "channel_names"!'
        if channel_ids is None:
            channel_ids = [self.channels.index(n) for n in channel_names]
        else:
            channel_names = [self.channels[i] for i in channel_ids]
        assert len(channel_ids) <= 7, "No more than 6 channels can be visualized simultaneously!"
        if len(channel_ids) > 3:
            warnings.warn(
                "Visualizing more than 3 channels the same time results in deteriorated visualization. \
                It is not recommended!")

        print("Visualizing channels: {}".format(', '.join(channel_names)))
        full_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white']
        color_values = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                        (0, 1, 1), (1, 0, 1), (1, 1, 0),
                        (1, 1, 1)]
        info = ["{} in {}\n".format(marker, c) for (marker, c) in \
                zip([self.channels[i] for i in channel_ids], full_colors[:len(channel_ids)])]
        print("Visualizing... \n{}".format(''.join(info)))
        merged_im = np.zeros((self.image.shape[0], self.image.shape[1], 3))
        if quantiles is None:
            quantiles = [np.quantile(self.image[..., _], 0.99) for _ in channel_ids]

        # max_vals = []
        for _ in range(min(len(channel_ids), 3)):  # first 3 channels, assign colors R, G, B
            gs = np.clip(self.image[..., channel_ids[_]] / quantiles[_], 0, 1)  # grayscale
            merged_im[..., _] = gs * 255
            max_val = [0, 0, 0]
            max_val[_] = gs.max() * 255
            # max_vals.append(max_val)

        chs = [[1, 2], [0, 2], [0, 1], [0, 1, 2]]
        chs_id = 0
        while _ < len(channel_ids) - 1:
            _ += 1
            max_val = [0, 0, 0]
            for j in chs[chs_id]:
                gs = np.clip(self.image[..., channel_ids[_]] / quantiles[_], 0, 1)
                merged_im[..., j] += gs * 255  # /2
                merged_im[..., j] = np.clip(merged_im[..., j], 0, 255)
                max_val[j] = gs.max() * 255
            chs_id += 1
            # max_vals.append(max_val)
        merged_im = merged_im.astype(np.uint8)
        if visualize:
            fig, ax = plt.subplots(1, 1)
            ax.imshow(merged_im)
            plt.show()

        vis_markers = [self.markers[i] if i < len(self.markers) else self.channels[i] for i in channel_ids]

        color_pool = dict((n, c) for (n, c) in zip(vis_markers, color_values[:len(channel_ids)]))
        fig_color_tb, color_pool = _get_colortable(color_pool=color_pool,
                                                   # title="",
                                                   emptycols=3, sort_names=True)
        return merged_im, quantiles, color_pool, fig_color_tb

    def get_seg(self, use_membrane=True, radius=5,
                sz_hole=1, sz_obj=3, min_distance=2, fg_marker_dilate=2, bg_marker_dilate=2,
                show_process=False):
        nuclei_img = self.image[..., self.channels.index('nuclei')]

        if show_process:
            print("Nuclei segmentation...")
        # else:
        #     print("Not showing segmentation process")
        nuclei_seg, color_dict = cytof_nuclei_segmentation(nuclei_img, show_process=show_process,
                                                           size_hole=sz_hole, size_obj=sz_obj,
                                                           fg_marker_dilate=fg_marker_dilate,
                                                           bg_marker_dilate=bg_marker_dilate,
                                                           min_distance=min_distance)

        membrane_img = self.image[..., self.channels.index('membrane')] \
            if (use_membrane and 'membrane' in self.channels) else None
        if show_process:
            print("Cell segmentation...")
        cell_seg, _ = cytof_cell_segmentation(nuclei_seg, radius, membrane_channel=membrane_img,
                                              show_process=show_process, colors=color_dict)

        self.nuclei_seg = nuclei_seg
        self.cell_seg   = cell_seg
        return nuclei_seg, cell_seg

    def visualize_seg(self, segtype="cell", seg=None, show=False, bg_label=1):
        assert segtype in ["nuclei", "cell"]
        # nuclei in red, membrane in green
        if "membrane" in self.channels:
            channel_ids = [self.channels.index(_) for _ in ["nuclei", "membrane"]]
        else:
            channel_ids = [self.channels.index("nuclei"), 0]
        if seg is None:
            if segtype == "cell":
                seg = self.cell_seg
                '''# membrane in red, nuclei in green
                channel_ids = [self.channels.index(_) for _ in ["membrane", "nuclei"]]'''
            else:
                seg = self.nuclei_seg

        marked_image = visualize_segmentation(self.image, self.channels, seg, channel_ids=channel_ids, show=show, bg_label=bg_label)
        print("{} boundary marked by white".format(segtype))
        return marked_image

    def extract_features(self, filename):
        from hyperion_analysis import extract_feature

        '''if channels is None:
            channels = self.channels
        else:
            temp = set(channels) - set(self.channels)
            if len(temp) > 0:
                channels = list(set(self.channels) - temp)
                print(f"Warning: {', '.join(temp)} not in available channels and are removed!")'''

        pattern = "\w+.*\(\w+\)"
        marker_idx      = [i for (i,x) in enumerate(self.channels) if len(re.findall(pattern, x))>0] # channel indices correspond to pure markers
        marker_channels = [self.channels[i] for i in marker_idx] # pure marker channels
        marker_image    = self.image[...,marker_idx] # channel images correspond to pure markers
        morphology = ["area", "convex_area", "eccentricity", "extent",
                      "filled_area", "major_axis_length", "minor_axis_length",
                      "orientation", "perimeter", "solidity", "pa_ratio"]
        self.features = {
            "nuclei_morphology": [_ + '_nuclei' for _ in morphology],  # morphology - nuclei level
            "cell_morphology": [_ + '_cell' for _ in morphology],  # morphology - cell level
            "cell_sum": [_ + '_cell_sum' for _ in marker_channels],
            "cell_ave": [_ + '_cell_ave' for _ in marker_channels],
            "nuclei_sum": [_ + '_nuclei_sum' for _ in marker_channels],
            "nuclei_ave": [_ + '_nuclei_ave' for _ in marker_channels],
        }
        self.df_feature = extract_feature(marker_channels, marker_image,
                                          self.nuclei_seg, self.cell_seg,
                                          filename, show_head=False)

    def calculate_quantiles(self, qs=[75, 99], savename=None):
        """ Calculate the q-quantiles of each marker with cell level summation given the q values
        """
        _expressions_cell_sum = []
        quantiles = {}
        colors = cm.rainbow(np.linspace(0, 1, len(qs)))
        for feature_name in self.features["cell_sum"]:  # all cell sum features except for nuclei_cell_sum and membrane_cell_sum
            if feature_name.startswith("nuclei") or feature_name.startswith("membrane"):
                continue
            _expressions_cell_sum.extend(self.df_feature[feature_name])

        plt.hist(np.log2(np.array(_expressions_cell_sum) + 0.0001), 100, density=True)
        for q, c in zip(qs, colors):
            quantiles[q] = np.quantile(_expressions_cell_sum, q / 100)
            plt.axvline(np.log2(quantiles[q]), label=f"{q}th percentile", c=c)
            print(f"{q}th percentile: {quantiles[q]}")
        plt.xlim(-15, 15)
        plt.xlabel("log2(expression of all markers)")
        plt.legend()
        if savename is not None:
            plt.savefig(savename)
        plt.show()
        # attach quantile dictionary to self
        self.dict_quantiles = quantiles
        # return quantiles

    def _vis_normalization(self, savename=None):
        """
        Compare before and after normalization
        """
        expressions = {}
        expressions["original"] = []

        ## before normalization
        for key, features in self.features.items():
            if key.endswith("morphology"):
                continue
            for feature_name in features:
                if feature_name.startswith('nuclei') or feature_name.startswith('membrane'):
                    continue
                expressions["original"].extend(self.df_feature[feature_name])
        log_exp = np.log2(np.array(expressions['original']) + 0.0001)
        plt.hist(log_exp, 100, density=True, label='before normalization')

        for q in self.dict_quantiles.keys():
            n_attr = f"df_feature_{q}normed"
            expressions[f"{q}_normed"] = []

            for key, features in self.features.items():
                if key.endswith("morphology"):
                    continue
                for feature_name in features:
                    if feature_name.startswith('nuclei') or feature_name.startswith('membrane'):
                        continue
                    expressions[f"{q}_normed"].extend(getattr(self, n_attr)[feature_name])
            plt.hist(expressions[f"{q}_normed"], 100, density=True, label=f"after {q}th percentile normalization")

        plt.legend()
        plt.xlabel('log2(expressions of all markers)')
        plt.ylabel('Frequency')
        if savename is not None:
            plt.savefig(savename)
        plt.show()
        return expressions

    def feature_quantile_normalization(self, qs=[75,99], vis_compare=True, savedir=None):
        """Normalize all features with given quantiles except for morphology features"""
        if savedir is not None:
            savename_quantile = os.path.join(savedir, "{}_{}_percentiles.png".format(self.slide, self.roi))
            savename_compare  = os.path.join(savedir, "{}_{}_comparison.png".format(self.slide, self.roi))
        else:
            savename_quantile, savename_compare = None, None
        self.calculate_quantiles(qs, savename=savename_quantile)
        for q, quantile_val in self.dict_quantiles.items():
            n_attr = f"df_feature_{q}normed" # attribute name
            log_normed = copy.deepcopy(self.df_feature)
            for key, features in self.features.items():
                if key.endswith("morphology"):
                    continue
                for feature_name in features:
                    if feature_name.startswith("nuclei") or feature_name.startswith("membrane"):
                        continue
                    # log-quantile normalization
                    log_normed.loc[:, feature_name] = np.log2(log_normed.loc[:, feature_name] / quantile_val + 0.0001)
            setattr(self, n_attr, log_normed)
        if vis_compare:
            _ = self._vis_normalization(savename=savename_compare)

    '''def feature_quantile_normalization(self, qs=[75,99]):
        """Normalize all features with given quantiles except for morphology features"""
        self.calculate_quantiles(qs)
        for key, features in self.features.items():
            if key.endswith("morphology"):
                continue
            for feature_name in features:
                if feature_name.startswith("nuclei") or feature_name.startswith("membrane"):
                    continue
                for q, quantile_val in self.dict_quantiles.items():
                    # log-quantile normalization
                    log_normed = np.log2(self.df_feature.loc[:, feature_name] / quantile_val + 0.0001)
                    df_temp = pd.DataFrame({f"{feature_name}_{q}normed": log_normed})
                    self.df_feature = pd.concat((self.df_feature, df_temp), axis=1)'''

    '''def feature_scaling(self):
        for key, features in self.features.items():
            cols = [col for f in features for col in self.df_feature.columns if col.startswith(f)]
            for col in cols:
                avg = np.average(self.df_feature[col])
                std = np.std(self.df_feature[col])
                self.df_feature[col] = (self.df_feature[col] - avg) / std'''

    def save_channel_images(self, savedir, channels=None, ext=".png", quantile_norm=99):
        """Save channel images

        """
        if channels is not None:
            if not all([cl in self.channels for cl in channels]):
                print("At least one of the channels not available, saving all channels instead!")
                channels = self.channels
        else:
            channels = self.channels
        '''assert all([x.lower() in channels_temp for x in channels]), "Not all provided channels are available!"'''
        # for (i, chn) in enumerate(channels):
        #     savename = os.path.join(savedir, f"{chn}.tiff")
        #     im_temp = self.image[..., i]
        #     im_temp_ = np.clip(im_temp / np.quantile(im_temp, 0.99), 0, 1)
        #     _save_multi_channel_img((im_temp_ * 255).astype(np.uint8), savename)
        for chn in channels:
            savename = os.path.join(savedir, f"{chn}{ext}")
            #         i = channels_temp.index(chn.lower())
            i = self.channels.index(chn)
            im_temp = self.image[..., i]
            quantile_temp = np.quantile(im_temp, quantile_norm / 100) \
                if np.quantile(im_temp, quantile_norm / 100) != 0 else 1

            im_temp_ = np.clip(im_temp / quantile_temp, 0, 1)
            _save_multi_channel_img((im_temp_ * 255).astype(np.uint8), savename)

    # def vis_marker_thresholding(self, marker, feature_type="normed", accumul_type="sum",
    #                             normq=75, vis_quantile_q=0.9, savefig=False, savedir="./", roiname="test"):
    #     """Visualize a marker's expression compared to the threshold
    #     Inputs:
    #         marker         = the desired marker to visualize result
    #         normq          = the q valued used to normalize feature
    #         accumul_type   = the aggregation of feature in a cell, either "sum" or "ave". (default="sum")
    #         scaled         = the flag indicating whether or not using scaled feature. (default=True)
    #         vis_quantile_q = the quantile value used to normalize the image when visualize. (default=0.9)
    #     Outputs:
    #     """
    #     from hyperion_analysis import visualize_thresholding_outcome
    #     assert feature_type in ["original", "normed",
    #                             "scaled"], 'accepted feature types are "original", "normed", "scaled"'
    #     if feature_type == "original":
    #         feat_name = ""
    #     elif feature_type == "normed":
    #         feat_name = f"_{normq}normed"
    #     else:
    #         feat_name = f"_{normq}normed_scaled"
    #
    #     n_attr = f"df_feature{feat_name}"
    #     count_attr = f"cell_count{feat_name}_{accumul_type}"
    #
    #     df_feat  = getattr(self, n_attr)
    #     df_thres = getattr(self, count_attr)
    #
    #     thresholds_cell_marker = dict((x, y) for (x, y) in zip(df_thres["feature"], df_thres["threshold"]))
    #
    #     marker_ = "{}_cell_{}".format([x for x in self.channels if re.search(marker, x)][0], accumul_type)
    #     if savefig:
    #         if not os.path.exists(savedir):
    #             os.makedirs(savedir)
    #         savepath = os.path.join(savedir, roiname + "_{}_{}.png".format(marker, accumul_type))
    #     else:
    #         savepath=None
    #
    #
    #     stain_nuclei, stain_cell, vis_img = visualize_thresholding_outcome(
    #         feat=marker_,
    #         feature_summary_df=df_feat,
    #         raw_image=self.image,
    #         channel_names=self.channels,
    #         thres=thresholds_cell_marker[marker_],
    #         nuclei_seg=self.nuclei_seg,
    #         cell_seg=self.cell_seg,
    #         vis_quantile_q=vis_quantile_q,
    #         savepath=savepath
    #     )
    #
    #     return stain_nuclei, stain_cell, vis_img
    def marker_positive(self, feature_type="normed", accumul_type="sum", normq=75):
        assert feature_type in ["original", "normed",
                                "scaled"], 'accepted feature types are "original", "normed", "scaled"'
        if feature_type == "original":
            feat_name = ""
        elif feature_type == "normed":
            feat_name = f"_{normq}normed"
        else:
            feat_name = f"_{normq}normed_scaled"

        n_attr = f"df_feature{feat_name}"  # class attribute name for feature table
        count_attr = f"cell_count{feat_name}_{accumul_type}"  # class attribute name for feature summary table

        df_feat = getattr(self, n_attr)
        df_thres = getattr(self, count_attr)

        thresholds_cell_marker = dict((x, y) for (x, y) in zip(df_thres["feature"], df_thres["threshold"]))

        columns = ["cell_id"] + [marker for marker in self.markers]
        df_marker_positive = pd.DataFrame(columns=columns, data=np.zeros((len(df_feat), len(self.markers) + 1),
                                                                         dtype=np.int32))
        df_marker_positive["cell_id"] = df_feat["id"]
        for im, marker in enumerate(self.markers):
            channel_ = "{}_cell_{}".format(self.channels[im], accumul_type)
            df_marker_positive.loc[df_feat[channel_] > thresholds_cell_marker[channel_], marker] = 1
        setattr(self, "df_marker_positive{}".format(feat_name), df_marker_positive)

    def vis_marker_positive(self,
                            marker,
                            feature_type,
                            accumul_type="sum",
                            normq=99,
                            show_boundary=True,
                            color_list = [(0,0,1), (0,1,0)], # negative, positive
                            color_bound = (0,0,0)
                            ):
        assert feature_type in ["original", "normed",
                                "scaled"], 'accepted feature types are "original", "normed", "scaled"'
        if feature_type == "original":
            feat_name = ""
        elif feature_type == "normed":
            feat_name = f"_{normq}normed"
        else:
            feat_name = f"_{normq}normed_scaled"
        if not hasattr(self, "df_marker_positive{}".format(feat_name)):
            self.marker_positive(feature_type=feature_type, accumul_type=accumul_type, normq=normq)
        df_marker_positive = getattr(self, "df_marker_positive{}".format(feat_name))

        #     color_list = [plt.cm.get_cmap('tab20').colors[x] for x in [0,2]]

        color_dict = dict((key, v) for (key, v) in zip(['negative', 'positive'], color_list))
        c_table = _get_colortable(color_pool=color_dict, title="", emptycols=3, sort_names=True)
        color_ids = []

        # stained Nuclei image
        stain_nuclei = np.zeros((self.nuclei_seg.shape[0], self.nuclei_seg.shape[1], 3)) + 1
        for i in range(2, np.max(self.nuclei_seg) + 1):
            color_id = df_marker_positive[marker][df_marker_positive['cell_id'] == i].values[0]
            if color_id not in color_ids:
                color_ids.append(color_id)
            stain_nuclei[self.nuclei_seg == i] = color_list[color_id][:3]
        # add boundary
        if show_boundary:
            stain_nuclei = mark_boundaries(stain_nuclei,
                                       self.nuclei_seg, mode="inner", color=color_bound)

        # stained Cell image
        stain_cell = np.zeros((self.cell_seg.shape[0], self.cell_seg.shape[1], 3)) + 1
        for i in range(2, np.max(self.cell_seg) + 1):
            color_id = df_marker_positive[marker][df_marker_positive['cell_id'] == i].values[0]
            stain_cell[self.cell_seg == i] = color_list[color_id][:3]
        if show_boundary:
            stain_cell = mark_boundaries(stain_cell,
                                     self.cell_seg, mode="inner", color=color_bound)
        return stain_nuclei, stain_cell

    def visualize_phenograph(self, name_pheno, channels, save_vis=False, dir_save="./", pheno_summary=False):
        """
        Visualize phenograph clustering
        """
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        orig_im, quantiles = self.visualize_channels(channel_names=channels, visualize=False)
        phenograph = getattr(self, 'phenograph')[name_pheno]
        communities, feat_name = phenograph['clusters'], phenograph['feat_name']
        df_feat = getattr(self, feat_name)

        # create color_pool
        fig_color, color_pool = _get_colortable(names=[n for n in communities.unique()],
                                                title="Phenograph clusters colors",
                                                emptycols=3)

        ## Stain nuclei and cell segmentation based on phenograph clustering output
        stain_nuclei = np.zeros((self.nuclei_seg.shape[0], self.nuclei_seg.shape[1], 3)) + 1
        for i in range(2, np.max(self.nuclei_seg) + 1):  # for each nuclei
            stain_nuclei[self.nuclei_seg == i] = color_pool[communities[df_feat['id'] == i].values[0]][:3]

        stain_cell = np.zeros((self.cell_seg.shape[0], self.cell_seg.shape[1], 3)) + 1
        for i in range(2, np.max(self.cell_seg) + 1):
            stain_cell[self.cell_seg == i] = color_pool[communities[df_feat['id'] == i].values[0]][:3]

        plt.figure(figsize=(24, 8))
        plt.subplot(131)
        plt.title('Original image')
        plt.imshow(orig_im)
        plt.subplot(132)
        plt.title('Showing cluster in nuclei segmentation')
        plt.imshow(stain_nuclei)
        plt.subplot(133)
        plt.title('Showing cluster in cell segmentation')
        plt.imshow(stain_cell)
        plt.show()

        # option to export visualization
        if save_vis:
            fig_color.savefig(os.path.join(dir_save, "color_dict.png"))
            _save_multi_channel_img(orig_im, os.path.join(dir_save, "orig_{}.png".format('_'.join(channels))))
            _save_multi_channel_img(np.array(stain_nuclei * 255, dtype=np.uint8),
                                    os.path.join(dir_save, "phenograph_nuclei_seg.png"))
            _save_multi_channel_img(np.array(stain_cell * 255, dtype=np.uint8),
                                    os.path.join(dir_save, "phenograph_cell_seg.png"))

        # option to generate simple quantification of phenograph results
        df_PG_sum = pd.DataFrame(columns=["cluster", "number of cells", "percentage"])
        if pheno_summary:
            n_total = len(df_feat)
            print("Total cell number n={}".format(n_total))

            for cluster in communities.unique():
                n_cell = np.sum(communities == cluster)
                # print("{}: n={}, percentage={}".format(cluster, n_cluster, n_cluster/len(df_roi)))
                percentage = n_cell / n_total
                df_PG_sum = df_PG_sum.append({"cluster": cluster,
                                              "number of cells": n_cell,
                                              "percentage": percentage}, ignore_index=True)
                df_PG_sum["total number of cells"] = n_total
                df_PG_sum.sort_values(by='cluster', inplace=True)
            df_PG_sum.to_csv(os.path.join(dir_save, "phenograph_summary.csv"), index=False)

        return stain_nuclei, stain_cell, orig_im, df_PG_sum


class CytofImageTiff(CytofImage):
    """ CytofImage for Tiff images, inherit from Cytofimage
    """

    def __init__(self, image, slide="", roi="", filename=""):
        self.image = image

        self.markers = None  # markers
        self.labels = None  # labels
        self.slide = slide
        self.roi = roi
        self.filename = filename

        self.channels = None  # ["{}({})".format(marker, label) for (marker, label) in zip(self.markers, self.labels)]

    def set_channels(self, markers, labels):
        self.markers = markers
        self.labels = labels
        self.channels = ["{}({})".format(marker, label) for (marker, label) in zip(self.markers, self.labels)]

    def check_channels(self, channels=None, xlim=None, ylim=None, ncols=5, vis_q=0.9, colorbar=False):
        """
        xlim = a list of 2 numbers indicating the ylimits to show image (default=None)
        ylim = a list of 2 numbers indicating the ylimits to show image (default=None)
        ncols = number of subplots per row (default=5)
        vis_q = percentile q used to normalize image before visualization  (default=0.9)
        """
        if channels is not None:
            if not all([cl in self.channels for cl in channels]):
                print("At least one of the channels not available, visualizing all channels instead!")
                channels = None
        if channels is None:  # if no desired channels specified, check all channels
            channels = self.channels
        if len(channels) <= ncols:
            ax_nrow = 1
            ax_ncol = len(channels)
        else:
            ax_ncol = ncols
            ax_nrow = int(np.ceil(len(channels) / ncols))
        fig, axes = plt.subplots(ax_nrow, ax_ncol, figsize=(3 * ax_ncol, 3 * ax_nrow))
        if ax_nrow == 1:
            axes = np.array([axes])
            if ax_ncol == 1:
                axes = np.expand_dims(axes, axis=1)
        for i, _ in enumerate(channels):
            _ax_nrow = int(np.floor(i / ax_ncol))
            _ax_ncol = i % ax_ncol
            _i = self.channels.index(_)
            image = self.image[..., _i]
            percentile_q = np.quantile(image, vis_q) if np.quantile(image, vis_q) != 0 else 1
            image = np.clip(image / percentile_q, 0, 1)
            axes[_ax_nrow, _ax_ncol].set_title(_)
            if xlim is not None:
                image = image[:, xlim[0]:xlim[1]]
            if ylim is not None:
                image = image[ylim[0]:ylim[1], :]
            im = axes[_ax_nrow, _ax_ncol].imshow(image, cmap="gray")
            if colorbar:
                fig.colorbar(im, ax=axes[_ax_nrow, _ax_ncol])
        plt.tight_layout()
        plt.show()

    def remove_special_channels(self, channels):
        for channel in channels:
            if channel not in self.channels:
                print("Channel {} not available, escaping...".format(channel))
                continue
            idx = self.channels.index(channel)
            self.channels.pop(idx)
            self.markers.pop(idx)
            self.labels.pop(idx)
            self.image = np.delete(self.image, idx, axis=2)

            if hasattr(self, "df"):
                self.df.drop(columns=channel, inplace=True)

    def define_special_channels(self, channels_dict, q=0.95, overwrite=False):
        channels_rm = []
        for new_name, old_names in channels_dict.items():
            if new_name in self.channels and (not overwrite):
                print("Warning: {} is already available, skipping...".format(new_name))
                continue
            if new_name in self.channels and overwrite:
                print("Warning: {} is already available, overwriting...".format(new_name))
                idx = self.channels.index(new_name)
                self.image = np.delete(self.image, idx, axis=2)
                self.channels.pop(idx)
            if len(old_names) == 0:
                continue

            old_nms = []
            for i, old_name in enumerate(old_names):
                if old_name not in self.channels:
                    warnings.warn('{} is not available!'.format(old_name['marker_name']))
                    continue
                old_nms.append(old_name)
            print("Defining channel '{}' by summing up channels: {}.".format(new_name, ', '.join(old_nms)))

            if len(old_nms) > 0:
                channels_rm += old_nms
                for i, old_name in enumerate(old_nms):
                    _i = self.channels.index(old_name)
                    _image = self.image[..., _i]
                    percentile_q = np.quantile(_image, q) if np.quantile(_image, q) != 0 else 1
                    _image = np.clip(_image / percentile_q, 0, 1)
                    if i == 0:
                        image = _image
                    else:
                        image += _image
                self.image = np.dstack([self.image, image[:, :, None]])
                self.channels.append(new_name)

        return channels_rm