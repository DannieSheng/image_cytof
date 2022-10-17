import itertools
import re
import os
import copy
import pickle as pkl
import skimage
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import cm
from hyperion_segmentation import cytof_nuclei_segmentation, cytof_cell_segmentation, visualize_segmentation


def load_saved_CytofImage(savename):
    cytof_img = pkl.load(open(savename, "rb"))
    return cytof_img

def _save_multi_channel_img(img, savename):
    """
    A helper function to save multi-channel images
    """
    skimage.io.imsave(savename, img)

def _get_colortable(names, title, emptycols=0, sort_names=True):
    """
    Generate the color table for visualization
    reference: https://matplotlib.org/stable/gallery/color/named_colors.html
    """
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    if sort_names:
        names.sort()
    color_pool = dict((n, plt.cm.get_cmap('tab20').colors[i]) for (i, n) in enumerate(names))

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
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, n in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, n, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y - 9), width=swatch_width,
                      height=18, facecolor=color_pool[n], edgecolor='0.7')
        )
    return fig, color_pool


class MultiPlexedImage():
    """ """
    def __init__(self, image, markers):
        self.image = image
        self.markers = markers



class CytofImage():
    def __init__(self, df=None):
        self.df       = df
        self.columns  = None # column names in original cytof data (dataframe)
        self.markers  = None # protein markers
        self.labels   = None # metal isotopes used to tag protein

        self.image    = None
        self.channels = None # channel names correspond to each channel of self.image

        self.features = None
        """self.df_orig = None"""

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

    def check_channels(self, channels=None, xlim=None, ylim=None):
        if channels is not None:
            if not all([cl.lower() in self.channels for cl in channels]):
                print("At least one of the channels not available, visualizing all channels instead!")
                channels = None
        if channels is None:  # if no desired channels specified, check all channels
            channels = self.channels
        nrow = max(self.df['Y'].values) + 1
        ncol = max(self.df['X'].values) + 1
        if len(channels) <= 5:
            ax_nrow = 1
            ax_ncol = len(channels)
            # fig, axes = plt.subplots(ax_nrow, ax_ncol, figsize=(3*ax_ncol, 3))
        else:
            ax_ncol = 5
            ax_nrow = int(np.ceil(len(channels) / 5))
        fig, axes = plt.subplots(ax_nrow, ax_ncol, figsize=(3 * ax_ncol, 3 * ax_nrow))
        if ax_nrow == 1:
            axes = np.array([axes])
            if ax_ncol == 1:
                axes = np.expand_dims(axes, axis=1)
        for i, _ in enumerate(channels):
            _ax_nrow = int(np.floor(i / ax_ncol))
            _ax_ncol = i % ax_ncol
            image = self.df[_].values.reshape(nrow, ncol)
            quantile_99 = np.quantile(image, 0.99) if np.quantile(image, 0.99)!= 0 else 1
            image = np.clip(image / quantile_99, 0, 1)
            axes[_ax_nrow, _ax_ncol].set_title(_)
            if xlim is not None:
                image = image[:, xlim[0]:xlim[1]]
            if ylim is not None:
                image = image[ylim[0]:ylim[1], :]
            im = axes[_ax_nrow, _ax_ncol].imshow(image, cmap="gray")
            fig.colorbar(im, ax=axes[_ax_nrow, _ax_ncol])
        plt.tight_layout()
        plt.show()

    def define_special_channels(self, channels_dict):
        # create a copy of original dataframe
        self.df_orig = self.df.copy()
        '''self.channels = self.markers.copy() if self.channels is None else self.channels'''
        for new_name, old_names in channels_dict.items():
            for i, old_name in enumerate(old_names):
                idx = self.channels.index(old_name)
                self.channels.pop(idx)
                self.markers.pop(idx)
                self.labels.pop(idx)
                if i == 0:
                    self.df[new_name] = self.df[old_name]
                else:
                    self.df[new_name] += self.df[old_name]
                self.df.drop(columns=old_name, inplace=True)
            self.channels.append(new_name)

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
        assert len(channel_ids) <= 6, "No more than 6 channels can be visualized simultaneously!"
        if len(channel_ids) > 3:
            warnings.warn(
                "Visualizing more than 3 channels the same time results in deteriorated visualization. \
                It is not recommended!")

        print(f"Visualizing channels: {', '.join(channel_names)}")
        full_colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow']
        info = [f"{marker} in {c}\n" for (marker, c) in \
                zip([self.channels[i] for i in channel_ids], full_colors[:len(channel_ids)])]
        print(f"Visualizing... \n{''.join(info)}")
        merged_im = np.zeros((self.image.shape[0], self.image.shape[1], 3))
        if quantiles is None:
            quantiles = [np.quantile(self.image[..., _], 0.99) for _ in channel_ids]

        for _ in range(min(len(channel_ids), 3)):
            merged_im[..., _] = np.clip(self.image[..., channel_ids[_]] / quantiles[_], 0, 1) * 255

        chs = [[1, 2], [0, 2], [0, 1]]
        chs_id = 0
        while _ < len(channel_ids) - 1:
            _ += 1
            for j in chs[chs_id]:
                merged_im[..., j] += np.clip(self.image[..., channel_ids[_]] / quantiles[_], 0, 1) * 255  # /2
                merged_im[..., j] = np.clip(merged_im[..., j], 0, 255)
            chs_id += 1
        merged_im = merged_im.astype(np.uint8)
        if visualize:
            plt.imshow(merged_im)
            plt.show()
        return merged_im, quantiles

    def get_seg(self, use_membrane=True, radius=5, show_process=False):
        nuclei_img = self.image[..., self.channels.index('nuclei')]

        if show_process:
            print("Nuclei segmentation...")
        # else:
        #     print("Not showing segmentation process")
        nuclei_seg, color_dict = cytof_nuclei_segmentation(nuclei_img, show_process=show_process)

        membrane_img = self.image[..., self.channels.index('membrane')] \
            if (use_membrane and 'membrane' in self.channels) else None
        if show_process:
            print("Cell segmentation...")
        cell_seg, _ = cytof_cell_segmentation(nuclei_seg, radius, membrane_channel=membrane_img,
                                              show_process=show_process, colors=color_dict)

        self.nuclei_seg = nuclei_seg
        self.cell_seg   = cell_seg
        return nuclei_seg, cell_seg

    def visualize_seg(self, segtype="cell", show=False):
        assert segtype in ["nuclei", "cell"]
        # nuclei in red, membrane in green
        channel_ids = [self.channels.index(_) for _ in ["nuclei", "membrane"]]
        if segtype == "cell":
            seg = self.cell_seg
            '''# membrane in red, nuclei in green
            channel_ids = [self.channels.index(_) for _ in ["membrane", "nuclei"]]'''
        else:
            seg = self.nuclei_seg

        marked_image = visualize_segmentation(self.image, self.channels, seg, channel_ids=channel_ids, show=show)
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

    def calculate_quantiles(self, qs=[75, 99]):
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
            plt.axvline(np.log2(quantiles[q]), label=f"{q}th quantile", c=c)
            print(f"{q}th quantile: {quantiles[q]}")
        plt.xlim(-15, 15)
        plt.xlabel("log2(expression of all markers)")
        plt.legend()
        plt.show()
        # attach quantile dictionary to self
        self.dict_quantiles = quantiles
        # return quantiles

    def feature_quantile_normalization_(self, qs=[75,99]):
        """Normalize all features with given quantiles except for morphology features"""
        self.calculate_quantiles(qs)
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
            im_temp_ = np.clip(im_temp / np.quantile(im_temp, quantile_norm / 100), 0, 1)
            _save_multi_channel_img((im_temp_ * 255).astype(np.uint8), savename)

    def vis_marker_thresholding(self, marker, feature_type="normed", accumul_type="sum",
                                normq=75, vis_quantile_q=0.9, savefig=False, savedir="./", roiname="test"):
        """Visualize a marker's expression compared to the threshold
        Inputs:
            marker         = the desired marker to visualize result
            normq          = the q valued used to normalize feature
            accumul_type   = the aggregation of feature in a cell, either "sum" or "ave". (default="sum")
            scaled         = the flag indicating whether or not using scaled feature. (default=True)
            vis_quantile_q = the quantile value used to normalize the image when visualize. (default=0.9)
        Outputs:
        """
        from hyperion_analysis import visualize_thresholding_outcome
        assert feature_type in ["original", "normed",
                                "scaled"], 'accepted feature types are "original", "normed", "scaled"'
        if feature_type == "original":
            feat_name = ""
        elif feature_type == "normed":
            feat_name = f"_{normq}normed"
        else:
            feat_name = f"_{normq}normed_scaled"

        n_attr = f"df_feature{feat_name}"
        count_attr = f"cell_count{feat_name}_{accumul_type}"

        df_feat  = getattr(self, n_attr)
        df_thres = getattr(self, count_attr)

        thresholds_cell_marker = dict((x, y) for (x, y) in zip(df_thres["feature"], df_thres["threshold"]))

        feat_name = "{}_cell_{}".format([x for x in self.channels if re.search(marker, x)][0], accumul_type)
        if savefig:
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            savepath = os.path.join(savedir, roiname + "_{}_{}.png".format(marker, accumul_type))
        else:
            savepath=None


        stain_nuclei, stain_cell, vis_img = visualize_thresholding_outcome(
            feat=feat_name,
            feature_summary_df=df_feat,
            raw_image=self.image,
            channel_names=self.channels,
            thres=thresholds_cell_marker[feat_name],
            nuclei_seg=self.nuclei_seg,
            cell_seg=self.cell_seg,
            vis_quantile_q=vis_quantile_q,
            savepath=savepath
        )

        return stain_nuclei, stain_cell, vis_img

    def visualize_phenograph(self, name_pheno, channels, dir_save="./", save_vis=False, pheno_summary=False):
        """
        Visualize phenograph clustering
        """
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        orig_im, quantiles = self.visualize_channels(channel_names=channels, visualize=False)
        phenograph = getattr(self, 'phenograph')[name_pheno]
        communities, feat_name = phenograph['phenotypes'], phenograph['feat_name']
        df_feat = getattr(cytof_img, feat_name)

        # create color_pool
        fig_color, color_pool = _get_colortable(names=[n for n in communities.unique()],
                                                title="phenotype colors")

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
        f_roi_ = f_roi.split("/")[-1].replace('_cytof_img.pkl', '')
        dir_roi = os.path.join(params.outdir, "phenograph_{}_{}normed_{}".format(params.feat_comb, params.normq, k),
                               f_roi_)
        if not os.path.exists(dir_roi):
            os.makedirs(dir_roi)

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
            print(f_roi_)
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