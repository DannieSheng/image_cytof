#!/usr/bin/env python
# coding: utf-8
import os
import glob
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import argparse
import yaml
import pandas as pd

import sys
sys.path.append('../cytof')
from hyperion_preprocess import cytof_read_data

def makelist(string):
    delim = ','
    # return [float(_) for _ in string.split(delim)]
    return [_ for _ in string.split(delim)]


def parse_opt():
    parser = argparse.ArgumentParser('Cytof batch process', add_help=False)
    parser.add_argument('--cohort_file', type=str,
                        help='a txt file with information of all file paths in the cohort')
    parser.add_argument('--params_ROI', type=str,
                        help='a txt file with parameters used to process single ROI previously')
    parser.add_argument('--outdir', type=str, help='directory to save outputs')
    parser.add_argument('--save_channel_images', default=True, type=bool, help='an indicator of whether or not save channel images')
    return parser


def main(args):
    # parameters used when processing single ROI
    params_ROI = yaml.load(open(args.params_ROI, "rb"), Loader=yaml.Loader)
    channel_dict = params_ROI["channel_dict"]


    # name of the batch and saving directory
    batch_name = os.path.basename(args.cohort_file).split('.txt')[0]
    print(batch_name)

    outdir = os.path.join(args.outdir, batch_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    feat_dirs = {}
    feat_dirs['orig'] = os.path.join(outdir, "feature")
    if not os.path.exists(feat_dirs['orig']):
        os.makedirs(feat_dirs['orig'])

    # process batch files
    cohort_files_ = pd.read_csv(args.cohort_file, header=None, index_col=None)[0].tolist()
    cohort_files = []
    for f in cohort_files_:
        if os.path.isfile(f) and f.endswith('.txt'):  # if f is a text file
            cohort_files.append(f)
        else:
            # if f is a directory, find all txt files in all sub-directories with the given root directory
            for root, dirs, files in os.walk(f, topdown=False):
                for name in files:
                    if name.endswith('.txt'):
                        name_ = os.path.join(root, name)
                        if name_ not in cohort_files:
                            cohort_files.append(name_)
    print("Start processing {} files".format(len(cohort_files)))

    cytof_imgs = {}  # a dictionary contain the full file path of all results
    seen = 0
    dfs_scale_params = {}  # key: quantile q; item: features to be scaled
    for f_roi in cohort_files:
        print("\nNow analyzing {}".format(f_roi))
        f_roi_ = os.path.basename(f_roi).split('.txt')[0]
        if args.save_channel_images:
            dir_roi_channel_img = os.path.join(outdir, "channel_images", f_roi_)
            if not os.path.exists(dir_roi_channel_img):
                os.makedirs(dir_roi_channel_img)

        ## 1) Read and preprocess data
        # read data: file name -> dataframe
        cytof_img = cytof_read_data(f_roi)

        # markers used when capturing the image
        cytof_img.get_markers()

        # preprocess: fill missing values with 0.
        cytof_img.preprocess()

        ## (optional): save channel images
        if args.save_channel_images:
            cytof_img.get_image()
            cytof_img.save_channel_images(dir_roi_channel_img)

        ## 2) nuclei & membrane channels and visualization
        cytof_img.define_special_channels(channel_dict)

        #### Dataframe -> raw image
        cytof_img.get_image()

        ## (optional): save channel images
        if args.save_channel_images:
            cytof_img.get_image()
            cytof_img.save_channel_images(dir_roi_channel_img, channels=['nuclei', 'membrane'])

        ## 3) Nuclei and cell segmentation
        nuclei_seg, cell_seg = cytof_img.get_seg(use_membrane=params_ROI["use_membrane"], radius=params_ROI["cell_radius"])

        ## 4) Feature extraction
        cytof_img.extract_features(f_roi)

        # save the original extracted feature
        cytof_img.df_feature.to_csv(os.path.join(feat_dirs['orig'], f"{f_roi_}_feature_summary.csv"), index=False)

        ### 4.1) Log transform and quantile normalization
        cytof_img.feature_quantile_normalization_(qs=params_ROI["normalize_qs"])

        # save the class instance
        pkl.dump(cytof_img, open(os.path.join(outdir, f"{f_roi_}_cytof_img.pkl"), "wb"))
        cytof_imgs[f_roi] = os.path.join(outdir, f"{f_roi_}_cytof_img.pkl")

        if seen == 0:
            for q in cytof_img.dict_quantiles.keys():
                feat_dirs[q] = os.path.join(outdir, f"feature_{q}normed")
                if not os.path.exists(feat_dirs[q]):
                    os.makedirs(feat_dirs[q])

        # calculate scaling parameters
        ## features to be scaled
        s_features = [col for key, features in cytof_img.features.items() \
                      for f in features \
                      for col in cytof_img.df_feature.columns if col.startswith(f)]

        ## loop over quantiles
        for q, quantile in cytof_img.dict_quantiles.items():
            n_attr = f"df_feature_{q}normed"
            df_normed = getattr(cytof_img, n_attr)
            # save the normalized features to csv
            df_normed.to_csv(os.path.join(feat_dirs[q], f"{f_roi_}_feature_summary.csv"), index=False)
            if seen == 0:
                dfs_scale_params[q] = df_normed[s_features]
            else:
                dfs_scale_params[q] = dfs_scale_params[q].append(df_normed[s_features], ignore_index=True)
        seen += 1

    for q in cytof_img.dict_quantiles.keys():
        df_scale_params = dfs_scale_params[q].mean().to_frame(name="mean").transpose()
        df_scale_params = df_scale_params.append(dfs_scale_params[q].std().to_frame(name="std").transpose(),
                                                 ignore_index=True)
        df_scale_params.to_csv(os.path.join(outdir, f"{batch_name}_{q}normed_scale_params.csv"), index=False)

    '''yaml.dump(cytof_imgs, open(os.path.join(outdir, "cytof_imgs.txt"), "w")) '''
    df_temp = pd.DataFrame.from_dict(cytof_imgs, orient="index", columns=['output_file'])
    df_temp.reset_index(inplace=True)
    df_temp.rename(columns={'index': 'input_file'}, inplace=True)
    df_temp.to_csv(os.path.join(outdir, "input_output.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Cytof batch process', parents=[parse_opt()])
    args  = parser.parse_args()
    main(args)
