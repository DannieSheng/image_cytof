import os
import sys
import time
import pickle as pkl
import random

import io
import base64
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Flask, url_for, flash, render_template, request, send_from_directory, redirect, Response, jsonify
from flask import session as se
from datetime import timedelta
# from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from werkzeug.utils import secure_filename

from cytof.classes import CytofImage, CytofCohort
from cytof.hyperion_preprocess import cytof_read_data_roi
from cytof.utils import show_color_table


WIN = sys.platform.startswith('win')
if WIN:  # if running on Windows system, use 3 ///
    prefix = 'sqlite:///'
else:  # otherwise use 4
    prefix = 'sqlite:////'

#### upload file ####
UPLOAD_FOLDER   = f'{os.getcwd()}/test_data' # test data upload folder
GENERATE_FOLDER = 'static/upload' # directory for generated info

ALLOWED_EXTENSIONS = set(['txt', 'tiff'])


app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'  # equilvalent to app.secret_key = 'dev'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['UPLOADS_DEFAULT_DEST'] = './test_data' # UPLOADS_DEFAULT_DEST = "./test_data"
# app.config['UPLOADED_SINGLE_DEST'] = f'{app.config["UPLOAD_FOLDER"]}/single' # UPLOADED_SINGLE_DEST = "./test_data/single"
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
app.config['GENERATE_FOLDER'] = GENERATE_FOLDER

# if not os.path.exists(app.config['UPLOADED_SINGLE_DEST']):
#     os.makedirs(app.config['UPLOADED_SINGLE_DEST'])

if not os.path.exists(app.config['GENERATE_FOLDER']):
    os.makedirs(app.config['GENERATE_FOLDER'])


def encode_img_data(img_data):
    img_data = io.BytesIO()
    plt.savefig(img_data, format="png")
    img_data.seek(0)
    encoded_img_data = base64.b64encode(img_data.getvalue())
    return encoded_img_data


#### index ####
@app.route('/', methods=["GET", "POST"])
def index():    
    return render_template('index.html')


@app.route('/select', methods=["GET", "POST"])
def handle_select():
    results = {'processed': 'false'}
    if request.method == "POST":
        results = {'processed': 'true'}
    return jsonify(results)


#### single ROI ####
@app.route('/single', methods=["GET", "POST"])
def single():

    if request.method == 'GET':
        return render_template('single.html')
    return redirect(url_for('single')) # return to home page

@app.route('/single_set', methods=["GET", "POST"])
def process_single_roi_set():
    if request.method == "POST":
        data = request.get_json()
        print(data)
    out_dir = data[0]['dir_out']
    se['out_dir'] = out_dir
    se.permanent = True
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        

    results = {'processed': 'true'}
    return jsonify(results)


@app.route('/single_upload', methods=['POST', 'GET'])
def handle_file_upload():
    results = {'processed': 'false'}
    if request.method == "POST":
        data_request = request.get_json()
        # print(data_request)
        file = data_request[0]['filename']
        # print(f"Filename: {file}")
        if file and allowed_file(file):
            filename = secure_filename(file)
            se['filename'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.isfile(se['filename']):
                file.save(se['filename'])
            se.permanent = True
            results['processed'] = 'true'
        else:
            # TODO: error message
            flash('File type not accepted!')
    return jsonify(results)


@app.route('/single_read', methods=["POST"])
def handle_roi_read():
    global cytof_img, channels_define
    channels_define = {}
    results = {'processed': 'false'}
    if request.method == "POST":
        data_request = request.get_json()
        slide = data_request[0]['slide']
        roi   = data_request[0]['roi']
        start_time = time.time()
        cytof_img = cytof_read_data_roi(se.get('filename'), slide, roi)
        print(f"Data loading time: {time.time() - start_time} seconds")# print(se.get('out_dir'), slide, roi) 
        dir_out        = os.path.join(se.get('out_dir'), slide, roi) # specific saving directory
        dir_channelimg = os.path.join(dir_out, "channel_images")
        dir_feature    = os.path.join(dir_out, "feature")
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        if not os.path.exists(dir_channelimg):
            os.makedirs(dir_channelimg)
        if not os.path.exists(dir_feature):
            os.makedirs(dir_feature)
        se['out_diri']        = dir_out
        se['channelimg_diri'] = dir_channelimg
        se['feature_diri']    = dir_feature
        se['slide']           = slide
        se['roi']             = roi
        se.permanent = True
        # flash(f"Saving directory for this image: {dir_out}")

        # get markers
        cytof_img.get_markers()

        # prepsocess
        cytof_img.preprocess()
        results['processed'] = 'true'
        results['markers']   = ', '.join(cytof_img.markers)
        results['channels']  = ', '.join(cytof_img.channels)
        results['savedir']   = dir_out

    return jsonify(results)
    # render_template('single.html', filename=se['filename'], dir_out=se['out_diri'], markers=cytof_img.markers, channels=cytof_img.channels)


@app.route('/single_check', methods=["GET", "POST"])
def handle_roi_check():
    results = {'processed': 'false'}
    if request.method == "POST":
        start_time = time.time()
        fig_check_channels = cytof_img.check_channels(savedir=se.get('channelimg_diri')) #app.config['GENERATE_FOLDER'])
        print(f"Data checking time: {time.time() - start_time} seconds")
        img_data = io.BytesIO()
        plt.savefig(img_data, format="png")
        img_data.seek(0)
        encoded_img_data = base64.b64encode(img_data.getvalue())

        results['processed'] = 'true'
        results['img_data']  = str(encoded_img_data) #f"data:image/png;base64, {encoded_img_data.decode()}"
        savedir = os.path.relpath(se.get('channelimg_diri'))
        results['savedir']   = f"{savedir}/check_channels.png" # f"{app.config['GENERATE_FOLDER']}/check_channels.png"

    return jsonify(results)


@app.route('/single_remove', methods=["GET", "POST"])
def handle_roi_remove():
    results = {'processed': 'false'}
    if request.method == "POST":
        data_request = request.get_json()
        # channels_remove = data_request[0]['selected']
        cytof_img.remove_special_channels(data_request[0]['selected'])
        cytof_img.get_image()

        fig_check_channels = cytof_img.check_channels(savedir=None) # TODO: save with incremented filename se.get('channelimg_diri')) 
        img_data = io.BytesIO()
        plt.savefig(img_data, format="png")
        img_data.seek(0)
        encoded_img_data = base64.b64encode(img_data.getvalue())

        results['img_data']  = str(encoded_img_data)
        results['channels']  = ', '.join(cytof_img.channels)
        results['processed'] = 'true'

    return jsonify(results)


@app.route('/single_define_add', methods=["GET", "POST"])
def handle_roi_define_add():
    results = {'processed': 'false'}
    if request.method == "POST":
        data_request = request.get_json()
        
        channels_define[data_request[0]['newName']] = data_request[0]['oldNames']
    return jsonify(results)

@app.route('/single_define_finish', methods=["GET", "POST"])
def handle_roi_define_finish():
    results = {'processed': 'false'}
    if request.method == "POST":
        channels_rm = cytof_img.define_special_channels(channels_define)
        print(channels_rm)
        cytof_img.remove_special_channels(channels_rm)
        cytof_img.get_image()
        print(cytof_img.image.shape)
        print(len(cytof_img.channels))
        results['channels']  = ', '.join(cytof_img.channels)
        results['processed'] = 'true'
    return jsonify(results)

@app.route('/single_cell_seg', methods=["GET", "POST"])
def handle_cell_seg():
    results = {'processed': 'false'}
    if request.method == "POST":
        data_request = request.get_json()

        start_time = time.time()
        nuclei_seg, cell_seg = cytof_img.get_seg(use_membrane=False, radius=int(data_request[0]['radius']), show_process=False)
        print(f"Cell segmentation time: {time.time() - start_time}")
        
        # visualize nuclei and cells segmentation
        marked_image_nuclei = cytof_img.visualize_seg(segtype="nuclei", show=False)
        marked_image_cell = cytof_img.visualize_seg(segtype="cell", show=False)
        fig, axs = plt.subplots(1,2, figsize=(15,8))
        axs[0].imshow(marked_image_nuclei[0:300, 0:300:])
        axs[1].imshow(marked_image_cell[0:300, 0:300:])
        axs[0].set_title("Nuclei Segmentation")
        axs[1].set_title("Cells Segmentation")
        axs[0].axis('off')
        axs[1].axis('off')

        img_data = io.BytesIO()
        plt.savefig(img_data, format="png")
        img_data.seek(0)
        encoded_img_data = base64.b64encode(img_data.getvalue())
        results['processed'] = 'true'
        results['img_data']  = str(encoded_img_data) 
    return jsonify(results)


@app.route('/single_feat_extract', methods=["GET", "POST"])
def handle_feat_extract():
    global cytof_cohort
    results = {'processed': 'false'}
    if request.method == "POST":
        print(cytof_img.filename)
        start_time = time.time()

        cytof_img.extract_features(filename=cytof_img.filename)
        cytof_img.feature_quantile_normalization(qs=[75]) 
        print(f"Feature extraction time: {time.time() - start_time}")
        # print(cytof_img.dict_quantiles.keys())
        
        # treat as a cohort (for upcoming analyses)
        slide, roi = se.get('slide'), se.get('roi')
        dict_cytof_img = {f"{slide}_{roi}": cytof_img}
        cytof_img.calculate_quantiles(qs=[75])
        cytof_cohort = CytofCohort(cytof_images=dict_cytof_img, dir_out=se.get('out_diri'))
        results['processed'] = 'true'

        # automatically save cytof image and cytof cohort
        # TODO: add button
        cytof_img.save_cytof(savename=os.path.join(se.get('out_diri'), "cytof_img.pkl"))
        cytof_cohort.save_cytof_cohort(savename=os.path.join(se.get('out_diri'), "cytof_cohort.pkl"))
    return jsonify(results)


@app.route('/single_PhenoGraph', methods=["GET", "POST"])
def handle_PG():
    results = {'processed': 'false'}
    if request.method == "POST":
        start_time = time.time() 
        # cytof_cohort.batch_process_feature()
        key_pheno = cytof_cohort.clustering_phenograph()  
        print(f"PhenoGraph clustering time: {time.time() - start_time}") 
        
        ## visualize
        df_feats, commus, cluster_protein_exps, figs, figs_scatter, figs_exps \
        = cytof_cohort.vis_phenograph(key_pheno=key_pheno, level="cohort", save_vis=True, plot_together=True, fig_width=5)
        results['n_cluster'] = cytof_cohort.phenograph[key_pheno]['N']
        # fig = figs['cohort']

        # img_data = io.BytesIO()
        # plt.savefig(img_data, format="png")
        # img_data.seek(0)
        # encoded_img_data = base64.b64encode(img_data.getvalue())
        
        encoded_vis_PG = encode_img_data(io.BytesIO())
        results['img_data']  = str(encoded_vis_PG) #f"data:image/png;base64, {encoded_img_data.decode()}"    

        ## visualize w/ original image
        # slide, roi = se.get('slide'), se.get('roi')
        # cytof_img = cytof_cohort.cytof_images[f"{slide}_{roi}"]
        vis_multi, quantiles, color_pool = \
        cytof_img.visualize_channels(channel_names=["nuclei", random.choice(cytof_img.channels)],#, "CD38(Pr141Di)"], 
            visualize=False)

        # attch PhenoGraph results to individual ROIs
        cytof_cohort.attach_individual_roi_pheno(key_pheno, override=True)

        # PhenoGraph clustering visualization
        stain_nuclei, stain_cell, color_dict = cytof_img.visualize_pheno(key_pheno=key_pheno)
        
        show_color_table(color_dict)
        # img_data = 
        encoded_color_table = encode_img_data(io.BytesIO())

        fig, axs = plt.subplots(1,3, figsize=(15,5))
        axs[0].imshow(vis_multi)
        axs[1].imshow(stain_nuclei)
        axs[2].imshow(stain_cell)
        axs[0].set_title("Original image")
        axs[1].set_title("PhenoGraph clusters on nuclei")
        axs[2].set_title("PhenoGraph clusters on cells")
        for i in range(3):
            axs[i].axis('off')
        # img_data = io.BytesIO()
        encoded_vis_PG_og = encode_img_data(io.BytesIO())
        results['color_table'] = str(encoded_color_table)
        results['PG_og']       = str(encoded_vis_PG_og)

        results['processed'] = 'true'
    return jsonify(results)


@app.route('/single_MarkerPosPre', methods=["GET", "POST"])
def prepare_MarkerPos():
    results = {'processed': 'false'}
    if request.method == "POST":
        channels = [x for x in cytof_img.channels if x != 'nuclei']
        results['channels']  = ', '.join(channels)
        results['markers']   = ', '.join(cytof_img.markers)
        results['processed'] = 'true'
    return jsonify(results)

@app.route('/single_MarkerPos', methods=["GET", "POST"])
def handle_MarkerPos():
    results = {'processed': 'false'}
    if request.method == "POST":
        start_time = time.time()
        cytof_cohort.generate_summary()
        print(f"Marker positive analysis time: {time.time()-start_time}")
        print(data_request[0]['selected'])
        vis_marker  = "CD19" #CD19(Nd142Di)
        vis_channel = cytof_img.raw_channels[cytof_img.raw_markers.index("CD19")]
        stain_nuclei, stain_cell, color_dict = cytof_img.visualize_marker_positive(
            marker=vis_marker,
            feature_type="normed",
            accumul_type="sum",
            normq=75,
            show_boundary=True,
            color_list=[(0,0,1), (0,1,0)], # negative, positive
            color_bound=(0,0,0),
            show_colortable=False)
        vis_channel_names = [vis_channel, "nuclei"]
        pseudoRGB, quantiles, color_pool = cytof_img.visualize_channels(channel_names=vis_channel_names, 
                                        visualize=False)

        # ## adjust brightness of channels
        # # adjust brightness of the red channel
        # pseudoRGB[..., 0] = np.array(np.clip(pseudoRGB[..., 0] / np.quantile(pseudoRGB[..., 0].reshape(-1), 0.95), 
        #                            0, 1) * 255, dtype=np.uint8)
        # # adjust brightness of the green channel
        # pseudoRGB[..., 1] = np.array(np.clip(pseudoRGB[..., 1] * 0.9, 0, 255), dtype=np.uint8)

        r, c, _ = pseudoRGB.shape
        maxr = min(r, 300)
        maxc = min(c, 300)
        fig, axs = plt.subplots(1,3,figsize=(15, 5))
        axs[0].imshow(pseudoRGB[0:maxr, 0:maxc, :])
        axs[0].axis('off')
        axs[1].imshow(stain_nuclei[0:maxr, 0:maxc, :])
        axs[1].axis('off')
        axs[2].imshow(stain_cell[0:maxr, 0:maxc, :])
        axs[2].axis('off')
        encoded_vis = encode_img_data(io.BytesIO())
        results['vis_marker_pos'] = str(encoded_vis)
        results['processed'] = 'true'
    return jsonify(results)


#### single ROI object ####
@app.route('/single_obj', methods=["GET", "POST"])
def single_obj():
    if request.method == 'GET':
        return render_template('single_obj.html')
    return redirect(url_for('single_obj')) # return to home page


@app.route('/single_upload_obj', methods=['POST', 'GET'])
def obj_file_upload():
    results = {'processed': 'false'}
    if request.method == "POST":
        files = request.files.getlist('files[]')

        file = files[0]
        filename = files[0].filename
        if filename and allowed_file(filename, set(['pkl'])):
            filename = secure_filename(filename)
            se['filename_obj'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.isfile(se['filename_obj']):
                file.save(se['filename_obj'])
            se.permanent = True
            results['processed'] = 'true'

        else:
            # TODO: error message
            flash('File type not accepted2!')
    return jsonify(results)


@app.route('/single_read_obj', methods=["POST"])
def obj_read():
    global cytof_img, cytof_cohort
    # global channels_define
    # channels_define = {}
    results = {'processed': 'false'}
    if request.method == "POST":
        start_time = time.time()
        cytof_img = pkl.load(open(se.get('filename_obj'), "rb"))
        print(f"Data loading time: {time.time() - start_time} seconds")
        slide, roi = cytof_img.slide, cytof_img.roi

        dir_out        = os.path.join(se.get('out_dir'), slide, roi) # specific saving directory
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        # dir_channelimg = os.path.join(dir_out, "channel_images")
        # dir_feature    = os.path.join(dir_out, "feature")
        
        # if not os.path.exists(dir_channelimg):
        #     os.makedirs(dir_channelimg)
        # if not os.path.exists(dir_feature):
        #     os.makedirs(dir_feature)
        # se['channelimg_diri'] = dir_channelimg
        # se['feature_diri']    = dir_feature

        se['out_diri']        = dir_out
        se['slide']           = slide
        se['roi']             = roi
        se.permanent = True

        # treat as a cohort (for upcoming analyses)
        dict_cytof_img = {f"{slide}_{roi}": cytof_img}
        cytof_img.calculate_quantiles(qs=[75])
        cytof_cohort = CytofCohort(cytof_images=dict_cytof_img, dir_out=se.get('out_diri'))
        cytof_cohort.batch_process_feature()

        results['processed'] = 'true'
        results['markers']   = ', '.join(cytof_img.markers)
        results['channels']  = ', '.join(cytof_img.channels)
        results['savedir']   = dir_out
    return jsonify(results)


#### cohort ####
@app.route('/cohort', methods=["GET", "POST"])
def cohort():
    return render_template('cohort.html')




# file validation
def allowed_file(filename, allowed_exts=ALLOWED_EXTENSIONS):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in allowed_exts



# send a file from a given directory with send_file()
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    print(app.config['UPLOADS_DEFAULT_DEST'])
    return send_from_directory(app.config['UPLOADS_DEFAULT_DEST'],
                               filename)

if __name__ == '__main__':
    app.run(port=8000, debug=True)
