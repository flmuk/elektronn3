from visualizer import Visualizer
from new_knossos import KnossosLabelsNozip
from lsd import LSDGaussVdtCom 
from elektronn3.data import transforms
from visualizer import Visualizer
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Set density of quiver plot')
parser.add_argument('-s', '--skip',type = int, default = 1, help = 'number of datapoints skipped after each arrow')
args = parser.parse_args()
skip = args.skip

conf_path_raw = "/wholebrain/songbird/j0251/j0251_72_clahe2/mag1/knossos.conf"

conf_path_labels = "/ssdscratch/songbird/j0251/segmentation/j0251_72_seg_20210127_agglo2/knossos.pyk.conf"

local_shape_descriptor = LSDGaussVdtCom()
common_transforms = [
    transforms.Normalize(mean=0, std=255.),
    local_shape_descriptor

]
transform = transforms.Compose(common_transforms + [
])

np.random.seed(1)
loader = KnossosLabelsNozip(conf_path_label = conf_path_labels, conf_path_raw_data = conf_path_raw, patch_shape=(10,200,250),transform=transform, raw_mode="caching")

data = loader[0]
#model_path_old = "/wholebrain/scratch/fkies/e3training/lsd/L1_seed0_SGD/model_best.pt"
model_path_new = "/wholebrain/scratch/fkies/e3training/lsd/L1_seed0_SGD_nocom_cl_persistentworkers_true_modloss_05tb/model_best.pt"
import numpy as np
#np.random.seed(0)
#viz_old = Visualizer(conf_path_labels, conf_path_raw, model_path_old, patch_shape = (70, 150, 200), transform=transform)
np.random.seed(0)
viz_new = Visualizer(conf_path_labels, conf_path_raw, model_path_new, patch_shape = (60, 100,100), transform=transform)

#viz_old.plot_vdt("old_BVDT")
#viz_old.plot_vdt_norm("old_norm_BVDT")
#viz_old.plot_gauss_div("old_gauss_div")
#viz_old.plot_com("old_com")
#viz_old.plot_raw("old_raw")
viz_new.plot_raw("new_raw")
viz_new.plot_vdt_quiver("new_vdt_quiver", skip = skip)
viz_new.plot_all(skip=skip)
nplots = 5
for count in range(nplots):
    viz_new._generate_sample()
    viz_new._make_prediction()
#    #viz_new.plot_vdt(str(count) + "new_BVDT")
#    #viz_new.plot_vdt_norm(str(count) + "new_norm_BVDT")
#    #viz_new.plot_gauss_div(str(count) +"new_gauss_div")
#    ##viz_new.plot_com(str(count) +"new_com")
#    #viz_new.plot_raw(str(count) +"new_raw")
    viz_new.plot_all(str(count), skip = skip)
    viz_new.plot_vdt_quiver("new_vdt_quiver_{}".format(nplots), skip=skip)
    viz_new.plot_raw("new_raw_{}".format(nplots))
