import numpy as np
from scipy import optimize
from scipy import stats
import uncertainties.unumpy as unp
import uncertainties as unc
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
from matplotlib import cm
from scipy.constants import k

class imageave:
    def __init__(self):
        param = {"pixeltomm": 0.107, "m": 109*1.66053873e-27, "confidence_band": 0.95}
        roi = {"xmin":150, "xmax":225, "ymin":100, "ymax":175}
        scale_factor = 6.45*2*4

        filepath = "C:/Users/13128/jmd/pixelfly-python-control/saved_images/"
        motimages = {"fname": "images_20230915.hdf", "gname":"First_Temperature_Measurement_20230915_162119", "subgname":"PulseBlasterUSB_instruction no. 3 (ns)_1000000.0", 'file_name': 'image_000002'}

        img = self.readhdf(filepath, motimages)[roi["xmin"]:roi["xmax"], roi["ymin"]:roi["ymax"]]
        print(type(img))
        img = np.transpose(img)
        plt.close('all')
        plt.xlabel(r'x [$\mu$m]')
        plt.ylabel(r'y [$\mu$m]')
        plt.imshow(img, cmap='viridis', extent=[0,(roi["xmax"] - roi["xmin"])*scale_factor,0,(roi["ymax"] - roi["ymin"])*scale_factor])
        plt.colorbar()
        plt.show()

    def readhdf(self, filepath, hdfinfo):
        with h5py.File(filepath+hdfinfo["fname"], "r") as f:
            image_list = f[hdfinfo["gname"]]
            img = image_list[hdfinfo["subgname"]][hdfinfo["file_name"]][:,:]

        return img

tof = imageave()
