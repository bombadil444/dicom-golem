import numpy as np
import pydicom as dicom
import os
import copy
import matplotlib.pyplot as plt
import logging
from glob import glob
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from time import time
from scipy.linalg import norm

logger = logging.getLogger('main')
fh = logging.FileHandler('python.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

data_path = Path("data/4.000000-24533")
output_path = working_path = Path("output/")

def execute():
    cluster = True
    sanity_check = False
    clusters = 5

    patient = load_scan(data_path)
    imgs = get_pixels_hu(patient)

    segmented_lungs = segment_lung_mask(imgs, fill_lung_structures=False)
    segmented_lungs_fill = segment_lung_mask(imgs, fill_lung_structures=True)
    internal_structures = segmented_lungs_fill - segmented_lungs

    # isolate lung from chest
    copied_pixels = copy.deepcopy(imgs)
    for i, mask in enumerate(segmented_lungs_fill): 
        get_high_vals = mask == 0
        copied_pixels[i][get_high_vals] = 0
    seg_lung_pixels = copied_pixels

    # sanity check
    if sanity_check:
        plt.imshow(seg_lung_pixels[50], cmap=plt.cm.bone)
        plt.imshow(internal_structures[2], cmap='jet', alpha=0.7)
        plt.show()

    # cluster slices
    selected_slices = seg_lung_pixels[50:55]
    if cluster:
        try:
            gk_clustered_imgs = np.load(output_path / "clusters.npy").astype(np.float64)
        except:
            # gk_clustered_imgs = np.array([gk_segment(x, clusters=clusters) for x in selected_slices])
            cluster_array = []
            for i, sl in enumerate(selected_slices):
                print(f"Processing slice: {i}")
                cluster_array.append(gk_segment(sl, clusters=clusters))
                plt.imshow(np.array(cluster_array)[i], cmap=plt.cm.bone)
                plt.savefig(output_path / f"test_{i}")

            gk_clustered_imgs = np.array(cluster_array)
            np.save(output_path / "clusters.npy", gk_clustered_imgs)

        v, f = make_mesh(gk_clustered_imgs, None)
        plt_3d(v, f)


#      
# Loop over the image files and store everything into a list.
# 
def load_scan(path):
    slices = [dicom.read_file(Path(path) / slicepath) for slicepath in os.listdir(path)]
    slices.sort(key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def make_mesh(image, threshold=-300, step_size=1):
    logger.debug("Transposing surface")
    p = image.transpose(2,1,0)
    
    logger.debug("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold) 
    return verts, faces


def plt_3d(verts, faces):
    logger.debug("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=1)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_zlim(0, 200)
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.savefig(output_path / "test2")


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def min_label_volume(im):
    vals, counts = np.unique(im, return_counts=True)
    return vals[np.argmin(counts)]


def segment_lung_mask(image, fill_lung_structures=True):
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image >= -700, dtype=np.int8)+1
    labels = measure.label(binary_image)
 
    # Pick the pixel in the very corner to determine which label is air.
    # Improvement: Pick multiple background labels from around the patient
    # More resistant to “trays” on which the patient lays cutting the air around the person in half
    background_label = labels[0,0,0]
 
    # Fill the air around the person
    binary_image[background_label == labels] = 2
 
    # Method of filling the lung structures (that is superior to 
    # something like morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
 
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
 
    # Remove other air pockets inside body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image


class GK:
    def __init__(self, n_clusters=4, max_iter=100, m=2, error=1e-6):
        super().__init__()
        self.u, self.centers, self.f = None, None, None
        self.clusters_count = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error


    def fit(self, z):
        N = z.shape[0]
        C = self.clusters_count
        centers = []
        u = np.random.dirichlet(np.ones(N), size=C)
        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()
            centers = self.next_centers(z, u)
            f = self._covariance(z, centers, u)
            dist = self._distance(z, centers, f)
            u = self.next_u(dist)
            iteration += 1
            # Stopping rule
            if norm(u - u2) < self.error:
                break
        self.f = f
        self.u = u
        self.centers = centers
        return centers


    def next_centers(self, z, u):
        um = u ** self.m
        return ((um @ z).T / um.sum(axis=1)).T


    def _covariance(self, z, v, u):
        um = u ** self.m

        denominator = um.sum(axis=1).reshape(-1, 1, 1)
        temp = np.expand_dims(z.reshape(z.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        temp = np.matmul(temp, temp.transpose((0, 1, 3, 2)))
        numerator = um.transpose().reshape(um.shape[1], um.shape[0], 1, 1) * temp
        numerator = numerator.sum(0)

        return numerator / denominator


    def _distance(self, z, v, f):
        dif = np.expand_dims(z.reshape(z.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        determ = np.power(np.linalg.det(f), 1 / self.m)
        det_time_inv = determ.reshape(-1, 1, 1) * np.linalg.pinv(f)
        temp = np.matmul(dif.transpose((0, 1, 3, 2)), det_time_inv)
        output = np.matmul(temp, dif).squeeze().T
        return np.fmax(output, 1e-8)


    def next_u(self, d):
        power = float(1 / (self.m - 1))
        d = d.transpose()
        denominator_ = d.reshape((d.shape[0], 1, -1)).repeat(d.shape[-1], axis=1)
        denominator_ = np.power(d[:, None, :] / denominator_.transpose((0, 2, 1)), power)
        denominator_ = 1 / denominator_.sum(1)
        denominator_ = denominator_.transpose()

        return denominator_


    def predict(self, z):
        if len(z.shape) == 1:
            z = np.expand_dims(z, axis=0)

        dist = self._distance(z, self.centers, self.f)
        if len(dist.shape) == 1:
            dist = np.expand_dims(dist, axis=0)

        u = self.next_u(dist)
        return np.argmax(u, axis=0)


def gk_segment(img, clusters=2):
    # expand dims of binary image (1 channel in z axis)
    new_img = np.expand_dims(img, axis=2)
    # reshape
    x, y, z = new_img.shape
    new_img = new_img.reshape(x * y, z)
    # segment using GK clustering
    algorithm = GK(n_clusters=clusters)
    cluster_centers = algorithm.fit(new_img)
    output = algorithm.predict(new_img)
    segments = cluster_centers[output].astype(np.int32).reshape(x, y)
    # get cluster that takes up least space (nodules / airway)
    min_label = min_label_volume(segments)
    segments[np.where(segments != min_label)] = 0
    segments[np.where(segments == min_label)] = 1
    return segments

if __name__ == "__main__":
    execute()
