import numpy as np
import pydicom as dicom
import os
import matplotlib.pyplot as plt
import logging
from glob import glob
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('python.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

data_path = Path("/golem/resource/C4KC-KiTS/KiTS-00000/06-29-2003-threephaseabdomen-41748/test/")
output_path = working_path = Path("/golem/output/")
g = list(data_path.glob('**/*.dcm'))

# Print out the first 5 file names to verify we're in the right folder.
logger.debug("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
for path in g[:5]:
    logger.debug(path.resolve())


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

id=0
patient = load_scan(data_path)
imgs = get_pixels_hu(patient)

np.save(output_path / f"fullimages_{id}.npy", imgs)

file_used=output_path / f"fullimages_{id}.npy"
imgs_to_process = np.load(file_used).astype(np.float64) 

plt.hist(imgs_to_process.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.savefig(output_path / "test")


def make_mesh(image, threshold=-300, step_size=1):
    logger.debug("Transposing surface")
    p = image.transpose(2,1,0)
    
    logger.debug("Calculating surface")
    verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True) 
    return verts, faces


def plt_3d(verts, faces):
    logger.debug("Drawing")
    x,y,z = zip(*verts) 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
    face_color = [1, 1, 0.9]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, max(x))
    ax.set_ylim(0, max(y))
    ax.set_zlim(0, max(z))
    ax.set_facecolor((0.7, 0.7, 0.7))
    plt.savefig(output_path / "test2")


v, f = make_mesh(imgs_to_process, 350)
plt_3d(v, f)
