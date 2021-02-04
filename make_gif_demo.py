import imageio
import glob
from tqdm import tqdm

demo_name = 'demo2'
filenames = sorted(glob.glob('./{}/*.jpg'.format(demo_name)))
with imageio.get_writer('./{}.gif'.format(demo_name), mode='I', duration=0.15) as writer:
    for filename in tqdm(filenames):
        image = imageio.imread(filename)
        writer.append_data(image)
