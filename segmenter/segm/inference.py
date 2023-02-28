import click
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F

import segm.utils.torch as ptu

from segm.data.utils import STATS
#from segm.data.ade20k import ADE20K_CATS_PATH
from segm.data.pannuke import PANNUKE_CATEGORIES_PATH
from segm.data.utils import dataset_cat_description, seg_to_rgb

from segm.model.factory import load_model
from segm.model.utils import inference


@click.command()
@click.option("--model-path", type=str)
@click.option("--input-dir", "-i", type=str, help="folder with input images")
@click.option("--output-dir", "-o", type=str, help="folder with output images")
@click.option("--gpu/--cpu", default=True, is_flag=True)
def main(model_path, input_dir, output_dir, gpu):
    ptu.set_gpu_mode(gpu)

    model_dir = Path(model_path).parent
    model, variant = load_model(model_path)
    model.to(ptu.device)

    normalization_name = variant["dataset_kwargs"]["normalization"]
    normalization = STATS[normalization_name]
    cat_names, cat_colors = dataset_cat_description(PANNUKE_CATEGORIES_PATH)

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    list_dir = list(input_dir.iterdir())
    for filename in tqdm(list_dir, ncols=80):
        if filename.suffix == '.npy':
          img_npy = np.load(filename)
          pil_im = Image.fromarray(img_npy.astype(np.uint8))
        else:
          pil_im = Image.open(filename).copy()
        im = F.pil_to_tensor(pil_im).float() / 255
        im = F.normalize(im, normalization["mean"], normalization["std"])
        im = im.to(ptu.device).unsqueeze(0)

        im_meta = dict(flip=False)
        logits = inference(
            model,
            [im],
            [im_meta],
            ori_shape=im.shape[2:4],
            window_size=variant["inference_kwargs"]["window_size"],
            window_stride=variant["inference_kwargs"]["window_stride"],
            batch_size=2,
        )
        seg_map = logits.argmax(0, keepdim=True)
        seg_rgb = seg_to_rgb(seg_map, cat_colors)
        seg_rgb = (255 * seg_rgb.cpu().numpy()).astype(np.uint8)
        pil_seg = Image.fromarray(seg_rgb[0])

        pil_blend = Image.blend(pil_im, pil_seg, 0.5).convert("RGB")
        save_filename = filename.name[:-4]+'.png' # replace .npy with .png
        pil_blend.save(output_dir / save_filename)

if __name__ == "__main__":
    main()
