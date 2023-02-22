from pytorch_grad_cam import GradCAM #, HiResCAM, ScoreCAM, GradCAMPlusPlus,
# AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from segm.model.factory import load_model
import numpy as np
from PIL import Image


MODEL_PATH = "/Final Project"
IMAGES_PATH = "/Final Project"
NUM_OF_IMAGES = 1

model = load_model(MODEL_PATH)
model = model.eval()
target_layers = [model.encoder.blocks[-1].norm1]

input_tensor = np.zeros((NUM_OF_IMAGES,256,256,3))
for i in range(NUM_OF_IMAGES):
    input_tensor[i] = np.load(f"{IMAGES_PATH}/f1_1_img.npy")

sem_classes = ['Neoplastic','Non-Neoplastic Epithelial','Inflammatory','Connective','Dead',
'Background']
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
category = sem_class_to_idx["Neoplastic"]

output = model(input_tensor)
output_mask = np.float32(output[0] == category)

# Construct the CAM object once, and then re-use it on many images:
# cam = GradCAM(model=model, target_layers=target_layers)#, use_cuda=args.use_cuda)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
with GradCAM(model=model, target_layers=target_layers) as cam:

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.


    targets = [SemanticSegmentationTarget(category,output_mask)]

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(input_tensor[0], grayscale_cam,
                                      use_rgb=True)
Image.fromarray(cam_image)
