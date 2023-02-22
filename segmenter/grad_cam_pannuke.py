from pytorch_grad_cam import GradCAM #, HiResCAM, ScoreCAM, GradCAMPlusPlus,
# AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
from segm.model.factory import load_model
import torch
import torch.functional as F
import numpy as np
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

MODEL_PATH = "/home/labs/testing/class54/project_code/segmenter/first_train_128epochs/checkpoint.pth"
IMAGES_PATH = "/home/labs/testing/class54/project_data/Fold_1/images"
NUM_OF_IMAGES = 1
model = load_model(MODEL_PATH,map_location=torch.device('cpu'))[0]
model = model.eval()
target_layers = [model.decoder.mask_norm]
print("mask norm: ", model.decoder.mask_norm)
#input_tensor = np.zeros((1,256,256,3))
#for i in range(NUM_OF_IMAGES):
input_tensor = np.load(f"{IMAGES_PATH}/f1_1_img.npy")
print("input image shape:",input_tensor.shape)
input_tensor = np.float32(input_tensor) / 255
input_tensor = preprocess_image(input_tensor)
print("input_tensor shape: ",input_tensor.shape)
sem_classes = ['Neoplastic','Non-Neoplastic Epithelial','Inflammatory','Connective','Dead',
'Background']
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
category = sem_class_to_idx["Neoplastic"]

output = model(input_tensor)
print("output shape: ",output.shape)
print("output range: ",torch.unique(output))
normalized_masks = torch.nn.functional.softmax(output, dim=1)
print("normalized_masks shape: ", normalized_masks.shape)
print("normalized_mask range: ", torch.unique(normalized_masks))
output_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().numpy()
print("car mask shape: ",output_mask.shape)
print("car mask range: ",np.unique(output_mask))
output_mask = np.float32(output_mask == category)
print(output_mask.shape)
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
    print(grayscale_cam)
    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(input_tensor, grayscale_cam,
                                      use_rgb=True)
    Image.fromarray(cam_image)

