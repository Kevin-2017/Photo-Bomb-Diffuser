from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.refinement import (
    refine_predict,
)  # refining the predictions

from omegaconf import OmegaConf
import yaml, torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import cv2
import albumentations as A


def resolve_torchpath(arg):
    if arg == "TORCH_HOME":
        return "C:\\Users\\Aditya Ojha\\Music\\CV_Graduate\\lama\\"
    else:
        print(arg)
        return None


def save_tensor_image(tensor, filename):
    """
    Saves the image in the tensor using plt.
    We assume that the tensor has the size [1,3 (# channels), height, width]
    The function will remove the first dim, and reorder the dims for saving
    """
    img = tensor.detach().numpy()
    if len(tensor.size()) == 4:  # ok, some inputs are [3 (# channels), height, width]
        # so we only do this squeeze to remove the 1 only if it's there
        img = np.squeeze(img, axis=0)
    img = np.einsum("cxy->xyc", img)
    if img.shape[-1] == 1:
        # grayscale image
        img = np.reshape(img, img.shape[:-1])
        plt.imshow(img)
    else:
        plt.imshow(img)
    print(filename, "->", img.shape)
    plt.savefig(filename + ".jpg")


def import_image_to_tensor(filename, grayscale=False):
    """
    Take the image at filename, makes it a tensor.
    Reshapes it to width = 512, height = 512
    Returns the tensor
    """
    img = plt.imread(filename)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # change to RGB
    if (
        "normal" in filename
    ):  # the test image I used needed to have its axises switched for it to work.
        img = np.transpose(img, (1, 0, 2))
    print("Img size is ", img.shape)
    # if img.shape[1] > 512 and img.shape[2] > 512:
    #     print("Resizing Img to 512,512")
    # img = transforms.Resize((512, 512))(img)
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # )
    out_size = 512
    transform_img = A.Compose(
        [
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5
            ),
            A.ToFloat(),
        ]
    )
    transform_mask = A.Compose([A.PadIfNeeded(min_height=out_size, min_width=out_size)])
    if grayscale:  # its a mask
        img = transform_mask(image=img)["image"]  # resize it
        img = np.mean(img, axis=-1)  # to greyscale
        # img = transforms.Grayscale()(img)
    else:  # its an image
        img = transform_img(image=img)["image"]  # transforms will pad the image
    # print(img.keys())
    # input()
    img = transforms.ToTensor()(img)
    return img


### These are the transforms applied to datasets by default:
# from saicinpainting.training.data.datasets
# import albumentations as A
# transform = A.Compose([
#             A.RandomScale(scale_limit=0.2),  # +/- 20%
#             A.PadIfNeeded(min_height=out_size, min_width=out_size),
#             A.RandomCrop(height=out_size, width=out_size),
#             A.HorizontalFlip(),
#             A.CLAHE(),
#             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
#             A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
#             A.ToFloat()
#         ])


if __name__ == "__main__":
    # train_config is a OmegaConf yaml object. So we need to load the yaml file first
    OmegaConf.register_new_resolver("env", resolve_torchpath)
    # there is an error where the yaml file says "env:TORCH_HOME"
    # but omegaconf doesn't know what that means so it throws a
    # "unsupported interpolation type" error
    # so we make a resolver that will return the torch hub directory
    # were model weights are stored.
    # with open("big-lama/config.yaml", "r") as f:
    #     train_config = OmegaConf.create(yaml.safe_load(f))
    # model = load_checkpoint(
    #     train_config=train_config,
    #     path="big-lama/models/best.ckpt",
    #     strict=False,
    #     map_location="cpu",
    # )

    img_prefix = "boat"
    test_img = import_image_to_tensor(
        "testing_imgs/" + img_prefix + ".jpg"
    )  # get image
    # input(test_img.shape)
    test_img = torch.unsqueeze(test_img, 0)  # so it has a batch size of 1
    test_mask = import_image_to_tensor(
        "testing_masks/" + img_prefix + "_mask.jpg", grayscale=True
    )  # get mask
    test_mask = torch.unsqueeze(test_mask, 0)  # make the dims 1,1,H,W

    print("test image size is ", test_img.size())
    print("test mask size is ", test_mask.size())
    print(torch.max(test_mask))
    plt.imshow(test_mask.numpy()[0][0])
    plt.show()
    test_batch = {
        "image": test_img,
        "mask": test_mask,
        # "unpad_to_size": ({0: torch.Tensor([1024])}, {0: torch.Tensor([1024])}),
    }  # package data

    # TODO Add the refinement and see if that works
    # got params for refine_predict from configs/prediction/default
    # model.training = False
    # model.add_noise_kwargs = False
    # model.concat_mask = (
    #     True  # need to set these three params so refinement function can work
    # )
    # img_pred = refine_predict(test_batch,model,
    #                           gpu_ids=["cpu"],
    #                           modulo=2,
    #                           n_iters=15,
    #                           lr=0.002,
    #                           min_side=512,
    #                           max_scales=3,
    #                           px_budget=1800000) # now img_pred isn't a dict; its just the image
    # inpainted image of size (1,3,H,W)
    # Any image will be resized to satisfy height*width <= px_budget

    #### Normal Prediction code
    # img_pred = model(test_batch)  # get prediction

    # coe when saving refined images
    # print(img_pred.size())
    # save_tensor_image(img_pred[0],"example_output")

    # inverse normalization:
    # invTrans = transforms.Compose(
    #     [
    #         transforms.Normalize(
    #             mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    #         ),
    #         transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
    #     ]
    # )
    # img_pred["predicted_image"] = invTrans(
    #     img_pred["predicted_image"]
    # )  # reverse the normalization
    # below is saving code when not using refinement
    # keys are ['image', 'mask', 'predicted_image', 'inpainted', 'mask_for_losses']
    # print(img_pred["inpainted"].size())
    # save_tensor_image(img_pred["inpainted"], "example_output/lama_output_"+img_prefix)
    # save_tensor_image(img_pred["mask"], "example_mask/lama_output_"+img_prefix)
    # save_tensor_image(
    #     img_pred["predicted_image"], "example_outputs/lama_output_" + img_prefix
    # )
    # save_tensor_image(img_pred["image"], "example_image")
    print("Test File")
