# Photo-Bomb-Diffuser
By: Kevin Wang & Adi Ojha

## How to run project code
**LaMa:**
Go to [Other_Repos/lama](Other_Repos/lama). Download requirements from [requirement.txt](Other_Repos/lama/requirements.txt) (using pip install). Run the [lama_output.ipynb](Other_Repos/lama/lama_output.ipynb) to see results. The 3rd and 4th cells control what images/masks are using for inpainting.

**Various Segmentation Models:**
Go to [Pose2Seg_trans](Pose2Seg_trans). Download requirements from [requirement.txt](Pose2Seg_trans/requirements.txt) (using pip install). Run [segmentation.ipynb](Pose2Seg_trans/segmentation.ipynb) to see SAM and transformer Pose2Seg output. You could download our Pose2Seg_trans check point [here](https://drive.google.com/file/d/1nt5izn_d86Xq5n_guGVS_VsBZFiP0dLd/view?usp=share_link). If you go to the SAM branch, you can see the code used to generate the normal Pose2Seg output.

**Stable Diffusion**
The Stabe diffusion model we used and fine tuned on is https://huggingface.co/runwayml/stable-diffusion-v1-5. The UI for the stable diffusion is base on [AUTOMATIC1111's repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui). To run our fine-tuned [model](https://drive.google.com/file/d/1c-Fixmmiy1rqTcuakVZeFd28Phpf8voK/view?usp=share_link) following the instruction of the original model and replace it with our model. 

## Project Abstract
There is nothing more annoying than seeing something you don't want to see. As we take photos of loved ones, friends, or ourselves we create albums of fond memories; the good times in our lives. But we are not alone in this world; our fellow humans inhabit not only our space but also these prized possessions. Removing these so called photo bombers usually is done when a photo is taken; you wait for a clearing and then quickly snap-a-pic! What if you didn't have to wait; what if you could remove unwanted people from your background in an instant! No photo-shop required. In this project, the authors aim to make a "Magic Eraser" that can automatically detect individuals and remove them from photographs. We aim to achieve this by training and deploying a instance segmentation and neural in-painting deep learning model.
