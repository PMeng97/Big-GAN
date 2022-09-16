import uuid

import torch
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncated_noise_sample,
                                       save_as_images, convert_to_images)
import numpy as np
import PIL.Image
import PIL
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')


def tmp_convert_to_images(obj):
    if not isinstance(obj, np.ndarray):
        obj = obj.detach().numpy()

    obj = obj.transpose((0, 2, 3, 1))
    obj = np.clip(((obj + 1) / 2.0) * 256, 0, 255)

    img = []
    for i, out in enumerate(obj):
        out_array = np.asarray(np.uint8(out), dtype=np.uint8)
        img.append(PIL.Image.fromarray(out_array))
    return img

def txt2img(prompt):
    # git lfs install
    # git clone https://huggingface.co/osanseviero/BigGAN-deep-128
    prompt = prompt.replace('+', ' ')
    print(prompt)
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = BigGAN.from_pretrained('biggan-deep-256')
    model = BigGAN.from_pretrained('./BigGAN-deep-128')

    # Prepare a input
    truncation = 0.4
    prompt = 'soap bubble'
    class_vector = one_hot_from_names([prompt], batch_size=len([prompt]))
    noise_vector = truncated_noise_sample(truncation=truncation, batch_size=len([prompt]))

    # All in tensors
    noise_vector = torch.from_numpy(noise_vector)
    class_vector = torch.from_numpy(class_vector)

    # If you have a GPU, put everything on cuda
    noise_vector = noise_vector.to(device)
    class_vector = class_vector.to(device)
    model.to(device)

    # Generate an image
    with torch.no_grad():
        output = model(noise_vector, class_vector, truncation)

    # If you have a GPU put back on CPU
    output = output.to('cpu')
    image = tmp_convert_to_images(output)[0]

    data_id = str(uuid.uuid4())
    img_name = data_id+'_'+('_').join(prompt.split())
    # Modification needed for MongoDB
    image.save(img_name+".png")
    return image

# RUN THE TWO COMMANDS BELOW FIRST TO CACHE
# git lfs install
# git clone https://huggingface.co/CompVis/stable-diffusion-v1-4
# if torch.cuda.is_available():
#     print("@@Predict: Starting generation with gpu")
#     pipe = pipe.to("cuda")
#     pipe.enable_attention_slicing()
#     with torch.autocast("cuda"):
#         image = pipe(prompt).images[0]
# else:
#     print("@@Predict: Starting generation with cpu")
#     pipe = pipe.to("cpu")
#     pipe.enable_attention_slicing()
#     image = pipe(prompt).images[0]
# print('@@Predict: End generation')
