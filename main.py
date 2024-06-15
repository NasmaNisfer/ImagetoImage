import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import Optional

import spaces
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
import ipown
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
#import gradio as gr
import cv2

#base_model_path = "SG161222/RealVisXL_V3.0"
#base_model_path = "cagliostrolab/animagine-xl-3.0"
#base_model_path = "playgroundai/playground-v2-1024px-aesthetic"
base_model_path = "frankjoshua/juggernautXL_v8Rundiffusion"  

try: from pip._internal.operations import freeze
except ImportError: # pip < 10.0
    from pip.operations import freeze

pkgs = freeze.freeze()
for pkg in pkgs: print(pkg)

ip_ckpt = hf_hub_download(repo_id="h94/IP-Adapter-FaceID", filename="ip-adapter-faceid_sdxl.bin", repo_type="model")
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
# vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    add_watermarker=False,
    use_safetensors=True,
    variant="fp16"
    # vae=vae,
    #feature_extractor=safety_feature_extractor,
    #safety_checker=safety_checker
)

ip_model = ipown.IPAdapterFaceIDXL(pipe, ip_ckpt, device)


api = FastAPI()
lock = asyncio.Lock()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import math
#@spaces.GPU(enable_queue=True)

@api.post("/get_image2image")
async def get_image2image(
                          gender: str,
                          image_height: int,
                          image_width: int,
                          output_image_count: int,
                          image1: UploadFile = File(...),
                          image2: UploadFile = File(None),
                          image3: UploadFile = File(None),
                          image4: UploadFile = File(None),
                          image5: UploadFile = File(None),
                          image6: UploadFile = File(None),
                          image7: UploadFile = File(None),
                          image8: UploadFile = File(None),
                         ): 
    async with lock:
        file = image1
        
        file_name = file.filename
        print(file_name)
        with open(file_name, "wb") as file_object:
            file_object.write(file.file.read())
            
        gender = gender.lower()
        output_image_count = math.ceil((output_image_count/4))
        total_image_list = []
        for no in range(output_image_count):
            if gender=="male":
               prompt = "make business man photos"
            if gender=="female":
               prompt = "make business woman photos"
            
            
                    
            negative_prompt = "(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"
            face_strength = 7.5
            likeness_strength = 0.7
            
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Start the process
            pipe.to(device)
            app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(512, 512))
            
            faceid_all_embeds = []
            try:
                face = cv2.imread(file_name)
                faces = app.get(face)
                faceid_embed = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
                faceid_all_embeds.append(faceid_embed)
            
                average_embedding = torch.mean(torch.stack(faceid_all_embeds, dim=0), dim=0)
                
                total_negative_prompt = negative_prompt
                
                print("Generating SDXL")
                image = ip_model.generate(
                    prompt=prompt, negative_prompt=total_negative_prompt, faceid_embeds=average_embedding,
                    scale=likeness_strength, width=image_width, height=image_height, guidance_scale=face_strength, num_inference_steps=30
                )
            
                print(image)
    
                total_image_list.extend(image)

            except Exception as e:
                   print(e)
                   return {"message":str(e)}

        print(total_image_list)

        import base64
            
        def convert_image_to_base64(filepath):
                # Open the image file in binary mode
            with open(filepath, 'rb') as image_file:
                # Read the file and encode it into Base64
                encoded_string = base64.b64encode(image_file.read())
                return encoded_string.decode('utf-8')
        
        number_for_images = 1
        base64_list = []
        for gen_images in total_image_list:
            gen_images.save('img'+str(number_for_images)+'.jpg')
            base64_list.append(convert_image_to_base64('img'+str(number_for_images)+'.jpg'))
            number_for_images = number_for_images + 1
            #image[1].save('img2.jpg')
            #image[2].save('img3.jpg')
            #image[3].save('img4.jpg')
        
            # Example usage:
            #img1 = convert_image_to_base64('img'+str(number_for_images)+'.jpg')
            #img2 = convert_image_to_base64('img2.jpg')
            #img3 = convert_image_to_base64('img3.jpg')
            #img4 = convert_image_to_base64('img4.jpg')
            
        return {"images":base64_list}
        

    


