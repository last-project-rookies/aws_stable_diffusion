import os
import sys
import time
import importlib
import signal
import re
import warnings
import uuid
import datetime
import boto3
import torch
import gc
import json
import random
import Config

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from packaging import version

import logging
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

from modules import paths, timer, import_hook, errors

startup_timer = timer.Timer()

import torch
startup_timer.record("import torch")

import gradio
startup_timer.record("import gradio")

import ldm.modules.encoders.modules
startup_timer.record("import ldm")

from modules import extra_networks, ui_extra_networks_checkpoints
from modules import extra_networks_hypernet, ui_extra_networks_hypernets, ui_extra_networks_textual_inversion
from modules.call_queue import wrap_queued_call, queue_lock, wrap_gradio_gpu_call

# Truncate version number of nightly/local build of PyTorch to not cause exceptions with CodeFormer or Safetensors
if ".dev" in torch.__version__ or "+git" in torch.__version__:
    torch.__long_version__ = torch.__version__
    torch.__version__ = re.search(r'[\d.]+[\d]', torch.__version__).group(0)

from modules import shared, devices, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks
import modules.codeformer_model as codeformer
import modules.face_restoration
import modules.gfpgan_model as gfpgan
import modules.img2img

import modules.lowvram
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.sd_vae
import modules.txt2img
import modules.script_callbacks
import modules.textual_inversion.textual_inversion
import modules.progress

import modules.ui
from modules import modelloader
from modules.shared import cmd_opts
import modules.hypernetworks.hypernetwork

startup_timer.record("other imports")

def txt_input():
    id_task = "task(3mxmuksh8blozaz)" 
    prompt = "<lora:aespakarina:1>, 8k, raw photo, (1girl), (solo), (masterpiece), (best quality), ultra high res, (realistic, photo-realistic:1.4), ultra detailed, physically-based rendering, detailed beautiful eyes and detailed face, looking at viewer, upper body, (white background:2), (slanted corners of the eyes:2), arched eyebrows, strong bridge of the nose, upturned semi-vertical nose tip, slender, sharp jawline, chic" # , (happy face:1.8), (sad face:2.0), (angry face:1.6)
    negative_prompt = "(worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.331), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), (more than 2 nipples:1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), bad hands, missing fingers, extra digit, (futa:1.1), bad body, NG_DeepNegative_V1_75T, pubic hair, glans, refraction, diffusion, diffraction, nude, open mouth, teeth, username, watermark, flower, hat"
    prompt_styles = []
    steps = 30
    sampler_index = 16
    restore_faces = False
    tiling = False
    n_iter = 1 
    batch_size = 1  
    cfg_scale = 9.0 
    seed = -1.0
    subseed = -1.0 
    subseed_strength = 0 
    seed_resize_from_h = 0 
    seed_resize_from_w = 0 
    seed_enable_extras = False 
    height = 512
    width = 512 
    enable_hr = False
    denoising_strength = 0.7
    hr_scale = 2.0 
    hr_upscaler = "Latent"
    hr_second_pass_steps =0  
    hr_resize_x =0  
    hr_resize_y =0  
    override_settings_texts = [] 
   
    args = 0, False, False, 'positive', 'comma', 0, False, False, '', 1, '', 0, '', 0, '', True, False, False, False, 0
    modules.txt2img.txt2img(id_task, prompt, negative_prompt, prompt_styles, steps, sampler_index, restore_faces, tiling, n_iter, batch_size, cfg_scale, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras, height, width, enable_hr, denoising_strength, hr_scale, hr_upscaler, hr_second_pass_steps, hr_resize_x, hr_resize_y, override_settings_texts, *args)
    return {"message": "Hello World"}

def img_input(img, seed, prompt):

    id_task = 'task(3mxmuksh8blozaz)'
    mode = 0
    prompt = prompt
    negative_prompt = '(EasyNegative:1.0), ng_deepnegative_v1_75t, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), manboobs, backlight, (ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), (poorly drawn hands:1.331), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), (more than 2 nipples:1.331), (missing arms:1.331), (extra legs:1.331), bad hands, missing hands, missing fingers, mutated fingers, mutated hands, missing arms, missing legs, mutated legs, (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), extra digit, (futa:1.1), bad body, pubic hair, glans, refraction, diffusion, diffraction, nude, open mouth, teeth, username, watermark, flower, hat'
    prompt_styles = []
    init_img = img
    sketch = None
    init_img_with_mask = None
    inpaint_color_sketch = None
    inpaint_color_sketch_orig = None
    init_img_inpaint = None
    init_mask_inpaint = None
    steps = 30
    sampler_index = 15
    mask_blur = 4
    mask_alpha = 0
    inpainting_fill = 1
    restore_faces = True
    tiling = False
    n_iter = 1
    batch_size = 1
    cfg_scale = 9
    image_cfg_scale = 2.0
    denoising_strength = 0.5
    seed = seed
    subseed = -1.0
    subseed_strength = 0
    seed_resize_from_h = 0
    seed_resize_from_w = 0
    seed_enable_extras = False
    selected_scale_tab = 0
    height = 512
    width = 512
    scale_by = 1
    resize_mode = 0
    inpaint_full_res = 0
    inpaint_full_res_padding = 35
    inpainting_mask_invert = 0
    img2img_batch_input_dir = None
    img2img_batch_output_dir = None
    img2img_batch_inpaint_mask_dir = None
    override_settings_texts = []
    args = (9, False, 1.6, 0.97, 0.4, 0, 20, 0, 12, '', True, False, False, False, 512, False, True, False, 0, '<ul>\n<li><code>CFG Scale</code> should be 2 or lower.</li>\n</ul>\n', True, True, '', '', True, 50, True, 1, 0, False, 4, 0.5, 'Linear', 'None', '<p style="margin-bottom:0.75em">Recommended settings: Sampling Steps: 80-100, Sampler: Euler a, Denoising strength: 0.8</p>', 128, 8, ['left', 'right', 'up', 'down'], 1, 0.05, 128, 4, 0, ['left', 'right', 'up', 'down'], False, False, 'positive', 'comma', 0, False, False, '', '<p style="margin-bottom:0.75em">Will upscale the image by the selected scale factor; use width and height sliders to set tile size</p>', 64, 0, 2, 7, '', [], 0, '', [], 0, '', [], True, False, False, False, 0, '<p style="margin-bottom:0.75em">Recommended settings: Use from inpaint tab, inpaint at full res ON, denoise <0.5</p>', 'segm/mmdet_dd-person_mask2former.pth [1c8dbe8d]', 35, 4, 0, 0, False, 'A&B', '<br>', 'segm/mmdet_dd-person_mask2former.pth [1c8dbe8d]', 35, 4, 0, 0, 4, 0.4, True, 35, 1.4, 0.9, 0.5, 0, 20, 0, 12, '', True, False, False, False, 512, False, True)
    modules.img2img.img2img(id_task, mode, prompt, negative_prompt, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, steps, sampler_index, mask_blur, mask_alpha, inpainting_fill, restore_faces, tiling, n_iter, batch_size, cfg_scale, image_cfg_scale, denoising_strength, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras, selected_scale_tab, height, width, scale_by, resize_mode, inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, override_settings_texts, *args)
    return {"message": "Hello World"}

if cmd_opts.server_name:
    server_name = cmd_opts.server_name
else:
    server_name = "0.0.0.0" if cmd_opts.listen else None


def check_versions():
    if shared.cmd_opts.skip_version_check:
        return

    expected_torch_version = "1.13.1"

    if version.parse(torch.__version__) < version.parse(expected_torch_version):
        errors.print_error_explanation(f"""
You are running torch {torch.__version__}.
The program is tested to work with torch {expected_torch_version}.
To reinstall the desired version, run with commandline flag --reinstall-torch.
Beware that this will cause a lot of large files to be downloaded, as well as
there are reports of issues with training tab on the latest version.

Use --skip-version-check commandline argument to disable this check.
        """.strip())

    expected_xformers_version = "0.0.16rc425"
    if shared.xformers_available:
        import xformers

        if version.parse(xformers.__version__) < version.parse(expected_xformers_version):
            errors.print_error_explanation(f"""
You are running xformers {xformers.__version__}.
The program is tested to work with xformers {expected_xformers_version}.
To reinstall the desired version, run with commandline flag --reinstall-xformers.

Use --skip-version-check commandline argument to disable this check.
            """.strip())


def initialize():
    check_versions()

    extensions.list_extensions()
    localization.list_localizations(cmd_opts.localizations_dir)
    startup_timer.record("list extensions")

    if cmd_opts.ui_debug_mode:
        shared.sd_upscalers = upscaler.UpscalerLanczos().scalers
        modules.scripts.load_scripts()
        return

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    startup_timer.record("list SD models")

    codeformer.setup_model(cmd_opts.codeformer_models_path)
    startup_timer.record("setup codeformer")

    gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    startup_timer.record("setup gfpgan")

    modelloader.list_builtin_upscalers()
    startup_timer.record("list builtin upscalers")

    modules.scripts.load_scripts()
    startup_timer.record("load scripts")

    modelloader.load_upscalers()
    startup_timer.record("load upscalers")

    modules.sd_vae.refresh_vae_list()
    startup_timer.record("refresh VAE")

    modules.textual_inversion.textual_inversion.list_textual_inversion_templates()
    startup_timer.record("refresh textual inversion templates")

    try:
        modules.sd_models.load_model()
    except Exception as e:
        errors.display(e, "loading stable diffusion model")
        print("", file=sys.stderr)
        print("Stable diffusion model failed to load, exiting", file=sys.stderr)
        exit(1)
    startup_timer.record("load SD checkpoint")

    shared.opts.data["sd_model_checkpoint"] = shared.sd_model.sd_checkpoint_info.title

    shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights()))
    shared.opts.onchange("sd_vae", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("sd_vae_as_default", wrap_queued_call(lambda: modules.sd_vae.reload_vae_weights()), call=False)
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    startup_timer.record("opts onchange")

    shared.reload_hypernetworks()
    startup_timer.record("reload hypernets")

    ui_extra_networks.intialize()
    ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
    ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
    ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

    extra_networks.initialize()
    extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())
    startup_timer.record("extra networks")

    if cmd_opts.tls_keyfile is not None and cmd_opts.tls_keyfile is not None:

        try:
            if not os.path.exists(cmd_opts.tls_keyfile):
                print("Invalid path to TLS keyfile given")
            if not os.path.exists(cmd_opts.tls_certfile):
                print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
        except TypeError:
            cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
            print("TLS setup invalid, running webui without TLS")
        else:
            print("Running with TLS")
        startup_timer.record("TLS")

    # make the program just exit at ctrl+c without waiting for anything
    def sigint_handler(sig, frame):
        print(f'Interrupted with signal {sig} in {frame}')
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)


def setup_middleware(app):
    app.middleware_stack = None # reset current middleware to allow modifying user provided list
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if cmd_opts.cors_allow_origins and cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cmd_opts.cors_allow_origins.split(','), allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    elif cmd_opts.cors_allow_origins_regex:
        app.add_middleware(CORSMiddleware, allow_origin_regex=cmd_opts.cors_allow_origins_regex, allow_methods=['*'], allow_credentials=True, allow_headers=['*'])
    app.build_middleware_stack() # rebuild middleware stack on-the-fly


def create_api(app):
    from modules.api.api import Api
    api = Api(app, queue_lock)
    return api


def wait_on_server(demo=None):
    while 1:
        time.sleep(0.5)
        if shared.state.need_restart:
            shared.state.need_restart = False
            time.sleep(0.5)
            demo.close()
            time.sleep(0.5)
            break


def api_only():
    initialize()

    app = FastAPI()
    setup_middleware(app)
    api = create_api(app)

    modules.script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(server_name="0.0.0.0" if cmd_opts.listen else "127.0.0.1", port=cmd_opts.port if cmd_opts.port else 7861)


def webui():
    launch_api = cmd_opts.api
    initialize()

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        modules.script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = modules.ui.create_ui();shared.demo.queue(concurrency_count=999999,status_update_rate=0.1)
        startup_timer.record("create ui")

        if cmd_opts.gradio_queue:
            shared.demo.queue(64)

        gradio_auth_creds = []
        if cmd_opts.gradio_auth:
            gradio_auth_creds += [x.strip() for x in cmd_opts.gradio_auth.strip('"').replace('\n', '').split(',') if x.strip()]
        if cmd_opts.gradio_auth_path:
            with open(cmd_opts.gradio_auth_path, 'r', encoding="utf8") as file:
                for line in file.readlines():
                    gradio_auth_creds += [x.strip() for x in line.split(',') if x.strip()]

        app, local_url, share_url = shared.demo.launch(
            share=cmd_opts.share,
            server_name=server_name,
            server_port=cmd_opts.port,
            ssl_keyfile=cmd_opts.tls_keyfile,
            ssl_certfile=cmd_opts.tls_certfile,
            debug=cmd_opts.gradio_debug,
            auth=[tuple(cred.split(':')) for cred in gradio_auth_creds] if gradio_auth_creds else None,
            inbrowser=cmd_opts.autolaunch,
            prevent_thread_lock=True
        )
        from pydantic import BaseModel
        import io
        from PIL import Image
        import base64
        import gc

        class Base64Request(BaseModel):
            base64_file: str
            user_id: str
            face_id: str
            sex: str
            
        class Base64Request2(BaseModel):
            user_id: str
        
        class Base64Request3(BaseModel):
            dump = ''
            
        class Base64Response(BaseModel):
            response_code: str
            response_message: str
        
        def use_content(content, seed, prompt, user_id):
            
            directory = './input'
            
            decode_content = base64.b64decode(content)
            
            if not os.path.exists(directory):
              os.makedirs(directory)
            
            user_id = user_id.split('/')
            
            with open(f'./input/{user_id[-1]}.jpg','wb') as image:
                image.write(decode_content)
                image.close()
                image = Image.open(f'./input/{user_id[-1]}.jpg')
                
            img_input(image, seed, prompt)
            
            gc.collect()
            
            return 'img return'
       
        def handle_upload_img(f, output_dir, img_emotion):
          s3_client = boto3.client('s3', aws_access_key_id=Config.ACCESS_KEY_ID, aws_secret_access_key=Config.ACCESS_SECRET_KEY)
          s3_client.upload_file(f, Config.BUCKET_NAME, output_dir + '/' + img_emotion + '.jpg')
          return img_emotion + '.jpg'

        @app.post("/img2img")
        async def use_base64file(request: Base64Request):
            global img_seed, face_id, cex, user_id
            file_content = request.base64_file
            user_id = request.user_id
            face_id = request.face_id
            cex = request.sex
            cat_prompt = '(slanted corners of the eyes:1.5), arched eyebrows, strong bridge of the nose, upturned semi-vertical nose tip, slender, sharp jawline, chic, '
            dog_prompt = 'gentle eyes, (droopy corners of the eyes:1.5), rounded tip of the nose, smooth jawline, plump cheeks, smile, '
            img_seed = random.randrange(1, 9999999)
            
            if cex == 'man':
                img_prompt = '<lora:oppav3:0.9>, 8k, raw photo, (1boy:1.5), (solo:1.7), male focus, (best quality), (masterpiece), HDR, (high resolution), (best quality), ultra high res, (intricate details), (cinematic light), (realistic, photo-realistic:1.4), ultra detailed, physically-based rendering, portrait, ultra high res, detailed beautiful eyes and detailed face, blemish-free face, both eyes balanced, looking at viewer, (face portrait:1.0), '
                if face_id == 'cat':
                    img_prompt = img_prompt + cat_prompt + 'upper body, head, (white background:2), '
                else :
                    img_prompt = img_prompt + dog_prompt + 'upper body, head, (white background:2), '
            
            else :
                img_prompt = '<lora:koreanDollLikeness_v15:0.4>, 8k, raw photo, (1girl:1.5), (solo:1.7), (masterpiece), HDR, high-definition, (best quality), ultra high res, (realistic, photo-realistic:1.4), ultra detailed, physically-based rendering, detailed beautiful eyes and detailed face, kpop make up, (face portrait:1.0), '
                if face_id == 'cat':
                    img_prompt = img_prompt + cat_prompt + 'upper body, head, looking at viewer, blemish-free face, both eyes balanced, (white background:2), '
                else :
                    img_prompt = img_prompt + dog_prompt + 'upper body, head, looking at viewer, blemish-free face, both eyes balanced, (white background:2), '
            
            base64_file = file_content.encode('utf-8')
            
            use_content(content=base64_file, seed=img_seed, prompt=img_prompt + '(happy face:1.52)', user_id=user_id)
            
            output_dir = f'./outputs/img2img-images/{(datetime.date.today().isoformat())}'
            file_list = os.listdir(output_dir)
            file_list.sort()
            filename = output_dir + '/' + file_list[-1]
            
            base64_content = base64.b64encode(open(filename, "rb").read())
             
            return json.dumps({'base_data':str(base64_content).split("'")[1]})
            
        @app.post("/emotion2img")
        async def use_base64file2(request: Base64Request2):
            user_id = request.user_id
            cat_prompt = '(slanted corners of the eyes:1.5), arched eyebrows, strong bridge of the nose, upturned semi-vertical nose tip, slender, sharp jawline, chic, '
            dog_prompt = 'gentle eyes, (droopy corners of the eyes:1.5), rounded tip of the nose, smooth jawline, plump cheeks, smile, '
            img_emotions = ['sad', 'angry']
            img_etc_emotion = ['(sad face:2.0)', '(angry face:1.6)']
            
            user_id_sp = user_id.split('/')[-1]
            
            image = Image.open(f'./input/{user_id_sp}.jpg')
            
            if cex == 'man':
                img_prompt = '<lora:oppav3:0.9>, 8k, raw photo, (1boy:1.5), (solo:1.7), male focus, (best quality), (masterpiece), HDR, (high resolution), (best quality), ultra high res, (intricate details), (cinematic light), (realistic, photo-realistic:1.4), ultra detailed, physically-based rendering, portrait, ultra high res, detailed beautiful eyes and detailed face, blemish-free face, both eyes balanced, looking at viewer, (face portrait:1.0), '
                if face_id == 'cat':
                    img_prompt = img_prompt + cat_prompt + 'upper body, head, (white background:2), '
                else :
                    img_prompt = img_prompt + dog_prompt + 'upper body, head, (white background:2), '
            
            else :
                img_prompt = '<lora:koreanDollLikeness_v15:0.4>, 8k, raw photo, (1girl:1.5), (solo:1.7), (masterpiece), HDR, high-definition, (best quality), ultra high res, (realistic, photo-realistic:1.4), ultra detailed, physically-based rendering, detailed beautiful eyes and detailed face, kpop make up, (face portrait:1.0), '
                if face_id == 'cat':
                    img_prompt = img_prompt + cat_prompt + 'upper body, head, looking at viewer, blemish-free face, both eyes balanced, (white background:2), '
                else :
                    img_prompt = img_prompt + dog_prompt + 'upper body, head, looking at viewer, blemish-free face, both eyes balanced, (white background:2), '

            for idx, emotion in enumerate(img_etc_emotion):
                
                img_input(image, img_seed, img_prompt + emotion)
                
                output_dir = f'./outputs/img2img-images/{(datetime.date.today().isoformat())}'
                file_list = os.listdir(output_dir)
                file_list.sort()
                filename = output_dir + '/' + file_list[-1]
                
                handle_upload_img(filename, user_id, img_emotions[idx])
            
            for f in os.listdir(output_dir):
                os.remove(os.path.join(output_dir, f))

            os.rmdir(output_dir)
            
            for f in os.listdir('./input'):
                os.remove(os.path.join('./input', f))
            
            return 'img return'
            
        @app.post("/reset2img")
        async def use_base64file3():
            global img_seed
            cat_prompt = '(slanted corners of the eyes:1.5), arched eyebrows, strong bridge of the nose, upturned semi-vertical nose tip, slender, sharp jawline, chic, '
            dog_prompt = 'gentle eyes, (droopy corners of the eyes:1.5), rounded tip of the nose, smooth jawline, plump cheeks, smile, '
            img_seed = random.randrange(1, 9999999)
            
            user_id_sp = user_id.split('/')[-1]
            
            image = Image.open(f'./input/{user_id_sp}.jpg')
            
            if cex == 'man':
                img_prompt = '<lora:oppav3:0.9>, 8k, raw photo, (1boy:1.5), (solo:1.7), male focus, (best quality), (masterpiece), HDR, (high resolution), (best quality), ultra high res, (intricate details), (cinematic light), (realistic, photo-realistic:1.4), ultra detailed, physically-based rendering, portrait, ultra high res, detailed beautiful eyes and detailed face, blemish-free face, both eyes balanced, looking at viewer, (face portrait:1.0), '
                if face_id == 'cat':
                    img_prompt = img_prompt + cat_prompt + 'upper body, head, (white background:2), '
                else :
                    img_prompt = img_prompt + dog_prompt + 'upper body, head, (white background:2), '
            
            else :
                img_prompt = '<lora:koreanDollLikeness_v15:0.4>, 8k, raw photo, (1girl:1.5), (solo:1.7), (masterpiece), HDR, high-definition, (best quality), ultra high res, (realistic, photo-realistic:1.4), ultra detailed, physically-based rendering, detailed beautiful eyes and detailed face, kpop make up, (face portrait:1.0), '
                if face_id == 'cat':
                    img_prompt = img_prompt + cat_prompt + 'upper body, head, looking at viewer, blemish-free face, both eyes balanced, (white background:2), '
                else :
                    img_prompt = img_prompt + dog_prompt + 'upper body, head, looking at viewer, blemish-free face, both eyes balanced, (white background:2), '
            
            filenames = []
            
            img_input(image, img_seed, img_prompt + '(happy face:1.5)')
                
            output_dir = f'./outputs/img2img-images/{(datetime.date.today().isoformat())}'
            file_list = os.listdir(output_dir)
            file_list.sort()
            filename = output_dir + '/' + file_list[-1]
            
            base64_content = base64.b64encode(open(filename, "rb").read())
             
            return json.dumps({'base_data':str(base64_content).split("'")[1]})
        
        # after initial launch, disable --autolaunch for subsequent restarts
        cmd_opts.autolaunch = False

        startup_timer.record("gradio launch")

        # gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
        # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
        # running web ui and do whatever the attacker wants, including installing an extension and
        # running its code. We disable this here. Suggested by RyotaK.
        app.user_middleware = [x for x in app.user_middleware if x.cls.__name__ != 'CORSMiddleware']

        setup_middleware(app)

        modules.progress.setup_progress_api(app)

        if launch_api:
            create_api(app)

        ui_extra_networks.add_pages_to_demo(app)

        modules.script_callbacks.app_started_callback(shared.demo, app)
        startup_timer.record("scripts app_started_callback")

        print(f"Startup time: {startup_timer.summary()}.")

        wait_on_server(shared.demo)
        print('Restarting UI...')

        startup_timer.reset()

        sd_samplers.set_samplers()

        modules.script_callbacks.script_unloaded_callback()
        extensions.list_extensions()
        startup_timer.record("list extensions")

        localization.list_localizations(cmd_opts.localizations_dir)

        modelloader.forbid_loaded_nonbuiltin_upscalers()
        modules.scripts.reload_scripts()
        startup_timer.record("load scripts")

        modules.script_callbacks.model_loaded_callback(shared.sd_model)
        startup_timer.record("model loaded callback")

        modelloader.load_upscalers()
        startup_timer.record("load upscalers")

        for module in [module for name, module in sys.modules.items() if name.startswith("modules.ui")]:
            importlib.reload(module)
        startup_timer.record("reload script modules")

        modules.sd_models.list_models()
        startup_timer.record("list SD models")

        shared.reload_hypernetworks()
        startup_timer.record("reload hypernetworks")

        ui_extra_networks.intialize()
        ui_extra_networks.register_page(ui_extra_networks_textual_inversion.ExtraNetworksPageTextualInversion())
        ui_extra_networks.register_page(ui_extra_networks_hypernets.ExtraNetworksPageHypernetworks())
        ui_extra_networks.register_page(ui_extra_networks_checkpoints.ExtraNetworksPageCheckpoints())

        extra_networks.initialize()
        extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())
        startup_timer.record("initialize extra networks")


if __name__ == "__main__":
    if cmd_opts.nowebui:
        api_only()
    else:
        webui()
