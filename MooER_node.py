# !/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import subprocess
import sys
from huggingface_hub import snapshot_download
import time
import torchaudio
from torchaudio.transforms import Resample
from .mooer.datasets.speech_processor import *
from .mooer.configs import asr_config
from .mooer.models import mooer_model
from .mooer.utils.utils import *
import folder_paths
import platform

MAX_SEED = np.iinfo(np.int32).max
current_path = os.path.dirname(os.path.abspath(__file__))
node_path_dir = os.path.dirname(current_path)
comfy_file_path = os.path.dirname(node_path_dir)
diff_current_path = os.path.join(folder_paths.models_dir, "diffusers")

model_config = asr_config.ModelConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

device = str(get_device())
dtype = torch.float16

ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None and platform.system() in ['Linux', 'Darwin']:
    try:
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0:
            ffmpeg_path = result.stdout.strip()
            print(f"FFmpeg is installed at: {ffmpeg_path}")
        else:
            print("FFmpeg is not installed. Please download ffmpeg-static and export to FFMPEG_PATH.")
            print("For example: export FFMPEG_PATH=/any_path/ffmpeg-4.4-amd64-static")
    except Exception as e:
        pass

if ffmpeg_path is not None and ffmpeg_path not in os.getenv('PATH'):
    print("Adding FFMPEG_PATH to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

mooer_file_dir= os.path.join(folder_paths.input_directory,"mooer_files")

if not os.path.exists(mooer_file_dir):
    os.makedirs(mooer_file_dir)

def find_directories(base_path):
    directories = []
    for root, dirs, files in os.walk(base_path):
        for name in dirs:
            directories.append(name)
    return directories

audio_dir_list = find_directories(mooer_file_dir)

if audio_dir_list:
    audio_dir_list = ["none"] + audio_dir_list
else:
    audio_dir_list = ["none", ]
    
def get_local_path(model_path):
    path = os.path.join(diff_current_path, model_path)
    model_path = os.path.normpath(path)
    if sys.platform == 'win32':
        model_path = model_path.replace('\\', "/")
    return model_path


def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform == 'win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


dif_paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "configuration.json" in files:
                dif_paths.append(os.path.relpath(root, start=search_path))

if dif_paths:
    dif_paths = ["none"] + [x for x in dif_paths if "Qwen2-7B-Instruct" in x or "MooER-MTL-5K" in x]
else:
    dif_paths = ["none", ]


def instance_path(path, repo):
    if repo == "":
        if path == "none":
            repo = "none"
        else:
            model_path = get_local_path(path)
            repo = get_instance_path(model_path)
    return repo


def process_wav(audio_raw,sample_rate,adapter_downsample_rate,cmvn,tokenizer,prompt_template,prompt_org):
    
    if sample_rate != 16000:
        # resample the data
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        audio_raw = resampler(audio_raw)
    
    if audio_raw.shape[0] > 1:
        # convert to mono
        audio_raw = audio_raw.mean(dim=0, keepdim=True)
    
    audio_raw = audio_raw[0]
    prompt = prompt_template.format(prompt_org)
    audio_mel = compute_fbank(waveform=audio_raw)
    audio_mel = apply_lfr(inputs=audio_mel, lfr_m=7, lfr_n=6)
    audio_mel = apply_cmvn(audio_mel, cmvn=cmvn)
    audio_length = audio_mel.shape[0]
    audio_length = audio_length // adapter_downsample_rate
    audio_pseudo = torch.full((audio_length,), -1)
    prompt_ids = tokenizer.encode(prompt)
    prompt_length = len(prompt_ids)
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.int64)
    example_ids = torch.cat((audio_pseudo, prompt_ids))  # [audio, prompt]
    example_mask = example_ids.ge(-1)
    
    items = {
        "input_ids": example_ids,
        "attention_mask": example_mask,
        "audio_mel": audio_mel,
        "audio_length": audio_length,
        "prompt_length": prompt_length,
    }
    return items

@torch.no_grad()
def process_main(model,tokenizer,cmvn,adapter_downsample_rate,audio_raw,sample_rate,audio_dir,prompt_template,prompt_org,context_scope):
    audio_suffix=["wav","mpa3","flac"]
    if audio_dir!="none":
        real_audio_dir = os.path.join(mooer_file_dir, audio_dir) #  audio list in
        file_list=os.listdir(real_audio_dir) #file list
        if len(file_list)>=1:
            batch_wav_paths=[os.path.join(real_audio_dir,i) for i in file_list if i.rsplit(".")[-1] in audio_suffix ]
            num_batches=len(batch_wav_paths)
            infer_time = []
            ASR_out = []
            AST_out = []
            for i in range(num_batches):
                samples = []
                for wav_path in batch_wav_paths:
                    audio_raw, sample_rate = torchaudio.load(wav_path)
                    samples.append(process_wav(audio_raw, sample_rate, adapter_downsample_rate, cmvn, tokenizer,prompt_template,prompt_org))
                    print(wav_path.rsplit(audio_dir)[-1])
                batch = process_batch(samples, tokenizer=tokenizer)
                for key in batch.keys():
                    batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
                with context_scope(dtype=dtype):
                    ss = time.perf_counter()
                    model_outputs = model.generate(**batch)
                    infer_time.append(time.perf_counter() - ss)
                    logging.info(f"Infer time: {time.perf_counter() - ss}")
                output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False,
                                                           skip_special_tokens=True)
               
                for idx, text in enumerate(output_text):
                    text = text.split('\n')
                    if len(text) == 2:
                        ASR_out.append([text[0].strip()])
                        AST_out.append([text[1].strip()])
                    else:
                        ASR_out.append([text[0].strip()])
            logging.info(sum(infer_time))
        else:
            print(f"No audio fiels in {real_audio_dir}")
            raise "No audio fiels in audio_dir... "
    else:
        items = process_wav(audio_raw, sample_rate, adapter_downsample_rate, cmvn, tokenizer,prompt_template,prompt_org)
        batch = process_batch([items], tokenizer=tokenizer)
        for key in batch.keys():
            batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
        with context_scope(dtype=dtype):
            ss = time.perf_counter()
            model_outputs = model.generate(**batch)
            logging.info(f"Infer time: {time.perf_counter() - ss}")
        output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False,
                                                   skip_special_tokens=True)
        ASR_out=[]
        AST_out = []
        for text in output_text:
            text = text.split('\n')
            if len(text) == 2:
                ASR_out.append([text[0].strip()])
                AST_out.append([text[1].strip()])
            else:
                ASR_out.append([text[0].strip()])
                
    ASR_text=' '.join(ASR_out[0]) if ASR_out else ("need asr or asr_ast.")
    AST_text = ' '.join(AST_out[0]) if AST_out else "need ast or asr_ast."
    return ASR_text,AST_text

class MooER_LoadModel:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Qwen2_repo":("STRING", {"default": "Qwen/Qwen2-7B-Instruct"}),
                "Qwen2_diff":(dif_paths,),
                "mooer_repo": ("STRING", {"default": "mtspeech/MooER-MTL-5K"}),
                "mooer_diff":(dif_paths,),
                "use_torch_musa":("BOOLEAN", {"default": False},),
                "use_modelscope":("BOOLEAN", {"default": False},),
                "encoder_name":(["paraformer","whisper","hubert","paraformer"],),
                "mode_choice":(["ASR_AST","ASR","AST"],),
            }
        }

    RETURN_TYPES = ("MODEL","MODEL","MODEL",)
    RETURN_NAMES = ("model","tokenizer","cmvn",)
    FUNCTION = "main_loader"
    CATEGORY = "MooER"
    
    def instance_path(self,repo,path):
        if repo == "":
            if path == "none":
                raise "need fill repo or local model path.."
            else:
                repo = get_instance_path(get_local_path(path))
        elif repo and path!="none":
            repo = get_instance_path(get_local_path(path))
        return repo

    def main_loader(self,Qwen2_repo,Qwen2_diff,mooer_repo,mooer_diff,use_torch_musa,use_modelscope,encoder_name,mode_choice):
        logger.info("Run on {}".format(device.upper()))
        if use_torch_musa:
            try:
                import torch_musa
            except ImportError as e:
                print(f"You should install torch_musa if you want to run on Moore Threads GPU,{e}")
                
        cache_dir = os.path.join(diff_current_path, "cache")
        # pre models
        qwen2_repo_id=self.instance_path(Qwen2_repo,Qwen2_diff)
        local_llm_dir = os.path.join(diff_current_path, "Qwen/Qwen2-7B-Instruct")
        if os.path.exists(local_llm_dir) and qwen2_repo_id=="Qwen/Qwen2-7B-Instruct":
            raise "fill'Qwen/Qwen2-7B-Instruct' will automatic download and And cover existing models"
         
        if qwen2_repo_id=="Qwen2-7B-Instruct":
            if use_modelscope:
                from modelscope.hub.snapshot_download import snapshot_download as snapshot_download_mo
                print("download from modelscope...")
                qwen2_repo_id=snapshot_download_mo("qwen/qwen2-7b-instruct",local_dir=local_llm_dir, )
            else:
                print("download from huggingface...")
                qwen2_repo_id=snapshot_download("Qwen/Qwen2-7B-Instruct",cache_dir=cache_dir,local_dir=local_llm_dir, local_dir_use_symlinks=False)
                
        mooer_repo_id = self.instance_path(mooer_repo,mooer_diff)
        local_mo_dir = os.path.join(diff_current_path, "mtspeech/MooER-MTL-5K")
        if os.path.exists(local_mo_dir) and mooer_repo_id=="mtspeech/MooER-MTL-5K":
            raise "fill'mtspeech/MooER-MTL-5K' will automatic download and And cover existing models"
        
        if mooer_repo_id=="mtspeech/MooER-MTL-5K":
            if use_modelscope:
                ignore_files = ["model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
                                "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors", ]
                from modelscope.hub.snapshot_download import snapshot_download as snapshot_download_mo
                print("download from modelscope...")
                mooer_repo_id=snapshot_download_mo("MooreThreadsSpeech/MooER-MTL-5K", ignore_file_pattern=ignore_files,
                                     local_dir=local_mo_dir, )
            else:
                print("download from huggingface...")
                mooer_repo_id=snapshot_download("mtspeech/MooER-MTL-5K", cache_dir=cache_dir,local_dir=local_mo_dir,local_dir_use_symlinks=False )
        elif mooer_repo_id=="mtspeech/MooER-MTL-80K":
            if use_modelscope:
                ignore_files = ["model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
                                "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors", ]
                from modelscope.hub.snapshot_download import snapshot_download as snapshot_download_mo
                print("download from modelscope...")
                mooer_repo_id=snapshot_download_mo("MooreThreadsSpeech/MooER-MTL-80K", ignore_file_pattern=ignore_files,
                                     local_dir=local_mo_dir, )
            else:
                print("download from huggingface...")
                mooer_repo_id=snapshot_download("mtspeech/MooER-MTL-80K", cache_dir=cache_dir,local_dir=local_mo_dir,local_dir_use_symlinks=False )
               
        model, tokenizer = mooer_model.init_model(
            model_config,qwen2_repo_id,mooer_repo_id,encoder_name,mode_choice)
        model.to(device)
        model.eval()
        cmvn = load_cmvn(os.path.join(diff_current_path,mooer_repo_id,"paraformer_encoder/am.mvn"))
        return (model,tokenizer,cmvn,)
    
class MooER_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model":("MODEL",),
                "tokenizer":("MODEL",),
                "cmvn":("MODEL",),
                "asr_prompt":("STRING", {"default": "Transcribe speech to text. "}),
                "ast_prompt": ("STRING", {"default": "Translate speech to english text."}),
                "audio_dir": (audio_dir_list,),
                "adapter_downsample_rate": ("INT", {"default": 2, "min": 0, "max": 10}),
                "prompt_key": (["asr","ast",],),
            },
        }
    
    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("ASR_text","AST_text",)
    FUNCTION = "em_main"
    CATEGORY = "MooER"
    
    def em_main(self,audio,model,tokenizer,cmvn,asr_prompt,ast_prompt,audio_dir,adapter_downsample_rate,prompt_key,):
        
        # data process
        PROMPT_TEMPLATE_DICT = {
            'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",}
        PROMPT_DICT = {'asr': asr_prompt,'ast': ast_prompt,}
        prompt_template_key = model_config.get('prompt_template_key', 'qwen')
        prompt_template = PROMPT_TEMPLATE_DICT[prompt_template_key]
        prompt_org = PROMPT_DICT[prompt_key]
        
        context_scope = torch.musa.amp.autocast if 'musa' in device else torch.cuda.amp.autocast
        
        # audio
        audio_raw=audio["waveform"].squeeze(0)
        sample_rate= audio["sample_rate"]

        ASR_text,AST_text=process_main(model,tokenizer,cmvn,adapter_downsample_rate,audio_raw,sample_rate,audio_dir,prompt_template,prompt_org,context_scope)
        torch.cuda.empty_cache()
        return (ASR_text,AST_text,)


NODE_CLASS_MAPPINGS = {
    "MooER_LoadModel":MooER_LoadModel,
    "MooER_Sampler": MooER_Sampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MooER_LoadModel":"MooER_LoadModel",
    "MooER_Sampler": "MooER_Sampler",
}
