import base64
import datetime
import io
import multiprocessing
import re
from typing import AsyncGenerator
from transformers import AutoTokenizer, AutoFeatureExtractor
from PIL import Image
from vllm import LLM, SamplingParams
import time
import torchaudio
import numpy as np
import os
from decord import VideoReader, cpu
import torch
import asyncio
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
import shortuuid
from vllm.utils import random_uuid
import gradio as gr
from collections import deque
from queue import Empty
import cv2
import json
from web_demo.wakeup_and_vad.wakeup_and_vad import WakeupAndVAD


from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.tts.v20190823 import tts_client, models


IMAGE_TOKEN_INDEX = 51000
AUDIO_TOKEN_INDEX = 51001
IMAGE_TOKEN = "<image>"
AUDIO_TOKEN = "<audio>"
VIDEO_TOKEN = "<video>"

httpProfile = HttpProfile()
httpProfile.endpoint = "tts.tencentcloudapi.com"
cred = credential.Credential("", "")
clientProfile = ClientProfile()
clientProfile.httpProfile = httpProfile
client = tts_client.TtsClient(cred, "ap-shanghai", clientProfile)

req = models.TextToVoiceRequest()


def clear_queue(queue):
    while not queue.empty():
        try:
            queue.get_nowait()
        except Empty:
            break

# The following code is used to run an async task in a synchronous way
def run_async_task(task):
    loop = asyncio.get_event_loop()
    if loop.is_running():
        # If the event loop is already running, run the task in the current loop
        return loop.run_until_complete(task)
    else:
        # Else, create a new loop and run the task in it
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(task)
        finally:
            loop.close()


# This is a function to tokenize the prompt with image and audio tokens
def tokenizer_image_audio_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = []
    for chunk in re.split(r'(<audio>|<image>)', prompt):
        if chunk == '<audio>':
            prompt_chunks.append([audio_token_index])
        elif chunk == '<image>':
            prompt_chunks.append([image_token_index])
        else:
            prompt_chunks.append(tokenizer(chunk).input_ids)
    
    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in prompt_chunks:
        if x != [image_token_index] and x != [audio_token_index]:
            input_ids.extend(x[offset:])
        else:
            input_ids.extend(x[:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.LongTensor(input_ids)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids



def load_model(
        llm_id,
        engine_args,
        cuda_devices,
        inputs_queue,
        outputs_queue,
        tts_outputs_queue,
        stop_event,
        other_stop_event,
        worker_ready,
        wait_workers_ready,
        start_event,
        other_start_event,
        start_event_lock,
        interrupt_signal,
        global_history,
        global_history_limit=0,
    ):

    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    multiprocessing.set_start_method('spawn', force=True)
    llm = AsyncLLMEngine.from_engine_args(engine_args)


    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, subfolder="feature_extractor", trust_remote_code=True)

    sampling_params = SamplingParams(temperature=0.001, max_tokens=512, best_of=1, skip_special_tokens=False)

    def _process_inputs(inputs):

        def _process_image(image_path):
            if isinstance(image_path, str):
                assert os.path.exists(image_path), f"Image file {image_path} does not exist."
                return Image.open(image_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
            else:
                assert isinstance(image_path, np.ndarray), "Image must be either a file path or a numpy array."
                return Image.fromarray(image_path).convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)


        def _process_audio(audio_path):
            assert os.path.exists(audio_path), f"Audio file {audio_path} does not exist."
            audio, sr = torchaudio.load(audio_path)
            audio_features = feature_extractor(audio, sampling_rate=sr, return_tensors="pt")["input_features"]
            audio_features = audio_features.squeeze(0)
            return audio_features
        
        def _process_video(video_path, max_frames=4, min_frames=4, s=None, e=None):
            # speed up video decode via decord.

            if s is None or e is None:
                start_time, end_time = None, None
            else:
                start_time = int(s)
                end_time = int(e)
                start_time = max(start_time, 0)
                end_time = max(end_time, 0)
                if start_time > end_time:
                    start_time, end_time = end_time, start_time
                elif start_time == end_time:
                    end_time = start_time + 1

            if os.path.exists(video_path):
                vreader = VideoReader(video_path, ctx=cpu(0))
            else:
                raise FileNotFoundError(f"Video file {video_path} does not exist.")

            fps = vreader.get_avg_fps()
            f_start = 0 if start_time is None else int(start_time * fps)
            f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
            num_frames = f_end - f_start + 1
            
            if num_frames > 0:
                # T x 3 x H x W
                all_pos = list(range(f_start, f_end + 1))
                if len(all_pos) > max_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
                elif len(all_pos) < min_frames:
                    sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
                else:
                    sample_pos = all_pos

                patch_images = [Image.fromarray(f).transpose(Image.FLIP_LEFT_RIGHT) for f in vreader.get_batch(sample_pos).asnumpy()]
                return patch_images

            else:
                print("video path: {} error.".format(video_path))

        if "multi_modal_data" in inputs:

            if "image" in inputs["multi_modal_data"]:
                image_inputs = inputs["multi_modal_data"]["image"]
                if not isinstance(image_inputs, list):
                    image_inputs = [image_inputs]
                inputs["multi_modal_data"]["image"] = [_process_image(f) for f in image_inputs]

                if "prompt" in inputs:
                    assert inputs["prompt"].count(IMAGE_TOKEN) == len(image_inputs), \
                        f"Number of image token {IMAGE_TOKEN} in prompt must match the number of image inputs."
                elif "prompt_token_ids" in inputs:
                    assert inputs["prompt_token_ids"].count(IMAGE_TOKEN_INDEX) == len(image_inputs), \
                        f"Number of image token ids {IMAGE_TOKEN_INDEX} in prompt_token_ids must match the number of image inputs."
                else:
                    raise ValueError("Either 'prompt' or 'prompt_token_ids' must be provided.")

            if "audio" in inputs["multi_modal_data"]:
                audio_inputs = inputs["multi_modal_data"]["audio"]
                if not isinstance(audio_inputs, list):
                    audio_inputs = [audio_inputs]
                inputs["multi_modal_data"]["audio"] = [_process_audio(f) for f in audio_inputs]

                if "prompt" in inputs:
                    assert inputs["prompt"].count(AUDIO_TOKEN) == len(inputs["multi_modal_data"]["audio"]), \
                        f"Number of audio token {AUDIO_TOKEN} in prompt must match the number of audio inputs."
                elif "prompt_token_ids" in inputs:
                    assert inputs["prompt_token_ids"].count(AUDIO_TOKEN_INDEX) == len(inputs["multi_modal_data"]["audio"]), \
                        f"Number of audio token ids {AUDIO_TOKEN_INDEX} in prompt_token_ids must match the number of audio inputs."
                else:
                    raise ValueError("Either 'prompt' or 'prompt_token_ids' must be provided.")

            if "video" in inputs["multi_modal_data"]:
                video_inputs = inputs["multi_modal_data"]["video"]
                if not isinstance(video_inputs, list):
                    video_inputs = [video_inputs]

                assert "prompt" in inputs, "Prompt must be provided when video inputs are provided."
                assert "image" not in inputs["multi_modal_data"], "Image inputs are not supported when video inputs are provided."

                assert inputs["prompt"].count(VIDEO_TOKEN) == 1, "Currently only one video token is supported in prompt."

                assert inputs["prompt"].count(VIDEO_TOKEN) == len(inputs["multi_modal_data"]["video"]), \
                    f"Number of video token {VIDEO_TOKEN} in prompt must match the number of video inputs."
                
                video_frames_inputs = []
                for video_input in video_inputs:
                    video_frames_inputs.extend(_process_video(video_input, max_frames=4, min_frames=4))
                
                inputs["prompt"] = inputs["prompt"].replace(VIDEO_TOKEN, IMAGE_TOKEN * len(video_frames_inputs))
                if "image" not in inputs["multi_modal_data"]:
                    inputs["multi_modal_data"]["image"] = []
                inputs["multi_modal_data"]["image"].extend(video_frames_inputs)

                inputs["multi_modal_data"].pop("video", None)

        return inputs

    def judge_negative(text):
        is_negative = text.startswith('<2>')
        return is_negative
    

    async def stream_results(results_generator) -> AsyncGenerator[bytes, None]:
        previous_text = ""
        async for request_output in results_generator:

            text = request_output.outputs[0].text
            newly_generated_text = text[len(previous_text):]
            previous_text = text
            yield newly_generated_text

    async def collect_results_demo(results_generator):
        async for newly_generated_text in stream_results(results_generator):
            continue
            

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


    worker_ready.set()
    if not isinstance(wait_workers_ready, list):
        wait_workers_ready = [wait_workers_ready]

    while True:
        # Wait for all workers to be ready
        if not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
            continue

        if not inputs_queue.empty():

            with start_event_lock:
                if start_event.is_set():
                    inputs = inputs_queue.get()

                    other_start_event.set()
                    start_event.clear()
                else:
                    continue
            
            inputs = _process_inputs(inputs)
            current_inputs = inputs.copy()
            inputs = merge_current_and_history(
                global_history[-global_history_limit:],
                inputs,
                skip_history_vision=True,
                move_image_token_to_start=True
            )

            print(f"Process {cuda_devices} is processing inputs: {inputs}")

            if "prompt" in inputs:
                # Process multimodal tokens
                inputs["prompt_token_ids"] = tokenizer_image_audio_token(inputs["prompt"], tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX)
            else:
                assert "prompt_token_ids" in inputs, "Either 'prompt' or 'prompt_token_ids' must be provided."
            inputs.pop("prompt", None)

            # print(f"Process {cuda_devices} is about to generate results, prompt: {current_inputs['prompt']}, prompt_token_ids: {inputs['prompt_token_ids']}")
        
            results_generator = llm.generate(
                inputs,
                sampling_params=sampling_params,
                request_id=random_uuid(),
            )

            async def stream_results(results_generator) -> AsyncGenerator[bytes, None]:
                previous_text = ""
                async for request_output in results_generator:

                    text = request_output.outputs[0].text
                    newly_generated_text = text[len(previous_text):]
                    previous_text = text
                    yield newly_generated_text

            async def collect_results(results_generator):
                results = []
                is_first_time_to_work = True
                history_generated_text = ''
                async for newly_generated_text in stream_results(results_generator):
                  
                    # if newly_generated_text.strip() == "":
                    #     continue

                    # newly_generated_text = newly_generated_text.strip()
                    is_negative = judge_negative(newly_generated_text)

                    if not is_negative:
                        history_generated_text += newly_generated_text
                        if is_first_time_to_work:
                            print(f"Process {cuda_devices} is about to interrupt other process")
                            stop_event.clear()
                            other_stop_event.set()
                            clear_queue(outputs_queue)
                            clear_queue(tts_outputs_queue)

                            is_first_time_to_work = False
                            interrupt_signal.value = llm_id

                        if not stop_event.is_set():

                            results.append(newly_generated_text)
                            history_generated_text = history_generated_text.replace('<1> ', '').replace('<1>', '')
                            
                            # print('newly_generated_text',newly_generated_text)
                            if newly_generated_text in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"]:
                                # print('history_generated_text:',history_generated_text)
                                outputs_queue.put({"id": llm_id, "response": history_generated_text})
                                history_generated_text = ''
                        else:
                            print(f"Process {cuda_devices} is interrupted.")
                            break

                    else:
                        print(f"Process {cuda_devices} is generating negative text.")
                        break
                
                
                current_inputs["response"] = "".join(results)
                if not current_inputs["response"] == "":
                    global_history.append(current_inputs)
                return results

            results = loop.run_until_complete(collect_results(results_generator))
            print(f"Process {cuda_devices} has generated results: {''.join(results)}")





def tts_tranform_text(text):
        print(text)
        params = {
            "Text": text,
            "SessionId": "session-1234",
            "Volume": 1,
            "Speed": 0,
            "ProjectId": 0,
            "ModelType": 1,
            "VoiceType": 301009,
            "PrimaryLanguage": 1,
            "SampleRate": 16000,
            "Codec": "wav",
            "EnableSubtitle": True
        }
        req.from_json_string(json.dumps(params))
        resp = client.TextToVoice(req)

        aaa=json.loads(resp.to_json_string())
        base64_audio_data = aaa['Audio']
        audio_data = base64.b64decode(base64_audio_data)

        wav_file = "tmp_audio/"
        if not os.path.exists(wav_file):
            os.makedirs(wav_file)
        tmp_saved_wav_file = wav_file + str(301009) + "_" + str(shortuuid.uuid()) + ".wav"

        with open(tmp_saved_wav_file, "wb") as audio_file:
            audio_file.write(audio_data)

        return tmp_saved_wav_file



def tts_worker(
    inputs_queue,
    outputs_queue,
    worker_ready,
    wait_workers_ready,
):

    def audio_file_to_html(audio_file: str) -> str:
        """
        Convert audio file to HTML audio player.

        Args:
            audio_file: Path to audio file

        Returns:
            audio_player: HTML audio player that auto-plays
        """
        # Read in audio file to audio_bytes
        audio_bytes = io.BytesIO()
        with open(audio_file, "rb") as f:
            audio_bytes.write(f.read())

        # Generate audio player HTML object for autoplay
        audio_bytes.seek(0)
        audio = base64.b64encode(audio_bytes.read()).decode("utf-8")
        audio_player = (
            f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'
        )
        return audio_player


    def remove_uncommon_punctuation(text):
        common_punctuation = ".,!?;:()[]，。！？、：；（） "
        uncommon_punctuation_pattern = rf"[^\w\s{re.escape(common_punctuation)}]"
        cleaned_text = re.sub(uncommon_punctuation_pattern, "", text)

        return cleaned_text
    
    def remove_special_tokens(input_str):
        # Remove special tokens
        special_tokens = ['<1>', '<2>', '<3>', '<unk>', '</s>']
        for token in special_tokens:
            input_str = input_str.replace(token, '')
        return input_str

    def replace_equation(sentence):

        special_notations = {
            "sin": " sine ",
            "cos": " cosine ",
            "tan": " tangent ",
            "cot": " cotangent ",
            "sec": " secant ",
            "csc": " cosecant ",
            "log": " logarithm ",
            "exp": "e^",
            "sqrt": "根号 ",
            "abs": "绝对值 ",
        }
        
        special_operators = {
            "+": "加",
            "-": "减",
            "*": "乘",
            "/": "除",
            "=": "等于",
            '!=': '不等于',
            '>': '大于',
            '<': '小于',
            '>=': '大于等于',
            '<=': '小于等于',
        }

        greek_letters = {
            "α": "alpha ",
            "β": "beta ",
            "γ": "gamma ",
            "δ": "delta ",
            "ε": "epsilon ",
            "ζ": "zeta ",
            "η": "eta ",
            "θ": "theta ",
            "ι": "iota ",
            "κ": "kappa ",
            "λ": "lambda ",
            "μ": "mu ",
            "ν": "nu ",
            "ξ": "xi ",
            "ο": "omicron ",
            "π": "派 ",
            "ρ": "rho ",
            "σ": "sigma ",
            "τ": "tau ",
            "υ": "upsilon ",
            "φ": "phi ",
            "χ": "chi ",
            "ψ": "psi ",
            "ω": "omega "
        }

        sentence = sentence.replace('**', ' ')

        sentence = re.sub(r'(?<![\d)])-(\d+)', r'负\1', sentence)

        for key in special_notations:
            sentence = sentence.replace(key, special_notations[key]) 
        for key in special_operators:
            sentence = sentence.replace(key, special_operators[key])
        for key in greek_letters:
            sentence = sentence.replace(key, greek_letters[key])


        sentence = re.sub(r'\(?(\d+)\)?\((\d+)\)', r'\1乘\2', sentence)
        sentence = re.sub(r'\(?(\w+)\)?\^\(?(\w+)\)?', r'\1的\2次方', sentence)
        
        return sentence

    worker_ready.set()
    if not isinstance(wait_workers_ready, list):
        wait_workers_ready = [wait_workers_ready]

    past_llm_id = 0

    while True:
        # Wait for all workers to be ready
        if not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
            continue

        tts_input_text = ""
        while not inputs_queue.empty():
            time.sleep(0.03)

            stop_at_punc_or_len = False
            response = inputs_queue.get()
            llm_id, newly_generated_text = response["id"], response["response"]

            for character in newly_generated_text:
                
                if  past_llm_id != 0 and past_llm_id != llm_id:
                    # print(f"Past llm id {past_llm_id} is not equal to current llm id {llm_id}, resetting tts input text and putting pause signal")
                    tts_input_text = ""
                    tts_output_queue.put(
                        {
                            "id": llm_id,
                            "response": ("|PAUSE|", None, 0.2)
                        }
                    )
                
                tts_input_text += character

                past_llm_id = llm_id
                # print('tts_input_text',tts_input_text)
                if character in [",", "，", ".", "。", "?", "\n", "？", "!", "！", "、"] and len(tts_input_text) >= 5:
                    stop_at_punc_or_len = True
                    break

            if stop_at_punc_or_len:
                break

        if tts_input_text.strip() == "":
            continue
        
        tts_input_text = remove_special_tokens(tts_input_text)
        tts_input_text = replace_equation(tts_input_text)
        tts_input_text = tts_input_text.lower()

        # print(f"Start to generate audio for: {tts_input_text}, llm id {llm_id}")
        if tts_input_text.strip() == "":
            continue
        audio_file = tts_tranform_text(tts_input_text)
        html = audio_file_to_html(audio_file)

        audio_duration = torchaudio.info(audio_file).num_frames / 24000

        if past_llm_id == 0 or past_llm_id == llm_id:
            outputs_queue.put(
                {
                    "id": llm_id,
                    "response": (tts_input_text, html, audio_duration)
                }
            )
        


def merge_current_and_history(
        global_history,
        current_request,
        skip_history_vision=False,
        move_image_token_to_start=False
    ):

    system_prompts = {
        "video": "system:You are an AI robot and your name is Vita. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the video given by the user, and it is strictly forbidden to answer the question without the content of the video. Please note that you are seeing the video, not the image.</s>\n",
        "image": "system:You are an AI robot and your name is Vita. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the image given by the user, and it is strictly forbidden to answer the question without the content of the image. Please note that you are seeing the image, not the video.</s>\n",
        "audio": "system:You are an AI robot and your name is Vita. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user.</s>\n"
    }
    
    def select_system_prompt(current_request):
        if "multi_modal_data" in current_request:
            if "video" in current_request["multi_modal_data"]:
                return system_prompts["video"]
            elif "image" in current_request["multi_modal_data"]:
                return system_prompts["video"]
            elif "audio" in current_request["multi_modal_data"]:
                return system_prompts["audio"]
        return system_prompts["audio"]

    system_prompt = select_system_prompt(current_request)
    print('current request:',current_request)
    user_prefix = "user:"
    bot_prefix = "bot:"
    eos = "</s>\n"

    if len(global_history) == 0:
        
        current_request["prompt"] = (system_prompt + user_prefix + current_request["prompt"] + eos + bot_prefix).replace('<1> ','<1>').replace('<2> ','<2>')
        return current_request
    
    # Initialize the current prompt and multimodal data
    current_prompt = system_prompt
    current_multi_modal_data = {"image": [], "audio": [], "video": []}

    # Add the history to the current prompt
    for history in global_history:
        assert "prompt" in history, "Prompt must be provided in history."
        assert "response" in history, "Response must be provided in history."

        if skip_history_vision:
            history_prompt = history["prompt"].replace(IMAGE_TOKEN, "").replace(VIDEO_TOKEN, "")
        else:
            history_prompt = history["prompt"]

        history_prompt = user_prefix + history_prompt + eos + bot_prefix + history["response"] + eos
        for modality in ["image", "audio", "video"]:
            if skip_history_vision and modality in ["image", "video"]:
                continue

            if "multi_modal_data" in history and modality in history["multi_modal_data"]:
                current_multi_modal_data[modality].extend(history["multi_modal_data"][modality])
        current_prompt += history_prompt

    # Add the current request to the current prompt
    current_prompt += user_prefix + current_request["prompt"] + eos + bot_prefix
    for modality in ["image", "audio", "video"]:
        if "multi_modal_data" in current_request and modality in current_request["multi_modal_data"]:
            current_multi_modal_data[modality].extend(current_request["multi_modal_data"][modality])

    for modality in ["image", "audio", "video"]:
        if current_multi_modal_data[modality] == []:
            current_multi_modal_data.pop(modality, None)
    
    if move_image_token_to_start:
        num_image_tokens = current_prompt.count(IMAGE_TOKEN)
        current_prompt = current_prompt.replace(IMAGE_TOKEN, "")
        current_prompt = current_prompt.replace(system_prompt, "")
        current_prompt = system_prompt + user_prefix + IMAGE_TOKEN * num_image_tokens + current_prompt.lstrip(user_prefix)

    current_request["prompt"] = current_prompt.replace('<1> ','<1>').replace('<2> ','<2>')
    current_request["multi_modal_data"] = current_multi_modal_data

    return current_request


def launch_demo(
    request_inputs_queue,
    tts_output_queue,
    worker_ready,
    wait_workers_ready,
    global_history,
    interrupt_signal,
):
    vad_path = "web_demo/wakeup_and_vad/resource"
    vad_model = WakeupAndVAD(vad_path, cache_history=10)

    collected_images = deque(maxlen=8)
    collecting_images = False

    collected_audio = torch.tensor([])
    collecting_audio = False

    start_time = time.time()
    last_time_to_collect_image = start_time
    last_time_to_collect_audio = start_time

    last_output_id = 0

    def save_video(images, video_filename):

        copy_images = list(images)

        if len(copy_images) == 0:
            return
        height, width, layers = copy_images[0].shape
        size = (width, height)

        out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

        for image in copy_images:
            out.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        out.release()

    def process_image(image):
        nonlocal last_time_to_collect_image
        current_time_to_collect_image = time.time()
        if current_time_to_collect_image - last_time_to_collect_image > 1:
            collected_images.clear()
            print("Clearing the collected images")

        collected_images.append(image)
        last_time_to_collect_image = current_time_to_collect_image

    def reset_state():
        nonlocal collected_images, collected_audio

        print("Resetting the state")
        while len(global_history) > 0:
            global_history.pop()

        collected_audio = torch.tensor([])
        collected_images.clear()


    def text_streamer():
        nonlocal last_output_id

        if tts_output_queue.empty():
            yield None, None

        while not tts_output_queue.empty():

            try:
                output = tts_output_queue.get_nowait()
                llm_id = output["id"]
                temp_output, audio, length = output["response"]

                if llm_id != interrupt_signal.value:
                    print(f"Received output from other process {llm_id}, skipping...")
                    continue

                # print(f"Received audio output {temp_output}")
                if last_output_id != 0 and last_output_id != llm_id:
                    print(f"Received pause signal, pausing for 0.2s")
                    time.sleep(0.2)

                last_output_id = llm_id

                yield None, audio
                time.sleep(length * 1.5 + 0.02)
            except Empty:
                print(f"The queue is empty, text output {temp_output}")
                yield None, None
        yield None, None



    def add_audio(
            audio,
            answer_ready,
        ):
        nonlocal collected_audio, collecting_audio
        nonlocal last_time_to_collect_audio
        current_time_to_collect_audio = time.time()
        if current_time_to_collect_audio - last_time_to_collect_audio > 1:
            collected_audio = torch.tensor([])
            print("Clearing the collected audio")
        last_time_to_collect_audio= current_time_to_collect_audio

        target_sample_rate = 16000

        # Load the audio file
        waveform, sr = torchaudio.load(audio)
        # Resample the audio if necessary
        if sr != target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sample_rate)(waveform)

        chunk_size = vad_model.get_chunk_size()

        # Add the audio to the FIFO tensor
        if collected_audio.numel() == 0:
            collected_audio = waveform
        else:
            collected_audio = torch.cat((collected_audio, waveform), dim=1)

        while collected_audio.shape[1] >= chunk_size:
            # Get the chunk of data
            data = collected_audio[:, :chunk_size]
            # Process the chunk
            res = vad_model.predict(data)
            
            # Remove the processed chunk from the FIFO tensor
            collected_audio = collected_audio[:, chunk_size:]

            if res is not None:
                if "start" in res:
                    print("Start of dialog: %f" % res["start"])
                    # collecting_images = True

                if  "cache_dialog" in res:
                    print('res', res)


                    directory = './chat_history'
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    audio_duration = len(res["cache_dialog"]) / target_sample_rate

                    if audio_duration < 1.5:
                        print("The duration of the audio is less than 1.5s, skipping...")
                        continue

                    current_time = datetime.datetime.now()

                    # Format the time to create a unique filename
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    audio_filename = f"{directory}/test_dialog_{timestamp}.wav"
                    
                    torchaudio.save(audio_filename, res["cache_dialog"].unsqueeze(0), target_sample_rate)

                    if len(collected_images) > 0:
                        video_filename = f"{directory}/test_video_{timestamp}.mp4"
                        save_video(collected_images, video_filename)
                    else:
                        video_filename = ""


                    print("Start to generate response")
                    if video_filename:

                        current_request = {
                            "prompt": "<video><audio>",
                            "multi_modal_data": {
                                "video": [video_filename],
                                "audio": [audio_filename],
                            },
                        }
                    else:
                        current_request = {
                            "prompt": "<audio>",
                            "multi_modal_data": {
                                "audio": [audio_filename],
                            },
                        }

                    print(f"Start to put request into queue {current_request}")
                    request_inputs_queue.put(current_request)
        
        if not tts_output_queue.empty():
            answer_ready = 1 - answer_ready

        return answer_ready
    

    with gr.Blocks(title="VITA") as demo:

        gr.Markdown("""<center><font size=8> VITA </center>""")

        with gr.Row():
            with gr.Column():
                webcam = gr.Image(sources="webcam", type="numpy", streaming=True, label="📹 Video Recording (视频录制)",scale=2)
            with gr.Column():
                audio_stream = gr.Audio(sources=["microphone"], type='filepath', streaming=True, label="🎤 Record Audio (录音)",scale=0.5)
                answer_ready = gr.State(value=0)
                reset_context = gr.Button("🧹 Clear History (清除历史)")
                html = gr.HTML(visible=True)



        audio_stream.change(add_audio, [audio_stream, answer_ready], [answer_ready], show_progress=True)

        answer_ready.change(fn=text_streamer,  inputs=[], outputs=[html])

        reset_context.click(fn=reset_state, inputs=[], outputs=[])
        webcam.stream(fn=process_image, inputs=webcam, outputs=[])


        while not all([worker.is_set() for worker in wait_workers_ready]):
            time.sleep(0.1)
        
        gradio_worker_ready.set()
        demo.launch(
            share=False, 
            debug=True,
            server_name="0.0.0.0",
            server_port=18806,
            show_api=False,
            show_error=False,
            auth=("123", "123")
        )


if __name__ == "__main__":


    manager = multiprocessing.Manager()
    request_inputs_queue = manager.Queue() 
    tts_inputs_queue = manager.Queue() 
    tts_output_queue = manager.Queue() 

    worker_1_stop_event = manager.Event() 
    worker_2_stop_event = manager.Event() 

    worker_1_start_event = manager.Event() 
    worker_2_start_event = manager.Event()
    worker_1_start_event.set()

    worker_1_2_start_event_lock = manager.Lock()

    llm_worker_1_ready = manager.Event()
    llm_worker_2_ready = manager.Event()

    tts_worker_ready = manager.Event()
    gradio_worker_ready = manager.Event()

    interrupt_signal = manager.Value("i", 0)

    model_path = "demo_VITA_ckpt/"

    global_history = manager.list()
    global_history_limit = 1

    # Engine arguments for vLLM     
    engine_args = AsyncEngineArgs(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.8,
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={"image": 256, "audio":50},
    )


    model_1_process = multiprocessing.Process(
        target=load_model,
        kwargs={
            "llm_id": 1,
            "engine_args": engine_args, 
            "cuda_devices": "0,1", 
            "inputs_queue": request_inputs_queue,
            "outputs_queue": tts_inputs_queue,
            "tts_outputs_queue": tts_output_queue,
            "start_event": worker_1_start_event,
            "other_start_event": worker_2_start_event,
            "start_event_lock": worker_1_2_start_event_lock,
            "stop_event": worker_1_stop_event,
            "other_stop_event": worker_2_stop_event,
            "worker_ready": llm_worker_1_ready,
            "wait_workers_ready": [llm_worker_2_ready, tts_worker_ready], 
            "global_history": global_history,
            "global_history_limit": global_history_limit,
            "interrupt_signal": interrupt_signal,
        }
    )


    model_2_process = multiprocessing.Process(
        target=load_model,
        kwargs={
            "llm_id": 2,
            "engine_args": engine_args, 
            "cuda_devices": "2,3", 
            "inputs_queue": request_inputs_queue,
            "outputs_queue": tts_inputs_queue,
            "tts_outputs_queue": tts_output_queue,
            "start_event": worker_2_start_event,
            "other_start_event": worker_1_start_event,
            "start_event_lock": worker_1_2_start_event_lock,
            "stop_event": worker_2_stop_event,
            "other_stop_event": worker_1_stop_event,
            "worker_ready": llm_worker_2_ready,
            "wait_workers_ready": [llm_worker_1_ready, tts_worker_ready], 
            "global_history": global_history,
            "global_history_limit": global_history_limit,
            "interrupt_signal": interrupt_signal,
        }
    )

    tts_worker_process = multiprocessing.Process(
        target=tts_worker,
        kwargs={
            "inputs_queue": tts_inputs_queue,
            "outputs_queue": tts_output_queue,
            "worker_ready": tts_worker_ready,
            "wait_workers_ready": [llm_worker_1_ready, llm_worker_2_ready], 
        }
    )

    gradio_demo_process = multiprocessing.Process(
        target=launch_demo,
        kwargs={
            "request_inputs_queue": request_inputs_queue,
            "tts_output_queue": tts_output_queue,
            "worker_ready": gradio_worker_ready,
            "wait_workers_ready": [llm_worker_1_ready, llm_worker_2_ready, tts_worker_ready],
            "global_history": global_history,
            "interrupt_signal": interrupt_signal,
        }
    )

    model_1_process.start()
    model_2_process.start()
    tts_worker_process.start()
    gradio_demo_process.start()


    model_1_process.join()
    model_2_process.join()
    tts_worker_process.join()
    gradio_demo_process.join()





