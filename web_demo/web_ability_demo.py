import torch
import os
import argparse
import numpy as np
import copy
import gradio as gr
import re
import torchaudio
import io
import ffmpeg
from vita.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_IMAGE_TOKEN, MAX_IMAGE_LENGTH, MIN_IMAGE_LENGTH
from vita.conversation import conv_templates, SeparatorStyle
from vita.util.mm_utils import tokenizer_image_token, tokenizer_image_audio_token 
from PIL import Image
from decord import VideoReader, cpu
from vllm import LLM, SamplingParams
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoFeatureExtractor



PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."



def remove_special_characters(input_str):
    return input_str.replace('<2>', '').replace('<1>', '').replace('<3>', '')

def is_video(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in video_extensions

def is_image(file_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions

def is_wav(file_path):
    wav_extensions = {'.wav'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in wav_extensions


def convert_webm_to_mp4(input_file, output_file):
    try:
        (
            ffmpeg
            .input(input_file)
            .output(output_file, vcodec='libx264', acodec='aac')
            .run()
        )
        print(f"Conversion successful: {output_file}")
    except ffmpeg.Error as e:
        print(f"Error: {e.stderr.decode()}")
        raise


def _get_rawvideo_dec(video_path, max_frames=MAX_IMAGE_LENGTH, min_frames=MIN_IMAGE_LENGTH, video_framerate=1, s=None, e=None):
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
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1

    if num_frames > 0:
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))
        all_pos = list(range(f_start, f_end + 1, t_stride))

        if len(all_pos) > max_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
        elif len(all_pos) < min_frames:
            sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=min_frames, dtype=int)]
        else:
            sample_pos = all_pos

        patch_images = [Image.fromarray(f).convert("RGB") for f in vreader.get_batch(sample_pos).asnumpy()]
        return patch_images, len(patch_images)
    else:
        print(f"video path: {video_path} error.")

def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0

    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = "<br></code></pre>"
        else:
            if i > 0 and count % 2 == 1:
                line = line.replace("`", r"\`")
                line = line.replace("<", "&lt;")
                line = line.replace(">", "&gt;")
                line = line.replace(" ", "&nbsp;")
                line = line.replace("*", "&ast;")
                line = line.replace("_", "&lowbar;")
                line = line.replace("-", "&#45;")
                line = line.replace(".", "&#46;")
                line = line.replace("!", "&#33;")
                line = line.replace("(", "&#40;")
                line = line.replace(")", "&#41;")
                line = line.replace("$", "&#36;")
            lines[i] = "<br>" + line

    return "".join(lines)



def _launch_demo(llm, model_config, sampling_params, tokenizer, feature_extractor):
    def predict(_chatbot, task_history):
        chat_query = task_history[-1][0]
        print(task_history)

        conv_mode = "mixtral_two"
        conv = conv_templates[conv_mode].copy()
        
        all_audio_path = []
        all_visual_tensor = []

        qs = ''
        input_mode = 'lang'
        for i, (q, a) in enumerate(task_history):
            if isinstance(q, (tuple, list)):
                if is_image(q[0]):
                    images = [Image.open(q[0]).convert("RGB")]
                    all_visual_tensor.extend(images)
                    input_mode = 'image'
                    qs += DEFAULT_IMAGE_TOKEN * len(images) + '\n'
                elif is_video(q[0]):             
                    video_frames, slice_len = _get_rawvideo_dec(q[0])
                    all_visual_tensor.extend(video_frames)
                    input_mode = 'video'
                    qs += DEFAULT_IMAGE_TOKEN * slice_len + '\n'
                elif is_wav(q[0]):
                    if a is not None and a.startswith('<2>'):
                        continue
                    else:
                        all_audio_path.append(q[0])
                        new_q = qs + DEFAULT_AUDIO_TOKEN
                        qs = ''
                        conv.append_message(conv.roles[0], new_q)
                        conv.append_message(conv.roles[1], a)
            else:
                new_q = qs + q
                qs = ''
                conv.append_message(conv.roles[0], new_q)
                conv.append_message(conv.roles[1], a)

        print(conv)
        prompt = conv.get_prompt(input_mode)

        if all_audio_path != []:
            input_ids = tokenizer_image_audio_token(
                prompt, tokenizer, 
                image_token_index=model_config.image_token_index, 
                audio_token_index=model_config.audio_token_index
            )
            audio_list = []
            for single_audio_path in all_audio_path:
                try:
                    audio, original_sr = torchaudio.load(single_audio_path)
                    # The FeatureExtractor was trained using a sampling rate of 16000 Hz
                    target_sr = 16000
                    # Resample
                    if original_sr != target_sr:
                        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                        audio = resampler(audio)
                    audio_features = feature_extractor(audio, sampling_rate=target_sr, return_tensors="pt")["input_features"]
                    audio_list.append(audio_features.squeeze(0))
                except Exception as e:
                    print(f"Error processing {single_audio_path}: {e}")
        else:
            input_ids = tokenizer_image_token(
                prompt, tokenizer, 
                image_token_index=model_config.image_token_index
            )



        if all_visual_tensor == [] and all_audio_path == []:
            datapromt={
                 "prompt_token_ids": input_ids,
            }
 
        elif all_visual_tensor != [] and all_audio_path == []:
            datapromt={
                "prompt_token_ids": input_ids,
                "multi_modal_data": {
                    "image": all_visual_tensor
                    },
            }
        elif all_visual_tensor == [] and all_audio_path != []:
            datapromt={
                "prompt_token_ids": input_ids,
                "multi_modal_data": {
                    "audio": audio_list
                    },
            }
        else:
            datapromt={
                "prompt_token_ids": input_ids,
                "multi_modal_data": {
                    "image": all_visual_tensor,
                    "audio": audio_list
                    },
            }

        
        output = llm.generate(datapromt, sampling_params=sampling_params)
        outputs = output[0].outputs[0].text
     
        task_history[-1] = (chat_query, outputs)
        remove_special_characters_output = remove_special_characters(outputs)  
        _chatbot[-1] = (chat_query, _parse_text(remove_special_characters_output))


        print("query",chat_query)
        print("task_history",task_history)
        print(_chatbot)
        print("answer:  ",outputs)
        yield _chatbot





    def add_text(history, task_history, text):
        task_text = text
        if len(text) >= 2 and text[-1] in PUNCTUATION and text[-2] not in PUNCTUATION:
            task_text = text[:-1]
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def add_audio(history, task_history, file):
        print(file)
        if file is None:
            return history, task_history
        history = history + [((file,), None)]
        task_history = task_history + [((file,), None)]
        return history, task_history

    def add_video(history, task_history, file):
        print(file)
        if file is None:
            return history, task_history
        new_file_name = file.replace(".webm",".mp4")
        if file.endswith(".webm"):
            convert_webm_to_mp4(file, new_file_name)
        task_history = task_history + [((new_file_name,), None)]
        return history, task_history


    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks(title="VideoMLLM") as demo:
        gr.Markdown("""<center><font size=8>VITA</center>""")
        chatbot = gr.Chatbot(label='VITA', elem_classes="control-height", height=500)
        query = gr.Textbox(lines=2, label='Text Input')
        task_history = gr.State([])
        with gr.Row():
            add_text_button = gr.Button("Submit Text (提交文本)")
            add_audio_button = gr.Button("Submit Audio (提交音频)")
        with gr.Row():
            with gr.Column(scale=2):
                addfile_btn = gr.UploadButton("📁 Upload (上传文件[视频,图片])", file_types=["video", "image"])
                video_input = gr.Video(sources=[ "webcam"], height=400, width=700, container=True, interactive=True, show_download_button=True, label="📹 Video Recording (视频录制)")
   
        
            with gr.Column(scale=1):
                empty_bin = gr.Button("🧹 Clear History (清除历史)")
                record_btn = gr.Audio(sources=[ "microphone","upload"], type="filepath", label="🎤 Record or Upload Audio (录音或上传音频)", show_download_button=True, waveform_options=gr.WaveformOptions(sample_rate=16000))


  
        add_text_button.click(add_text, [chatbot, task_history, query], [chatbot, task_history], show_progress=True).then(
            reset_user_input, [], [query]
        ).then(
                predict, [chatbot, task_history], [chatbot], show_progress=True  
        )


        video_input.stop_recording(add_video, [chatbot, task_history, video_input], [chatbot, task_history], show_progress=True)
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)



        add_audio_button.click(add_audio, [chatbot, task_history,record_btn], [chatbot, task_history], show_progress=True).then(
                predict, [chatbot, task_history], [chatbot], show_progress=True   
        )
     


    server_port = 18806
    demo.launch(
        share=False,
        debug=True,
        server_name="0.0.0.0",
        server_port=server_port,
        show_api=False,
        show_error=False,
        auth=('123','123'),
        )

def main(model_path):
  
    llm = LLM(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=2,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        disable_custom_all_reduce=True,
        limit_mm_per_prompt={'image':256,'audio':50}
    )  

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = SamplingParams(temperature=0.01, max_tokens=512, best_of=1, skip_special_tokens=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, subfolder="feature_extractor", trust_remote_code=True)

    _launch_demo(llm, model_config, sampling_params, tokenizer, feature_extractor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the web demo with your model path.')
    parser.add_argument('model_path', type=str, help='Path to the model')
    args = parser.parse_args()
    main(args.model_path)
