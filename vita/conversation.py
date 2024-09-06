import dataclasses
from enum import Enum, auto
from typing import List


class SeparatorStyle(Enum):
    """Different separator style."""

    TWO = auto()
    PLAIN = auto()
    MixtralZh = auto()
    MixtralTwo = auto


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self, modality=None):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            if "mmtag" in self.version:
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<Image><image></Image>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"

        elif self.sep_style == SeparatorStyle.MixtralZh:
            seps = [self.sep, self.sep2]
            ret = "system:" + self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += "\n" + role + ":" + message + seps[i % 2]
                else:
                    ret += "\n" + role + ":"

        elif self.sep_style == SeparatorStyle.MixtralTwo:
            seps = [self.sep, self.sep2]
            has_image = False
            for i, (role, message) in enumerate(messages):
                if message and "<image>" in message:
                    has_image = True
                    break
            if has_image:
                assert modality == "image" or modality == "video"
                if modality == "image":
                    self.system = self.system[0]
                elif modality == "video":
                    self.system = self.system[1]
                else:
                    raise ValueError
            else:
                assert modality == "lang"
                self.system = self.system[2]
            ret = "system:" + self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += "\n" + role + ":" + message + seps[i % 2]
                else:
                    ret += "\n" + role + ":"

        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    from PIL import Image

                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":

                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(f"Invalid image_process_mode: {image_process_mode}")

                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                    img_str = (
                        f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    )
                    msg = img_str + msg.replace("<image>", "").strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_mixtral_zh = Conversation(
    system="你是一个人工智能机器人。\n- 你是研究社区开发的大语言模型。你的设计宗旨是有益、诚实且无害。\n- 你支持使用用户选择的多种语言流利地进行交流并解答用户的问题。\n- 如果用户更正你生成的错误答案，你会向用户致歉并与用户探讨正确的答案。",
    roles=("user", "bot"),
    version="mixtral_zh",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MixtralZh,
    sep="</s>",
    sep2="</s>",
)

conv_mixtral_two = Conversation(
    system=[
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the image given by the user, and it is strictly forbidden to answer the question without the content of the image. Please note that you are seeing the image, not the video.",
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user. \n- You must answer the question strictly according to the content of the video given by the user, and it is strictly forbidden to answer the question without the content of the video. Please note that you are seeing the video, not the image.",
        "You are an AI robot and your name is VITA. \n- You are a multimodal large language model developed by the open source community. Your aim is to be helpful, honest and harmless. \n- You support the ability to communicate fluently and answer user questions in multiple languages of the user's choice. \n- If the user corrects the wrong answer you generated, you will apologize and discuss the correct answer with the user.",
    ],
    roles=("user", "bot"),
    version="mixtral_two",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MixtralTwo,
    sep="</s>",
    sep2="</s>",
)

conv_phi3 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="phi3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|endoftext|>",
)

conv_minicpm = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="minicpm",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="llama",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|end_of_text|>",
)

conv_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

default_conversation = conv_mixtral_two
conv_templates = {
    "default": conv_mixtral_two,
    "mixtral_zh": conv_mixtral_zh,
    "mixtral_two": conv_mixtral_two,
    "phi3": conv_phi3,
    "plain": conv_plain,
    "minicpm": conv_minicpm,
    "llama": conv_llama,
}

if __name__ == "__main__":
    print(default_conversation.get_prompt())
