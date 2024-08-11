# VITA: Towards Open-Source Interactive Omni Multimodal LLM


<p align="center">
    <img src="./asset/vita_log2.png" width="100%" height="100%">
</p>

<font size=7><div align='center' > [[🍎 Project Page]()] [[📖 arXiv Paper]()] [[🤗 Model Weights]()]  </div></font>


---

## 🔥 News
* **`2024.08.12`** 🌟 We launch VITA, the **First-Ever** open-source interactive omni multimodal LLM!



## 👀 VITA Overview
 The remarkable multimodal capabilities and interactive experience of GPT-4o underscore their necessity in practical applications, yet open-source models rarely excel in both areas. In this paper, we introduce VITA, the first-ever open-source Multimodal Large Language Model (MLLM) adept at simultaneous processing and analysis of Video, Image, Text, and Audio modalities, and meanwhile has an advanced multimodal interactive experience. Our work distinguishes from existing open-source MLLM through **three key features**:

-  **Omni Multimodal Understanding**: VITA demonstrates robust foundational capabilities of multilingual, vision, and audio understanding, as evidenced by its strong performance across a range of both unimodal and multimodal benchmarks.  
 - **non-awakening interaction**: VITA can be activated and respond to user audio questions in the environment without the need for a wake-up word or button. 
 - **audio interrupt interaction**: VITA is able to simultaneously track and filter external queries in real-time. This allows users to interrupt the model's generation at any time with new questions, and VITA will respond to the new query accordingly.

<p align="center">
    <img src="./asset/VITA_features.png" width="88%" height="88%">
</p>

VITA is capable of processing inputs in the form of pure text/audio, as well as video/image combined with text/audio. Besides, two key techniques are adopted to advance the multimodal interactive experience: 
 
 - **State token**. We set different state tokens for different query inputs. <1> corresponds to the effective query audio, such as “what is the biggest animal in the world?”, for which we expect a response from the model. <2> corresponds to the noisy audio, such as someone in the environment calls me to eat, for which we expect the model not to reply. <3> corresponds to the query text, i.e., the question given by the user in text form. During the training phase, we try to teach the model to automatically distinguish different input queries. During the deployment phase, with <2> we can implement non-awakening interaction. 
 - **Duplex scheme**. We further introduce a duplex scheme for the audio interrupt interaction. Two models are running at the same time, where the generation model is responsible for handling user queries. When the generation model starts working, the other model monitors the environment. If the user interrupts with another effective audio query, the monitoring model aggregates the historical context to respond to the latest query, while the generation model is paused and tune to monitor, i.e., the two models swap identities.
<p align="center">
    <img src="./asset/VITA_duplex.png" width="88%" height="88%">
</p>



## 📈 Experimental Results
- **Comparison of official Mixtral 8x7B Instruct and our trained Mixtral 8x7B**

<p align="center">
    <img src="./asset/language_eval2.png" width="68%" height="50%">
</p>


- **Evaluation on ASR tasks.**

<p align="center">
    <img src="./asset/audio_eval.png" width="96%" height="96%">
</p>

- **Evaluation on image and video understanding.**

<p align="center">
    <img src="./asset/vision_eval.png" width="96%" height="96%">
</p>



## ✒️ Citation

If you find our work helpful for your research, please consider citing our work.   

```bibtex
@article{fu2024vita,
  title={VITA: Towards Open-Source Interactive Omni Multimodal LLM},
  author={Fu, Chaoyou and Lin, Haojia and Long, Zuwei and Shen, Yunhang and Zhao, Meng and Zhang, Yifan and Wang, Xiong and Yin, Di and Ma, Long and Zheng, Xiawu and He, Ran and Ji, Rongrong and Wu, Yunsheng and Shan, Caifeng and Sun, Xing},
  journal={arXiv preprint arXiv:2408},
  year={2024}
}
```

## 📜 Related Works

Explore our related researches:
-  **[Video-MME]** [Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis](https://github.com/BradyFU/Video-MME) 
-  **[MME]** [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
-  **[Awesome-MLLM]** [A Survey on Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)

