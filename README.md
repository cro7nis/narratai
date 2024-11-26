
<img src="assets/image/Narratai_lg.png"  alt="Narratai logo"/>

## Description
Narratai is an AI-powered app built for [Akashathon](https://app.buidlbox.io/akash-network/akashathon-3) 
that runs on the decentralized Akash Network <img src="assets/image/akash-logo-sm.png" alt="drawing" style="width:15px;"/>.
It transcribes audiovisual media, translates text into multiple languages, and generates AI-powered voiceovers. 
Designed to enhance media accessibility and localization using decentralized infrastructure.

## Motivation
The motivation behind building this app stemmed from my experience participating in Akash Zealy campaigns, 
where I deployed numerous AI models on the Akash Network. 
In some tasks, users were required to create video tutorials demonstrating their deployments. 
As someone who isn’t confident in my English accent, 
I thought it would be amazing to build an app that generates AI-powered voiceovers for such content.

As I started developing the app, I realized it could go beyond just voiceovers. 
Adding a translation feature would make media accessible to a global audience, 
breaking language barriers and promoting inclusivity. 
This inspired me to create a tool that not only supports creators but also enhances accessibility for everyone.

## AI and Tech stack

1. [faster-whisper](https://github.com/SYSTRAN/faster-whisper):
   - A reimplementation of OpenAI's Whisper model using CTranslate2, which is a fast inference engine for Transformer models.
   - The medium whisper model because it offers a nice tradeoff of speed and accuracy.
2. [No Language Left Behind](https://ai.meta.com/research/no-language-left-behind/):
   - Created by Meta, NLLB is a cutting-edge machine translation model designed to work with over 200 languages, including low-resource languages.
   - Narratai uses the [nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) model.
3. [Coqui TTS](https://github.com/coqui-ai/TTS):
   - Coqui-TTS is an open-source AI-driven text-to-speech synthesis tool capable of generating natural, human-like voices.
   - [XTTSv2](https://docs.coqui.ai/en/latest/models/xtts.html) model is used that supports 17 languages and voice cloning.
4. [Gradio](https://www.gradio.app/)
   - Gradio is an open-source Python library that simplifies creating and sharing machine learning models and web-based user interfaces
    
## Application Flow  
![alt text](assets/image/diagram.png)

The diagram illustrates the workflow of the **Narratai** application, which processes media files to generate accessible content with transcription, translation, and AI voiceovers.

1. **Input**:  
   - The user uploads a video, audio or SRT file.  
   - If a video is uploaded, the app extracts the audio for further processing.  
   - If an SRT file is uploaded, the app omits the transcriptions step

2. **Transcription with Whisper**:  
   - The extracted audio is processed using the **Whisper** model to generate an accurate transcription.  
   - **Whisper** supports multilingual transcription and provides **word-level timestamps**, making it straightforward to create an **SRT file** for subtitles.  
   - To ensure smoother text-to-speech synthesis, **SRT segments** are refined to include full sentences or meaningful phrases.  

3. **Translation with NLLB**:  
   - If the user specifies a language different from the original transcription, the **NLLB** model translates the SRT file.  
   - This results in a new SRT file with segments translated into the desired language, maintaining the original timing structure.  

4. **Voice Generation with XTTS**:  
   - The **XTTS** (based on Coqui-TTS) model is applied to each SRT segment to generate audio voiceovers.  
   - Users have several options for voice generation:
     - **Predefined Speakers**: Choose from a set of available voices.  
     - **Voice Cloning**: Submit their own recordings to clone their voice.  
     - **Input Media Voice Cloning**: Use the voice from the uploaded media file.  
     - Adjust the generated voice speed.

5. **Output Presentation**:  
   - The app combines the generated voice segments into a cohesive voiceover and presents the final output to the user. 
   - If the input media is video then it combines it with the generated voiceover.
   - The generated media can be downloaded

## Examples

| Original                                  | Predefined Speaker - English | Predefined Speaker - Spanish | Voice Cloning - Japanese |
|-------------------------------------------|------------------------------|------------------------------|--------------------------|
|  |                              |                              |



## Challenges and Limitations
Synchronizing an SRT file with AI-generated voice models poses several challenges due to differences in timing, speech dynamics, and model behavior. Here's why this process can be complex:
 1. **Text-to-Speech Timing Variability**
    - AI-generated voices, even with advanced models like **Coqui-TTS**, may not perfectly match the original audio timing.
    - Generated speech can vary in duration due to:
      - Differences in **speech speed** (AI voices may speak faster or slower than expected).
      - Changes in **pausing** or phrasing that differ from the original delivery.
2. **Subtitle Segmentation**
   - **SRT files** are often split into small segments based on word timestamps or pauses. However, AI-generated voices typically perform better with complete sentences or meaningful phrases.

To deal with these issues:
- The generated SRT segments are divided based on sentences, unless a sentence exceeds 40 words.
- A simple heuristic approach is developed, to adjust the generated voice speed based on the number of words in a segment and the available time. If a segment contains many words and has limited time, the voice speed will increase. Conversely, if a segment has fewer words or more time, the voice speed will decrease.

These approaches although imporve the generation result, still have limitations. For example, the voice generation for a segment has to be applied iteratively with increasing speed until the generated segment fits the available time interval of the SRT segment. This slows down the voice generation procedure.

## How to host on Akash


## Future Work
There are a lot of things that can be improved in the app. Would be happy to get some help with the following:
- Include speaker diarization and generate different voice for different speakers
- Improve syncronization between SRT and generated voice segments
- Improve UI 
