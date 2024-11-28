import os
from datetime import timedelta
from uuid import uuid4

import gradio as gr
import numpy as np
import srt
from scipy.io import wavfile

from configs import settings
from transcription.faster_whisper_transcriber import FasterWhisperTranscriber
from transcription.utils.subtitle import generate_srt
from translation.translator import NLLBTranslator
from translation.utils import Preprocessor
from tts.narration import SubtitleNarrator
from tts.utils import read_srt, get_longest_srt_segment, add_segment, remove_last
from tts.xtts import XttsGenerator
from utils.basic import is_number, audio_file_extensions, video_file_extensions, SUPPORTED_FILES
from utils.logger import logger
from utils.signal import convert_2_wav, merge_video_and_audio, clip_audio

voice_generator = XttsGenerator(**settings.voice_generator)
narrator = SubtitleNarrator(voice_generator)
transcriber = FasterWhisperTranscriber(settings.transcriber)
translator = NLLBTranslator(**settings.translator, text_preprocessor=Preprocessor())
SUPPORTED_LANGUAGES = ['Default'] + voice_generator.SUPPORTED_LANGUAGES


def translate_srt(input_subs: str, input_language, target_language, batch_size=4):
    input_subs = list(srt.parse(input_subs))
    translated_subs = []
    for_translation = [i.content for i in input_subs]
    translated = translator.translate(for_translation, input_language, target_language, batch_size=batch_size,
                                      return_sentences=True)
    for i, subtitle in enumerate(input_subs):
        subtitle.content = translated[i]
        translated_subs.append(subtitle)
    return srt.compose(translated_subs)


def check_file(file, state):
    logger.debug(f'File is uploaded in {file.name}')
    extension = '.' + os.path.basename(file.name).split('.')[-1].lower()
    if extension in audio_file_extensions:
        file_type = 'audio'
    elif extension in video_file_extensions:
        file_type = 'video'
    elif extension in '.srt':
        file_type = 'srt'
    else:
        raise ValueError(f"Not supported file type {extension}.")
    state['uploaded_file'] = {"path": file.name, "type": file_type, "extension": extension}
    state['tmp_path'] = os.path.dirname(file.name)
    # print(state)
    return state


def check_example(example, state):
    logger.debug(f'File is uploaded in {example}')
    extension = '.' + os.path.basename(example).split('.')[-1].lower()
    if extension in audio_file_extensions:
        file_type = 'audio'
    elif extension in video_file_extensions:
        file_type = 'video'
    elif extension in '.srt':
        file_type = 'srt'
    else:
        raise ValueError(f"Not supported file type {extension}.")
    state['uploaded_file'] = {"path": example, "type": file_type, "extension": extension}
    state['tmp_path'] = os.path.dirname(example)
    return state


def clear_state(state):
    state['input_file'] = None
    return state


def generate(state, speed, language, radio, details, progress=gr.Progress()):
    progress(.0, desc="Transcribing")
    # print(state, language, radio, details)
    if state['uploaded_file'] is None:
        gr.Warning('Please Upload a Video, Audio or SRT file', duration=5)
        return state, None
    elif state['uploaded_file']['type'] == 'srt':
        if radio == 'Clone voice from uploaded file':
            gr.Warning('You cannot use the option Clone voice from uploaded file when you upload SRT.', duration=5)
            return state, None
        if language == 'Default':
            language = 'en'
        asr_language = 'en'
        srt_file = state['uploaded_file']['path']
    else:
        if state['uploaded_file']['type'] == 'video':
            audio_file = state['uploaded_file']['path'].replace(state['uploaded_file']['extension'], '.wav')
            logger.debug(audio_file)
            convert_2_wav(state['uploaded_file']['path'], audio_file, log=False)
            assert os.path.exists(audio_file)
        elif state['uploaded_file']['type'] == 'audio':
            audio_file = state['uploaded_file']['path']
        else:
            raise gr.Error('Something went wrong')

        result = transcriber.transcribe(audio_file, language=None)
        progress(.15, desc="Transcribing")
        asr_language = result.language
        logger.info(asr_language)
        if language == 'Default':
            language = asr_language
        words = []
        for i in result.raw_result['segments']:
            words.extend(i['words'])
        srt_file = os.path.join(state['tmp_path'], 'whisper.srt')
        generate_srt(words, output_file=srt_file, punctuation=['.', '!', '?'], delay_threshold=0.5, max_words=40)

    captions = read_srt(srt_file)
    if language != asr_language:
        progress(.20, desc="Translating")
        subtitles = translate_srt(captions, asr_language, language)
        srt_file = srt_file.replace('.srt', f'_{language}.srt')
        with open(srt_file, "w", encoding="utf-8") as file:
            file.write(subtitles)
        progress(.25, desc="Translating")
    else:
        progress(.25, desc="Transcribing")

    progress(.25, desc="Generating Voice")
    try:
        if radio == 'Predifined speakers':
            speaker_id = details
            audio_filepath = narrator.narrate(file=srt_file, voice_speed=speed, language=language,
                                              output_dir=state['tmp_path'],
                                              speaker_id=speaker_id, progress=progress, initial_progress=0.25)
        elif radio == 'Clone your voice':
            sr, audio_array = details
            user_wav_file = os.path.join(state['tmp_path'], 'user.wav')
            wavfile.write(user_wav_file, sr, audio_array)
            voice_generator.load_model_if_not_loaded()
            gpt_cond_latent, speaker_embedding = voice_generator.model.get_conditioning_latents(audio_path=user_wav_file)
            audio_filepath = narrator.narrate(file=srt_file, voice_speed=speed, language=language,
                                              output_dir=state['tmp_path'],
                                              speaker_data=(gpt_cond_latent, speaker_embedding), progress=progress,
                                              initial_progress=0.25)
        else:
            start_time, end_time, _ = get_longest_srt_segment(srt_file)
            max_segment_file = os.path.join(state['tmp_path'], 'max_segment.wav')
            clip_audio(audio_file, max_segment_file, start_time, end_time)
            voice_generator.load_model_if_not_loaded()
            gpt_cond_latent, speaker_embedding = voice_generator.model.get_conditioning_latents(audio_path=max_segment_file)
            audio_filepath = narrator.narrate(file=srt_file, voice_speed=speed, language=language,
                                              output_dir=state['tmp_path'],
                                              speaker_data=(gpt_cond_latent, speaker_embedding), progress=progress,
                                              initial_progress=0.25)
    except Exception as err:
        raise gr.Error(str(err))
    progress(1.0, desc="Generating Voice")

    if state['uploaded_file']['type'] == 'video':
        output_video_path = merge_video_and_audio(state['uploaded_file']['path'], audio_filepath,
                                                  output_video_name=str(uuid4()), log=False)
        assert os.path.exists(output_video_path)
        state['result'] = {"path": output_video_path, "type": 'video',
                           "extension": '.' + output_video_path.split('.')[-1]}
    else:
        state['result'] = {"path": audio_filepath, "type": 'audio', "extension": '.' + audio_filepath.split('.')[-1]}
    return state, None  # return state and label (label is not so it will stop appearing as it works like progress bar)


def generate_from_srt(input_component, speed, language, radio, details, progress=gr.Progress()):
    if input_component is None or input_component.strip() == '':
        raise gr.Error('SRT should not be empty')
    progress(.0, desc="Transcribing")
    if language == 'Default':
        language = 'en'
    asr_language = 'en'

    captions = input_component
    if language != asr_language:
        progress(.20, desc="Translating")
        captions = translate_srt(captions, asr_language, language)
        progress(.25, desc="Translating")
    else:
        progress(.20, desc="Generating Voice")

    try:
        if radio == 'Predifined speakers':
            speaker_id = details
            audio_filepath = narrator.narrate(raw_subtitles=captions, voice_speed=speed, language=language,
                                              output_dir=settings.app.cache_dir,
                                              speaker_id=speaker_id, progress=progress, initial_progress=0.2)
        else:
            sr, audio_array = details
            user_wav_file = os.path.join(settings.app.cache_dir, 'user.wav')
            wavfile.write(user_wav_file, sr, audio_array)
            voice_generator.load_model_if_not_loaded()
            gpt_cond_latent, speaker_embedding = voice_generator.model.get_conditioning_latents(audio_path=user_wav_file)
            audio_filepath = narrator.narrate(raw_subtitles=captions, voice_speed=speed, language=language,
                                              output_dir=settings.app.cache_dir,
                                              speaker_data=(gpt_cond_latent, speaker_embedding), progress=progress,
                                              initial_progress=0.2)
    except Exception as err:
        raise gr.Error(str(err))
    progress(1.0, desc="Generating Voice")
    return audio_filepath, None


def change_visibility(label):
    if label == "In progress":
        return gr.Label(visible=True)  # make it visible
    else:
        return gr.Label(visible=False)


def clear_result(state):
    state['result'] = None
    return state


def check_segment_inputs(subs, text, start, end):
    subs = list(srt.parse(subs))
    if text == '':
        raise gr.Error('You cannot insert empty text', duration=5)
    if not is_number(start) or not is_number(end):
        raise gr.Error('Start and stop should be numeric values', duration=5)
    if float(end) <= float(start):
        raise gr.Error('End time should be greater than start time', duration=5)
    if len(subs) > 0:
        if timedelta(seconds=float(start)) < subs[-1].end:
            raise gr.Error('End time should be greater than start time of the previous segment', duration=5)


css = """
        #logo {
          display: block;
          width: 10%;
          margin: auto;
        }
       .footer_main {
       display: inline-block;
       margin: auto;
       width: 100%;
       text-align: center;}    
       .about {
       display: inline-block;
       margin: auto;
       width: 100%}         
       footer{display:none !important};
       font-family: 'DroidArabicKufiRegular';   
    """

with gr.Blocks(
        gr.themes.Soft(font=['DroidArabicKufiRegular', "Arial", "sans-serif"],
                       text_size=gr.themes.sizes.text_lg),
        css=css,
        title='Narratai', ) as app:
    gr.Image('assets/image/banner.png', show_label=False, show_download_button=False, show_share_button=False,
             show_fullscreen_button=False, container=False)
    with gr.Tab('Media'):
        gr.HTML('<h2 style="text-align:center;">1. Upload media or select a sample</h1>')
        file_component = gr.File(label="Upload Media",
                                 file_types=['.srt'] + audio_file_extensions + video_file_extensions)
        gr.HTML(f'<p style="text-align:right;font-size:small;">*Supported files: {SUPPORTED_FILES} </p>')
        # <p style="font-size:14px; "> Any text whose font we want to change </p>
        state = gr.State(value={'uploaded_file': None, 'tmp_path': None, 'result': None})
        file_component.upload(check_file, [file_component, state], [state])

        with gr.Row():
            video_example = gr.Video(label='Hidden Input Video', autoplay=False, height=400, visible=False)
            video_examples = gr.Examples(examples=[os.path.abspath('assets/video/aDSS4QdQ30c.mp4'),
                                                   os.path.abspath('assets/video/1846897589246885888.mp4'),
                                                   os.path.abspath('assets/video/1825879772334829568.mp4')],
                                         inputs=[video_example], cache_examples=False, fn=None, run_on_click=False,
                                         label='Video samples')
            video_example.change(check_example, [video_example, state], state)

            audio_example = gr.File(label='Hidden Input Audio', visible=False)
            audio_examples = gr.Examples(examples=[os.path.abspath('assets/audio/aDSS4QdQ30c.wav'),
                                                   os.path.abspath('assets/audio/lAsjkZBNO4A.wav')],
                                         inputs=[audio_example], cache_examples=False, fn=None, run_on_click=False,
                                         label='Audio samples')
            audio_example.change(check_example, [audio_example, state], state)

            srt_example = gr.File(label='Hidden Input SRT', visible=False)
            srt_examples = gr.Examples(examples=[os.path.abspath('assets/srt/sample_1.srt'),
                                                 os.path.abspath('assets/srt/sample_2.srt')],
                                       inputs=[srt_example], cache_examples=False, fn=None, run_on_click=False,
                                       label='SRT samples')
            srt_example.change(check_example, [srt_example, state], state)


        @gr.render(inputs=state)
        def show_file(file):
            input_file = file['uploaded_file']
            if input_file is not None:
                if input_file['type'] == 'audio':
                    input_component = gr.Audio(label='Input Audio', value=input_file['path'], autoplay=False)
                elif input_file['type'] == 'video':
                    input_component = gr.Video(label='Input Video', value=input_file['path'], autoplay=False,
                                               height=400)
                elif input_file['type'] == 'srt':
                    content = read_srt(input_file['path'])
                    input_component = gr.Text(label='Input SRT', value=content, max_lines=30, interactive=False,
                                              show_copy_button=True, autoscroll=False)


        # gr.HTML('<hr>')
        gr.HTML('<h2 style="text-align:center;">2. Generate voice!</h1>')
        with gr.Accordion('Generation settings', open=False):
            speed = gr.Slider(label='Generated Voice Speed:', minimum=0.0, maximum=1.0, value=0.5, interactive=True)
            language = gr.Dropdown(label='Generated Voice Language:', choices=SUPPORTED_LANGUAGES,
                                   value='Default',
                                   interactive=True)
            radio = gr.Radio(label='Speaker Options:',
                             choices=['Predifined speakers', 'Clone your voice', 'Clone voice from uploaded file'],
                             value='Predifined speakers')


            @gr.render(inputs=radio)
            def show_file(choice):
                if choice == 'Predifined speakers':
                    details = gr.Dropdown(label='Speaker', choices=voice_generator.speakers, interactive=True,
                                          value=np.random.choice(voice_generator.speakers))
                elif choice == 'Clone your voice':
                    details = gr.Audio(label='Record your voice', sources=['microphone'])
                else:
                    details = gr.Text(visible=False)  # dummy

                generate_button.click(fn=clear_result, inputs=state, outputs=state). \
                    then(fn=lambda: "In progress", inputs=None, outputs=[loader]). \
                    then(fn=lambda: gr.update(interactive=False), inputs=None, outputs=generate_button). \
                    then(fn=generate, inputs=[state, speed, language, radio, details], outputs=[state, loader]). \
                    then(fn=lambda: gr.update(interactive=True), inputs=None, outputs=generate_button)

        # file_component.clear(clear_state, inputs=state, outputs=state)
        generate_button = gr.Button(value='Generate', variant='primary', icon='assets/image/logo_white.png')
        loader = gr.Label(show_label=False, visible=False)
        loader.change(change_visibility, loader, loader)


        @gr.render(inputs=state)
        def show_result(file):
            if file['result'] is not None:
                result = file['result']
                if result['type'] == 'audio':
                    output_component = gr.Audio(label='Generated voice', autoplay=False, show_download_button=True,
                                                show_share_button=True, editable=False, sources=[],
                                                value=result['path'])
                elif result['type'] == 'video':
                    output_component = gr.Video(label='Video with generated voice', autoplay=False, height=400,
                                                show_download_button=True, show_share_button=True, sources=[],
                                                value=result['path'])

    with gr.Tab('Create SRT'):
        gr.HTML('<h2 style="text-align:center;">1. Create SRT</h1>')
        with gr.Column():
            with gr.Row():
                content = gr.Textbox(label='Segment text', placeholder='Example: This is a test')
                start = gr.Textbox(label='Start time', placeholder='Example: 0.2')
                end = gr.Textbox(label='End time', placeholder='Example: 2.245')
            with gr.Row():
                add_segment_btn = gr.Button(value='Add segment', variant='secondary')
                remove_btn = gr.Button(value='Remove last', variant='secondary')
                remove_all_btn = gr.Button(value='Remove all', variant='secondary')

        input_component_2 = gr.Text(label='SRT', value=None, max_lines=30, interactive=True, show_copy_button=True,
                                    autoscroll=False, info='Customize srt file', lines=10, placeholder=None)
        add_segment_btn.click(check_segment_inputs,
                              inputs=[input_component_2, content, start, end],
                              outputs=None).success(add_segment,
                                                    [input_component_2, content, start, end],
                                                    [input_component_2, content, start, end])
        remove_btn.click(remove_last, [input_component_2], input_component_2)
        remove_all_btn.click(lambda: '', None, [input_component_2])

        gr.HTML('<h2 style="text-align:center;">2. Generate voice!</h1>')
        with gr.Accordion('Generation settings', open=False):
            speed_2 = gr.Slider(label='Generated Voice Speed:', minimum=0.0, maximum=1.0, value=0.5, interactive=True)
            language_2 = gr.Dropdown(label='Generated Voice Language:', choices=SUPPORTED_LANGUAGES,
                                     value='Default',
                                     interactive=True)
            radio_2 = gr.Radio(label='Speaker Options:', choices=['Predifined speakers', 'Clone your voice'],
                               value='Predifined speakers')


            @gr.render(inputs=radio_2)
            def show_file(choice):
                if choice == 'Predifined speakers':
                    details_2 = gr.Dropdown(label='Speaker', choices=voice_generator.speakers, interactive=True,
                                            value=np.random.choice(voice_generator.speakers))
                elif choice == 'Clone your voice':
                    details_2 = gr.Audio(label='Record your voice', sources=['microphone'])

                generate_button_2.click(fn=lambda: gr.update(visible=False), inputs=None, outputs=output_component_2). \
                    then(fn=lambda: "In progress", inputs=None, outputs=[loader_2]). \
                    then(fn=lambda: gr.update(interactive=False), inputs=None, outputs=generate_button_2). \
                    then(fn=generate_from_srt, inputs=[input_component_2, speed_2, language_2, radio_2, details_2],
                         outputs=[output_component_2, loader_2]). \
                    then(fn=lambda: gr.update(interactive=True), inputs=None, outputs=generate_button_2). \
                    then(fn=lambda: gr.update(visible=True), inputs=None, outputs=output_component_2)

        # file_component.clear(clear_state, inputs=state, outputs=state)
        generate_button_2 = gr.Button(value='Generate', variant='primary', icon='assets/image/logo_white.png')
        loader_2 = gr.Label(show_label=False, visible=False)
        loader_2.change(change_visibility, loader_2, loader_2)

        output_component_2 = gr.Audio(label='Generated AI voice', autoplay=False, show_download_button=True,
                                      show_share_button=True, editable=False, sources=[], visible=False)

    with gr.Tab('About'):
        gr.Markdown("""
        ## Description
        **NarratAI** is an AI-powered app built for [Akashathon](https://app.buidlbox.io/akash-network/akashathon-3),
        running on the decentralized Akash Network. 
        It processes audio and video files by transcribing speech into text, translating it into multiple languages, and generating AI-driven voiceovers. 
        The app uses Whisper to create accurate transcriptions with word-level timestamps, enabling precise subtitles. 
        It leverages the NLLB model to translate subtibles into many languages. 
        For voiceovers, it uses Coqui-TTS, an advanced AI technology to generate natural-sounding voices, offering options for predefined voices, voice cloning, or using the original speaker's voice. 
        **NarratAI** enhances media accessibility and localization while showcasing the power of decentralized infrastructure and AI.
        
        ## Motivation
        The motivation behind building this app stemmed from my experience participating in Akash [Zealy](https://zealy.io/cw/akashnetwork/questboard) campaigns, 
        where I deployed numerous AI models on the Akash Network. 
        In some tasks, users were required to create video tutorials demonstrating their deployments. 
        As someone who isn’t confident in my English accent, 
        I thought it would be amazing to build an app that generates AI-powered voiceovers for such content. As I started developing the app, I realized it could go beyond just voiceovers. 
        Adding a translation feature would make media accessible to a global audience, 
        breaking language barriers and promoting inclusivity. 
        This inspired me to create a tool that not only supports creators but also enhances accessibility for everyone.
        
        ## AI and Tech stack
        
        1. [faster-whisper](https://github.com/SYSTRAN/faster-whisper):
           - A reimplementation of OpenAI's Whisper model using CTranslate2, which is a fast inference engine for Transformer models.
           - The [medium](https://huggingface.co/Systran/faster-whisper-medium) whisper model is used because it offers a nice tradeoff of speed and accuracy.
        2. [No Language Left Behind](https://ai.meta.com/research/no-language-left-behind/):
           - Created by Meta, NLLB is a cutting-edge machine translation model designed to work with over 200 languages, including low-resource languages.
           - **NarratAI** uses the [nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) model.
        3. [Coqui TTS](https://github.com/coqui-ai/TTS):
           - Coqui-TTS is an open-source AI-driven text-to-speech synthesis tool capable of generating natural, human-like voices.
           - [XTTSv2](https://docs.coqui.ai/en/latest/models/xtts.html) model is used that supports 17 languages and voice cloning.
        4. [Gradio](https://www.gradio.app/)
           - Gradio is an open-source Python library that simplifies creating and sharing machine learning models and web-based user interfaces
        """)

        gr.Markdown("""
        ## Application Flow  
        
        """)

        diagram = gr.Image('assets/image/diagram.png', show_label=False, show_download_button=False,
                              show_share_button=False,
                              show_fullscreen_button=False, container=False, visible=False)
        diagram_path = diagram.value['url']
        gr.HTML(f'<img src="{diagram_path}" alt="Diagram">')

        gr.Markdown("""
        The diagram illustrates the workflow of the **NarratAI** application, which processes media files to generate accessible content with transcription, translation, and AI voiceovers.
        
        1. **Input**:  
           - The user uploads a video, audio or SRT file.  
             - Supported types: .mp4 .mkv .mov .mp3 .wav .flac .srt
             - Current max file size: 20MB 
           - If a video is uploaded, the app extracts the audio for further processing.  
           - If an SRT file is uploaded, the app skips the transcription step entirely.
        
        2. **Transcription with Whisper**:  
           - The extracted audio is processed using the **Whisper** model to generate an accurate transcription.  
           - **Whisper** supports multilingual transcription and provides **word-level timestamps**, making it straightforward to create an **SRT file** with captions.  
           - To ensure smoother text-to-speech synthesis, **SRT segments** are refined to include full sentences or meaningful phrases.
        
        3. **Translation with NLLB**:  
           - If the user specifies a language different from the original transcription, the **NLLB** model translates the SRT file.  
           - This results in a new SRT file with segments translated into the desired language, maintaining the original timing structure.  
        
        4. **Voice Generation with XTTS**:  
           - The **XTTS** model is applied to each SRT segment to generate artificial speech.  
           - Users have several options for voice generation:
             - **Predefined Speakers**: Choose from a set of available voices.  
             - **Voice Cloning**: Submit their own recordings to clone their voice.  
             - **Input Media Voice Cloning**: Use the voice from the uploaded media file for cloning.  
             - Adjust the generated voice speed.
        
        5. **Output Presentation**:  
           - The app combines the generated voice segments into a cohesive voiceover and presents the final output to the user. 
           - If the input media is video then it combines it with the generated voiceover in a new video.
           - The generated media can be downloaded.
           
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
        
        While these approaches improve the quality of the generated results, they still have limitations. For instance, voice generation for a segment must be repeated iteratively with increasing speed until it fits within the available time interval of the SRT segment. This iterative process slows down the overall voice generation, espesially for longer videos. 
        For this reason, **NarratAI** currently supports media files up to 20MB and is not thoroughly tested on videos longer than 3 minutes.
        
        ## Future Work
        There are a lot of things that can be improved in the app. Would be happy to get some help with the following:
        - Include speaker diarization and generate different voice for different speakers
        - Improve syncronization between SRT and generated voice segments
        - Improve UI 
        - Support longer videos by making the voice generation process faster
        
        ## Disclaimer
        
        **NarratAI** is build for educational purposes and is not intented for commercial use.
        """)

    akash_logo = gr.Image('assets/image/akash-logo-sm.png', show_label=False, show_download_button=False,
                          show_share_button=False,
                          show_fullscreen_button=False, container=False, visible=False)
    akash_logo_path = akash_logo.value['url']

    narratai_logo = gr.Image('assets/image/logo_white.png', show_label=False, show_download_button=False,
                          show_share_button=False,
                          show_fullscreen_button=False, container=False, visible=False)
    narratai_logo_path = narratai_logo.value['url']
    gr.HTML(
        f'<div class="footer_main"> \
            <p> \
                <img src="{narratai_logo_path}" alt="Akash logo" width="20" style="vertical-align: top; display: inline;"> created with ♥️ by <a href="https://x.com/cro7nis">cro7</a> and hosted on Akash Network <img src="{akash_logo_path}" alt="Akash logo" width="20" style="vertical-align: top; display: inline;"> \
            </p> \
        </div>\
        ') \

# Launch the app
app.launch(max_file_size="20mb",
           allowed_paths=[os.path.abspath('assets')],
           favicon_path='assets/image/favicon.png')
