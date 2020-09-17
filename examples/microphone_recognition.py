import sys

sys.path.append("C:/Users/lavml/Documents/GitHub/speech_recognition")
sys.path.append("C:/Users/lavml/Documents/GitHub/TTS")

import speech_recognition as sr
import random
import time
import os
import torch
import IPython
import numpy as np
import logging
import pafy
import vlc

from scipy.io.wavfile import write
from playsound import playsound
from googleapiclient.discovery import build
from TTS.vocoder.utils.generic_utils import setup_generator
from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.synthesis import synthesis

logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)

TTS_MODEL = "./data/tts_model.pth.tar"
TTS_CONFIG = "./data/config.json"
TTS_CONFIG = load_config(TTS_CONFIG)
VOCODER_MODEL = "C:/Users/lavml/Documents/GitHub/TTS/notebooks/data/vocoder_model.pth.tar"
VOCODER_CONFIG = "C:/Users/lavml/Documents/GitHub/TTS/notebooks/data/config_vocoder.json"
VOCODER_CONFIG = load_config(VOCODER_CONFIG)

DEVELOPER_KEY = 'AIzaSyBVKHXFrtQ9utGWUsh6Q_f2L_Ezdj5dMFs'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'

prefix = ['IMG ', 'IMG_', 'IMG-', 'DSC ']
postfix = [' MOV', '.MOV', ' .MOV']

TTS_CONFIG.audio['stats_path'] = 'data/scale_stats.npy'
ap = AudioProcessor(**TTS_CONFIG.audio)


def tts(model, text, CONFIG, use_cuda, ap, use_gl, figures=True):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(model, text, CONFIG, use_cuda, ap,
                                                                                     speaker_id, style_wav=None,
                                                                                     truncated=False,
                                                                                     enable_eos_bos_chars=CONFIG.enable_eos_bos_chars)
    # mel_postnet_spec = ap._denormalize(mel_postnet_spec.T)
    if not use_gl:
        waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
        waveform = waveform.flatten()
    if use_cuda:
        waveform = waveform.cpu()
    waveform = waveform.numpy()
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(waveform.shape)
    print(" > Run-time: {}".format(time.time() - t_1))
    print(" > Real-time factor: {}".format(rtf))
    print(" > Time per step: {}".format(tps))
    IPython.display.display(IPython.display.Audio(waveform, rate=CONFIG.audio['sample_rate']))
    return alignment, mel_postnet_spec, stop_tokens, waveform


def youtube_search():
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    search_response = youtube.search().list(
        q=random.choice(prefix) + str(random.randint(999, 9999)) + random.choice(postfix),
        part='snippet',
        maxResults=5
    ).execute()

    videos = []

    for search_result in search_response.get('items', []):
        if search_result['id']['kind'] == 'youtube#video':
            videos.append('%s' % (search_result['id']['videoId']))
    return (videos[random.randint(0, 2)])


def text2speech(target):
    # align, spec, stop_tokens, wav = tts(model, text, TTS_CONFIG, use_cuda, ap, use_gl=False, figures=True)
    *_, wav = tts(model, target, TTS_CONFIG, use_cuda, ap, use_gl=False, figures=True)
    # print("align, spec, stop_tokens, wav", align, spec, stop_tokens, wav)
    # print(wav.dtype)
    write('./wavs/bla_bla.wav', 21750, wav)  # 44100, 22500 is ok
    playsound('./wavs/bla_bla.wav')
    os.remove('./wavs/bla_bla.wav')
    return


######################################################################################
use_cuda = False
# LOAD TTS MODEL
# multi speaker
speaker_id = None
speakers = []

# load the model
num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, len(speakers), TTS_CONFIG)

# load model state
cp = torch.load(TTS_MODEL, map_location=torch.device('cpu'))

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()

# set model stepsize
if 'r' in cp:
    model.decoder.set_r(cp['r'])

# LOAD VOCODER MODEL
vocoder_model = setup_generator(VOCODER_CONFIG)
vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location="cpu")["model"])
vocoder_model.remove_weight_norm()
vocoder_model.inference_padding = 0

ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])
if use_cuda:
    vocoder_model.cuda()
vocoder_model.eval()

while 1:
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    # # recognize speech using Sphinx
    # try:
    #     print("Sphinx thinks you said " + r.recognize_sphinx(audio))
    # except sr.UnknownValueError:
    #     print("Sphinx could not understand audio")
    # except sr.RequestError as e:
    #     print("Sphinx error; {0}".format(e))

    # recognize speech using Google Speech Recognition
    try:
        # for testing purposes, we're just using the default API key
        # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
        # instead of `r.recognize_google(audio)`
        text = r.recognize_google(audio)
        print("Google Speech Recognition thinks you said " + text)
        # align, spec, stop_tokens, wav = tts(model, text, TTS_CONFIG, use_cuda, ap, use_gl=False, figures=True)
        text2speech(text)

        if 'YouTube' in text:
            random_ID = str(youtube_search())
            reply = 'There you go! a random YouTube video. The ID is: ' + random_ID
            print(reply)
            text2speech(reply)

            # open and reproduce automatically
            url = "https://www.youtube.com/watch?v=" + random_ID
            video = pafy.new(url)
            best = video.getbest()
            media = vlc.MediaPlayer(best.url)
            media.play()
            print("title: ", video.title)

            time.sleep(2)
            reply = 'Soo!,' + str(video.title) + '... Is it something? Any good stuff?'
            print(reply)
            text2speech(reply)

        if 'go nerdy' in text:
            reply = 'All right, try me!: '
            print(reply)
            text2speech(reply)
        # string to calculation program here...

        if 'feel this' in text.lower():
            print('my god...!!!')
            text2speech('my god!!!')
            playsound('./wavs/Exhale_1.wav')
            playsound('./wavs/Moan_2.wav')
            playsound('./wavs/Whimper_3.wav')
            playsound('./wavs/Whine_4.wav')
            time.sleep(0.5)
            playsound('./wavs/RobotArm_5.wav')
            print('ow, ok')

        if 'thank you' in text:
            reply = 'You are welcome! bye!'
            print(reply)
            text2speech(reply)
            break

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))

    # # recognize speech using Google Cloud Speech
    # GOOGLE_CLOUD_SPEECH_CREDENTIALS = r"""INSERT THE CONTENTS OF THE GOOGLE CLOUD SPEECH JSON CREDENTIALS FILE HERE"""
    # try:
    #     print("Google Cloud Speech thinks you said " + r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS))
    # except sr.UnknownValueError:
    #     print("Google Cloud Speech could not understand audio")
    # except sr.RequestError as e:
    #     print("Could not request results from Google Cloud Speech service; {0}".format(e))

    # recognize speech using Wit.ai
    # WIT_AI_KEY = "INSERT WIT.AI API KEY HERE"  # Wit.ai keys are 32-character uppercase alphanumeric strings
    # try:
    #     print("Wit.ai thinks you said " + r.recognize_wit(audio, key=WIT_AI_KEY))
    # except sr.UnknownValueError:
    #     print("Wit.ai could not understand audio")
    # except sr.RequestError as e:
    #     print("Could not request results from Wit.ai service; {0}".format(e))
    #
    # # recognize speech using Microsoft Bing Voice Recognition
    # BING_KEY = "INSERT BING API KEY HERE"  # Microsoft Bing Voice Recognition API keys 32-character lowercase hexadecimal strings
    # try:
    #     print("Microsoft Bing Voice Recognition thinks you said " + r.recognize_bing(audio, key=BING_KEY))
    # except sr.UnknownValueError:
    #     print("Microsoft Bing Voice Recognition could not understand audio")
    # except sr.RequestError as e:
    #     print("Could not request results from Microsoft Bing Voice Recognition service; {0}".format(e))
    #
    # # recognize speech using Microsoft Azure Speech
    # AZURE_SPEECH_KEY = "INSERT AZURE SPEECH API KEY HERE"  # Microsoft Speech API keys 32-character lowercase hexadecimal strings
    # try:
    #     print("Microsoft Azure Speech thinks you said " + r.recognize_azure(audio, key=AZURE_SPEECH_KEY))
    # except sr.UnknownValueError:
    #     print("Microsoft Azure Speech could not understand audio")
    # except sr.RequestError as e:
    #     print("Could not request results from Microsoft Azure Speech service; {0}".format(e))
    #
    # # recognize speech using Houndify
    # HOUNDIFY_CLIENT_ID = "INSERT HOUNDIFY CLIENT ID HERE"  # Houndify client IDs are Base64-encoded strings
    # HOUNDIFY_CLIENT_KEY = "INSERT HOUNDIFY CLIENT KEY HERE"  # Houndify client keys are Base64-encoded strings
    # try:
    #     print("Houndify thinks you said " + r.recognize_houndify(audio, client_id=HOUNDIFY_CLIENT_ID, client_key=HOUNDIFY_CLIENT_KEY))
    # except sr.UnknownValueError:
    #     print("Houndify could not understand audio")
    # except sr.RequestError as e:
    #     print("Could not request results from Houndify service; {0}".format(e))
    #
    # # recognize speech using IBM Speech to Text
    # IBM_USERNAME = "INSERT IBM SPEECH TO TEXT USERNAME HERE"  # IBM Speech to Text usernames are strings of the form XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
    # IBM_PASSWORD = "INSERT IBM SPEECH TO TEXT PASSWORD HERE"  # IBM Speech to Text passwords are mixed-case alphanumeric strings
    # try:
    #     print("IBM Speech to Text thinks you said " + r.recognize_ibm(audio, username=IBM_USERNAME, password=IBM_PASSWORD))
    # except sr.UnknownValueError:
    #     print("IBM Speech to Text could not understand audio")
    # except sr.RequestError as e:
    #     print("Could not request results from IBM Speech to Text service; {0}".format(e))
