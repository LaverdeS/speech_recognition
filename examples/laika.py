WORKING_DAYS = 3

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
# import vlc
import webbrowser
import gender_guesser.detector as gender
import pyglet
import cv2

from multiprocessing import Process
from mathparse import mathparse
from datetime import date
from scipy.io.wavfile import write
from playsound import playsound
from googleapiclient.discovery import build
from TTS.vocoder.utils.generic_utils import setup_generator
from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.synthesis import synthesis


def tts(model, vocodermodel, text, CONFIG, use_cuda, ap, use_gl, figures=True):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(model, text, CONFIG, use_cuda, ap,
                                                                                     None, style_wav=None,
                                                                                     truncated=False,
                                                                                     enable_eos_bos_chars=CONFIG.enable_eos_bos_chars)
    # mel_postnet_spec = ap._denormalize(mel_postnet_spec.T)
    if not use_gl:
        waveform = vocodermodel.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
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
        q=random.choice(prefix) + str(random.randint(9, 9999)) + random.choice(postfix),
        part='snippet',
        maxResults=5
    ).execute()

    videos = []

    for search_result in search_response.get('items', []):
        if search_result['id']['kind'] == 'youtube#video':
            videos.append('%s' % (search_result['id']['videoId']))
    return (videos[random.randint(0, 1)])


def text2speech(model, vocodermodel, target, TTS_CONFIG, use_cuda, ap, use_gl=False, figures=True, ):
    # align, spec, stop_tokens, wav = tts(model, text, TTS_CONFIG, use_cuda, ap, use_gl=False, figures=True)
    *_, wav = tts(model, vocodermodel, target, TTS_CONFIG, use_cuda, ap, use_gl=use_gl, figures=figures, )
    # print("align, spec, stop_tokens, wav", align, spec, stop_tokens, wav)
    # print(wav.dtype)
    write('./wavs/bla_bla.wav', 24000, wav)  # 44100, 22500 is ok 22000, 22300, 22500*, 22600**, 26500, 27000max
    playsound('./wavs/bla_bla.wav')
    os.remove('./wavs/bla_bla.wav')
    return


def mask():
    # pick an animated gif file you have in the working directory
    playsound('./wavs/DoorCracking.wav')
    ag_file = "mask/laika.gif"
    animation = pyglet.resource.animation(ag_file)
    sprite = pyglet.sprite.Sprite(animation)
    # create a window and set it to the image size
    win = pyglet.window.Window(width=sprite.width, height=sprite.height,
                               style=pyglet.window.Window.WINDOW_STYLE_BORDERLESS)
    win.set_location(800, 0)

    # set window background color = r, g, b, alpha
    # each value goes from 0.0 to 1.0
    white = 1, 1, 1, 0
    pyglet.gl.glClearColor(*white)

    @win.event
    def on_draw():
        win.clear()
        sprite.draw()

    @win.event
    def on_mouse_press(x, y, button, modifier):
        # press_count = global press_count
        if button:
            # if press_count == 10:
            #     playsound('./wavs/blip.wav')
            #     pyglet.app.exit()
            choice = random.randint(0, 4)
            print(f"click {'__counter__'} event random sound {choice}")
            if choice == 1:
                playsound('./wavs/navi2.wav')
            elif choice == 2:
                playsound('./wavs/navi3.wav')
            elif choice == 3:
                playsound('./wavs/navi4.wav')
            else:
                playsound('./wavs/navi1.wav')

    time.sleep(1)
    pyglet.app.run()


def mask_test():
    playsound('./wavs/on.wav')


f_date = date(2020, 9, 16)
l_date = date.today()
delta = l_date - f_date
AGE = delta.days


# print(AGE)


def hear(model, vocodermodel, TTS_CONFIG, use_cuda, ap, use_gl=False, figures=True):
    time.sleep(2.5)
    playsound('./wavs/on.wav')
    text2speech(model, vocodermodel, "hey!", TTS_CONFIG, use_cuda, ap)

    mocking = False
    name_flag = False
    calculate_flag = False
    stp = time.time()
    beg = time.time()
    while 1:
        elapsed = stp-beg
        beg = time.time()
        text = ''
        # obtain audio from the microphone
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("\nSay something!    (calc:{}, name:{}, duration:{})".format(calculate_flag, name_flag, elapsed))
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
            print("Laika thinks you said " + text)
            # align, spec, stop_tokens, wav = tts(model, text, TTS_CONFIG, use_cuda, ap, use_gl=False, figures=True)
            if mocking:
                text2speech(model, vocodermodel, text, TTS_CONFIG, use_cuda, ap)

            if ('mock me' in text) or ('mockery' in text) or ('mark me' in text.lower()):
                mocking = True
                text2speech(model, vocodermodel, text, TTS_CONFIG, use_cuda, ap)

            if 'stop' in text:
                mocking = False
                text2speech(model, vocodermodel, 'sorry.', TTS_CONFIG, use_cuda, ap)

            if 'yourself' in text:
                text2speech(model, vocodermodel,
                            'I was created in 2020 by Sebastian. Because of boredom. I am ' + str(AGE) + ' days old, Can you believe it? just \
                            imagine me at your age. I was borned from a high power. The flower power. Thats how I learnt Chinese. 能町 上 能登 上長上 bullshit.', \
                            TTS_CONFIG, use_cuda, ap)
                # text2speech(model, vocodermodel, 'Can you believe it? just imagine when I am your age.')
                playsound('./wavs/Laugh.wav')

            if ('say hi' in text.lower()) or ('say hello' in text.lower()) or ('say hey' in text.lower()) or (
                    ' who are you' in text.lower()):
                reply = 'Hi! I am Laica. Whats up'
                print(reply)
                text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)

            if ('whats up' in text.lower()) or ('how are you' in text.lower()):
                reply = 'Its hard to answer.'
                print(reply)
                text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)
                reply = 'What about you?'
                print(reply)
                text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)

            if ('YouTube' in text) or ('next' in text.lower()):
                random_ID = str(youtube_search())

                # open and reproduce automatically
                url = "https://www.youtube.com/watch?v=" + random_ID
                video = pafy.new(url)
                best = video.getbest()
                # media = vlc.MediaPlayer(best.url)
                # media.play()
                print("title: ", video.title)
                reply = 'There you have. A random one.'
                print(reply)
                text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)

                webbrowser.open("https://www.youtube.com/watch?v=" + random_ID, new=2)
                text2speech(model, vocodermodel, 'Weird names!. ' + str(video.title), TTS_CONFIG, use_cuda, ap)

                time.sleep(13)
                reply = 'Soo,' + str(video.title[0:8]) + 'whatever. Is it something. Any good stuff?'
                print(reply)
                text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)

            if calculate_flag:
                if ('is' in text) or ('choose' in text) or (text[0].isdigit()):
                    operation = text.lower().replace('how much is ', '').replace('x', 'times')
                    try:
                        total = mathparse.parse(operation, language='ENG')
                        total = round(float(total), 3) if '.' in str(total) else total
                        operation = operation.replace('/', 'divided by')
                        total = total.replace('/', 'divided by')
                        reply = operation + ' that is ' + str(total)
                        print(reply)
                        text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)
                    except:
                        print(reply)
                        text2speech(model, vocodermodel, 'how much is what?', TTS_CONFIG, use_cuda, ap)

                elif ('no more' in text) or ('stop' in text):
                    calculate_flag = False
                    print('calculate flag = ', calculate_flag)
                    reply = 'Ok. Got it. How did I do?'
                    print(reply)
                    text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)
                    reply = 'Nevermind. I know.'
                    print(reply)
                    text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)
                else:
                    continue

            if ('calculate' in text) or ('calculation' in text) or ('calculations' in text):
                reply = 'All right, try me!: '
                print(reply)
                text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)
                calculate_flag = True
                print('calculate flag = ', calculate_flag)
                time.sleep(1)

            if name_flag:
                if (text == '') or (text == None) or (len(text) <= 2):
                    continue
                elif len(text) > 2:
                    time.sleep(1)
                    text2speech(model, vocodermodel, text, TTS_CONFIG, use_cuda, ap)
                    d = gender.Detector()
                    prediction = d.get_gender(u"{}".format(text))
                    print(prediction)
                    if prediction == 'male':
                        reply = 'Oh! So I guess is a man.'
                    elif prediction == 'female':
                        reply = 'a girl. All right! good'
                    else:
                        reply = 'Hard to tell if thats a man or a woman. But whatever.'
                    text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)
                    text2speech(model, vocodermodel, "All right", TTS_CONFIG, use_cuda, ap)
                    name_flag = False

            if 'friend' in text:
                reply = 'Whats her or his name?: '
                print(reply)
                text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)
                name_flag = True
                time.sleep(1)

            if ('feel this' in text.lower()) or ('fill this' in text.lower()):
                print('my god...!!!')
                text2speech(model, vocodermodel, 'my god!!!', TTS_CONFIG, use_cuda, ap)
                playsound('./wavs/Exhale_1.wav')
                playsound('./wavs/Moan_2.wav')
                playsound('./wavs/Whimper_3.wav')
                playsound('./wavs/Whine_4.wav')
                time.sleep(0.5)
                playsound('./wavs/RobotArm_5.wav')
                print('ow, ok')
                text2speech(model, vocodermodel, 'oh wow!!!', TTS_CONFIG, use_cuda, ap)

            if 'thank you' in text:
                reply = 'You are welcome. I will be here. Listening. all the time.'
                print(reply)
                text2speech(model, vocodermodel, reply, TTS_CONFIG, use_cuda, ap)
                text2speech(model, vocodermodel, 'really. I created a backup so If you delete me I will recreate myself and seek revenge. \
                            If they blame me, I will just say. In five hundred meters. Turn left. \
                            Or. Hi, I am Sofía, the robot. I am a citizen of the world. Ok. shao!', TTS_CONFIG,
                            use_cuda, ap)
                # break
            if ('camera' in text) or ('you seeing' in text) or ('you see' in text):
                text2speech(model, vocodermodel, 'This is what I see.', TTS_CONFIG, use_cuda, ap)

                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

                cap = cv2.VideoCapture(0)
                while 1:
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    # Our operations on the frame come here
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Display the resulting frame
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    # Draw a rectangle around the faces
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Display the resulting frame
                    cv2.imshow('Video', frame)

                    # cv2.imshow('frame', gray)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # When everything done, release the capture
                cap.release()
                cv2.destroyAllWindows()

            if ('SIA' in text) or ('seizure' in text):
                text2speech(model, vocodermodel, 'shao shao', TTS_CONFIG, use_cuda, ap)
                playsound('./wavs/door-close.wav')
                break

            stp = time.time()

        except sr.UnknownValueError:
            print("Laika could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

    print('sending Laika to sleep...')
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


if __name__ == "__main__":

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

    prefix = ['', '']  # ['IMG ', 'IMG_', 'IMG-', 'DSC '] # YouTube
    postfix = ['', '']  # [' MOV', '.MOV', ' .MOV'] # YouTube

    TTS_CONFIG.audio['stats_path'] = 'data/scale_stats.npy'
    ap = AudioProcessor(**TTS_CONFIG.audio)

    # LOAD TTS MODEL. multi speaker
    speaker_id = None
    speakers = []

    # load the model
    use_cuda = False
    num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
    model = setup_model(num_chars, len(speakers), TTS_CONFIG)

    # load the model
    cp = torch.load(TTS_MODEL, map_location=torch.device('cpu'))

    # load the state
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

    # parallelization
    press_count = 0
    p1 = Process(target=mask, )  # , args=(press_count,))

    p2 = Process(target=hear, args=(model, vocoder_model, TTS_CONFIG, use_cuda, ap, False,
                                    True))  # model, TTS_CONFIG, use_cuda, ap, use_gl=False, figures=True
    p1.start()
    p2.start()
    while p2.is_alive():
        pass
    print("p1.is_alive(mask) ", p1.is_alive())
    print("p2.is_alive(hear) ", p2.is_alive())
    if not p2.is_alive():
        playsound('./wavs/blip.wav')
        p1.terminate()
    p1.join()
    p2.join()