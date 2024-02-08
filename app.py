import functions

if __name__ == '__main__':
    print('Working loading functions')
    unet = functions.load_model()

    functions.audio_process('audio.mp3')