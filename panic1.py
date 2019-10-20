def alert():
    from pygame import mixer  # Load the popular external library

    mixer.init()
    mixer.music.load('alert_signal.mp3')
    mixer.music.play()

alert()
