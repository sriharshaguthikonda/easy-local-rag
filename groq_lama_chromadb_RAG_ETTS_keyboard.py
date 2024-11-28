from pynput import keyboard
from threading import Event

# Add Events for playback control


# Global variable to control playback
control_flags = {"stop": False, "pause": False, "resume": False, "next": False}

pause_event = Event()  # Used to pause and resume playback
stop_event = Event()


def on_press(key):
    try:
        if key == keyboard.Key.media_play_pause:
            if pause_event.is_set():
                print("Resuming playback")
                pause_event.set()  # Resume playback
                control_flags["pause"] = False
            else:
                print("Pausing playback")
                pause_event.clear()  # Pause playback
                control_flags["pause"] = True
        elif key == keyboard.Key.media_stop:
            print("Stopping playback")
            stop_event.set()  # Signal to stop playback
            control_flags["stop"] = True
        elif key == keyboard.Key.media_next:
            print("Skipping to next audio")
            control_flags["next"] = True
        elif key == keyboard.Key.media_previous:
            print("Restarting/resuming playback")
            pause_event.set()  # Resume if paused
            control_flags["resume"] = True
    except AttributeError:
        pass  # Ignore keys without media mappings


def handle_media_keys():
    """
    Listens for media key presses and updates control flags.
    """
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
