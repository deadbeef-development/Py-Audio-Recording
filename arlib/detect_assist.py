from .util import get_stream_info
import time

if __name__ == '__main__':
    print("Plug in a microphone to identify it.")

    previous = set()

    try:
        while True:
            current = dict(get_stream_info())
            added = set(current) - previous

            if added:
                for card_number in added:
                    print(card_number)
                    print(current[card_number].splitlines()[0])
                    print()
                
                previous = set(current)

            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
