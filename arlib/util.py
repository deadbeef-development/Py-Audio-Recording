from typing import Optional
import os
import sounddevice as sd

ASOUND_DIR = '/proc/asound'

def get_stream_info():
    for child in os.listdir(ASOUND_DIR):
        if child.startswith('card'):
            card_number = child[4:]
        else:
            continue

        try:
            card_number = int(card_number)
        except ValueError:
            continue
    
        card_file_path = os.path.join(ASOUND_DIR, child)
        stream_file_path = os.path.join(card_file_path, 'stream0')

        if os.path.exists(stream_file_path):
            with open(stream_file_path, 'r') as fio:
                stream_info = fio.read()

            yield card_number, stream_info

def get_card_number(pci_device:str) -> Optional[int]:
    for card_number, stream_info in get_stream_info():
        if pci_device in stream_info:
            return card_number    
    
    return None

def get_hw_identifier(card_number:int):
    return f"(hw:{card_number},0)"

def identify_sd_device(arlib_id:str) -> Optional[int]:
    if arlib_id.startswith('sd:'):
        return int(arlib_id[3:])
    elif arlib_id.startswith('sdn:'):
        sd_name = arlib_id[4:]

        i = 0
        for device_info in sd.query_devices():
            if device_info['name'] == sd_name:
                return i
        
            i += 1
        
        return None
    elif arlib_id.startswith('pci:'):
        pci_device = arlib_id[4:]

        card_number = get_card_number(pci_device)
        hw_id = get_hw_identifier(card_number)

        i = 0
        for device_info in sd.query_devices():
            if hw_id in device_info['name']:
                return i

            i += 1
        
        return None
    else:
        raise ValueError
