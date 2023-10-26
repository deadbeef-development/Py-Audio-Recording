from typing import Tuple, Callable, Any, Dict, Optional, List, Union
from functools import partial
from . import util
from .rec import recorder, library, session, segment, clip
import asyncio
from bson import ObjectId
from bson.errors import InvalidId
from aiohttp import web as aioweb
import contextvars
import aiohttp
import numpy as np
from scipy.io import wavfile
from io import BytesIO

API_START_RECORDING = '/api/start-recording'
API_LIST_ACTIVE = '/api/list-active'
API_LIST_DEFAULT_DEVICES = '/api/list-default-devices'
API_STOP_RECORDING = '/api/stop-recording'
API_CREATE_CLIP = '/api/create-clip'
API_GET_TIMELINE_SEGMENTS = '/api/get-timeline-segments'
API_GET_OBJECT = '/api/get-object'

async def asyncio_to_thread(func, /, *args, **kwargs):
    """Asynchronously run function *func* in a separate thread.

    Any *args and **kwargs supplied for this function are directly passed
    to *func*. Also, the current :class:`contextvars.Context` is propagated,
    allowing context variables from the main thread to be accessed in the
    separate thread.

    Return a coroutine that can be awaited to get the eventual result of *func*.
    """
    loop = asyncio.events.get_running_loop()
    ctx = contextvars.copy_context()
    func_call = partial(ctx.run, func, *args, **kwargs)
    return await loop.run_in_executor(None, func_call)

class endpoint_exception(Exception):
    args:Tuple[int, str, Any]

    def __init__(self, code:int, description:str, value:Any=None):
        super().__init__(code, description, value)
    
ee = partial(endpoint_exception)

class handler:
    def __init__(self, lib:library, devices:Dict[str, str]):
        self.lib = lib
        self.devices = devices

    async def start_recording(self, timeline:str, device:str=None, force:bool=False):
        already_recording = ee(1, "Aready recording for specified timeline")
        default_device_not_found = ee(2, "Default device not found for specified timeline")

        if device is None:
            device = self.devices.get(timeline)

            if device is None:
                raise default_device_not_found

            sd_device_id = util.identify_sd_device(device)
            
            if sd_device_id is None:
                raise default_device_not_found

        worker = await self.lib.start(timeline, device, force)

        if worker is None:
            raise already_recording

    async def list_active(self):
        return [
            {
                "timeline": recorder.timeline,
                "device": recorder.device
            }
            for recorder in await self.lib.active
        ]

    async def list_default_devices(self):
        return [{"timeline":tl, "device":dev} for tl, dev in self.devices.items()]

    async def stop_recording(self, timeline:str):
        not_recording = ee(1, "Not recording for specified timeline")
        
        worker = await self.lib.stop(timeline)

        if worker is None:
            raise not_recording

        return {
            "timeline": worker.timeline,
            "device": worker.device
        }

    async def create_clip(self, session_id:str, time_start:float, time_stop:float, amplify:float=None):
        no_audio_found = ee(1, "No audio found for the specified session and time range")

        c:clip = await self.lib.create_clip(session_id, time_start, time_stop, amplify)

        if c is None:
            raise no_audio_found

        return c
    
    async def get_timeline_segments(self, timeline:str, time_start:float, time_stop:float):
        return await self.lib.get_timeline_segments(timeline, time_start, time_stop)
    
    async def get_object(self, object_id:str):
        return await self.lib.get_object(object_id)

def create_aiohttp_app(lib:library, devices:Dict[str, str]) -> aioweb.Application:
    h = handler(lib, devices)
    app = aioweb.Application()

    def r(path:str, webfunc:Callable):
        async def aiohttp_handler(req:aioweb.Request):
            req_json = await req.json()

            try:
                ret = await webfunc(**req_json)
                body = 0, "Success", ret
            except endpoint_exception as e:
                body = e.args
            
            return aioweb.json_response(body)

        app.router.add_post(path, aiohttp_handler)
    
    r(API_START_RECORDING, h.start_recording)
    r(API_LIST_ACTIVE, h.list_active)
    r(API_LIST_DEFAULT_DEVICES, h.list_default_devices)
    r(API_STOP_RECORDING, h.stop_recording)
    r(API_CREATE_CLIP, h.create_clip)
    r(API_GET_TIMELINE_SEGMENTS, h.get_timeline_segments)
    r(API_GET_OBJECT, h.get_object)

    async def get_wav_file(req:aioweb.Request):
        file_name:str = req.match_info['file_name']

        if file_name.lower().endswith('.wav'):
            try:
                oid = ObjectId(file_name[-4])
            except InvalidId:
                raise aioweb.HTTPNotFound
        else:
            raise aioweb.HTTPNotFound

        filepath = f"{lib.data_dir}/{oid}.wav"
        return aioweb.FileResponse(filepath)

    app.router.add_get('/files/{file_name}', get_wav_file)

    return app

class remote_recorder(recorder):
    def __init__(self, timeline:str, device:str):
        self.timeline = timeline
        self.device = device
    
    async def start(self):
        raise NotImplementedError
    
    async def stop(self):
        raise NotImplementedError

class remote_library(library):
    def __init__(self, ar_server:aiohttp.ClientSession):
        self.ar_server = ar_server

    def _recorder(self, timeline:str, device:str):
        rr = remote_recorder(timeline, device)
        
        rr.start = partial(self.start, timeline, device, True)
        rr.stop = partial(self.stop, timeline)

        return rr
    
    async def _post(self, path:str, payload:dict):
        async with self.ar_server.post(path, json=payload) as response:
            response.raise_for_status()
            return await response.json()
    
    @property
    async def active(self) -> Tuple[recorder]:
        code, msg, val = await self._post(API_LIST_ACTIVE, {})
        return tuple([self._recorder(**rec_info) for rec_info in val])
    
    async def start(self, timeline:str, device:str, force=False) -> Optional[recorder]:
        code, msg, val = await self._post(API_START_RECORDING, {
            "timeline": timeline,
            "device": device,
            "force": force
        })

        if code == 1:
            return None
        
        return self._recorder(timeline, device)

    async def stop(self, timeline:str) -> Optional[recorder]:
        code, msg, val = await self._post(API_STOP_RECORDING, {
            "timeline": timeline
        })

        if code == 1:
            return None
        
        return self._recorder(**val)

    async def create_clip(self, session_id:str, time_start:float, time_stop:float, amplify:float=None) -> Optional[clip]:
        code, msg, val = await self._post(API_CREATE_CLIP, {
            "session_id": session_id,
            "time_start": time_start,
            "time_stop": time_stop,
            "amplify": amplify
        })

        if code == 1:
            return None

        return clip(
            ObjectId(val['clip_id']),
            ObjectId(val['session_id']),
            val['time_start'],
            val['time_stop']
        )

    async def get_timeline_segments(self, timeline:str, time_start:float, time_stop:float) -> List[Tuple[session, List[segment]]]:
        code, msg, val = await self._post(API_GET_TIMELINE_SEGMENTS, {
            "timeline": timeline,
            "time_start": time_start,
            "time_stop": time_stop
        })

        converted = list()

        for sess, segs in val:
            converted_sess = session(
                ObjectId(sess['session_id']),
                sess['timeline'],
                sess['device'],
                sess['sample_rate']
            )

            converted_segs = list()

            for seg in segs:
                converted_segs.append(segment(
                    ObjectId(seg['segment_id']),
                    ObjectId(seg['session_id']),
                    seg['time_start'],
                    seg['time_stop']
                ))
            
            converted_item = converted_sess, converted_segs
            converted.append(converted_item)
        
        return converted

    async def _get_wav_data(self, file_id:str) -> np.ndarray:
        async with self.ar_server.get(f"/files/{file_id}.wav") as response:
            response.raise_for_status()
            wav_data = BytesIO(await response.read())
        
        sr, audio_data = wavfile.read(wav_data)
        return audio_data
    
    async def get_clip_data(self, clip_id:Union[ObjectId, str]) -> np.ndarray:
        return await self._get_wav_data(clip_id)

    async def get_segment_data(self, segment_id:Union[ObjectId, str]) -> np.ndarray:
        return await self._get_wav_data(segment_id)

    async def get_object(self, oid:Union[ObjectId, str]) -> Union[session, segment, clip, None]:
        val:dict
        code, msg, val = await self._post(API_GET_OBJECT, {
            "object_id": str(oid)
        })

        if val == None:
            return None

        otype = val.pop('type')

        if otype == session.type_name:
            return session(**val)
        elif otype == segment.type_name:
            return segment(**val)
        elif otype == clip.type_name:
            return clip(**val)
