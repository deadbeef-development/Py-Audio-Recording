from typing import Dict, Tuple, List, Optional, Union
import multiprocessing
from multiprocessing import Process, Event
import pymongo
from pymongo import MongoClient
from bson import ObjectId
from pymongo.collection import Collection
from pymongo.results import InsertOneResult
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import os
import librosa
from . import util
from time import time
from abc import ABC, abstractmethod, abstractproperty
import asyncio

APPROX_SEGMENT_DURATION = 60.0 # Seconds

INSERT_FAILED_SESSION_DOC = 1
INSERT_FAILED_SEGMENT_DOC = 2
INSERT_FAILED_CLIP_DOC = 3

MESSAGES_INSERT_FAILED = {
    1: "Failed to insert session doc",
    2: "Failed to insert segment doc",
    3: "Failed to insert clip doc"
}

class session:
    type_name = "arlib.session"

    def __init__(self, session_id:ObjectId, timeline:str, device:str, sample_rate:int):
        self.session_id = session_id
        self.timeline = timeline
        self.device = device
        self.sample_rate = sample_rate
    
    def to_dict(self) -> dict:
        return {
            "type": self.type_name,
            "session_id": str(self.session_id),
            "timeline": self.timeline,
            "device": self.device,
            "sample_rate": self.sample_rate
        }

class segment:
    type_name = "arlib.segment"

    def __init__(self, segment_id:ObjectId, session_id:ObjectId, time_start:float, time_stop:float):
        self.segment_id = segment_id
        self.session_id = session_id
        self.time_start = time_start
        self.time_stop = time_stop

    def to_dict(self) -> dict:
        return {
            "type": self.type_name,
            "segment_id": str(self.segment_id),
            "session_id": str(self.session_id),
            "time_start": self.time_start,
            "time_stop": self.time_stop
        }

class clip:
    type_name = "arlib.clip"

    def __init__(self, clip_id:ObjectId, session_id:ObjectId, time_start:float, time_stop:float):
        self.clip_id = clip_id
        self.session_id = session_id
        self.time_start = time_start
        self.time_stop = time_stop
    
    def to_dict(self) -> dict:
        return {
            "type": self.type_name,
            "clip_id": str(self.clip_id),
            "session_id": str(self.session_id),
            "time_start": self.time_start,
            "time_stop": self.time_stop
        }

def get_collection(uri:str, database:str, name:str) -> Collection:
    client = MongoClient(uri)
    c:Collection = client[database][name]

    c.create_index('type')

    c.create_index('properties.session')
    c.create_index('properties.time_start')
    c.create_index('properties.time_stop')

    c.create_index('lookup.timeline')

    return c

class data_dir_not_found(Exception): ...

class insert_failed(Exception):
    def __init__(self, exc_id:int):
        super().__init__(MESSAGES_INSERT_FAILED[exc_id])
        self.exc_id

def raise_not_ack(result:InsertOneResult, exc_id:int):
    if not result.acknowledged:
        raise insert_failed(exc_id)

class recorder(ABC):
    @abstractproperty
    def timeline(self) -> str: ...

    @abstractproperty
    def device(self) -> str: ...

    @abstractmethod
    async def start(self): ...

    @abstractmethod
    async def stop(self): ...

class library(ABC):
    @abstractproperty
    async def active(self) -> Tuple[recorder]: ...
    
    @abstractmethod
    async def start(self, timeline:str, device:str, force=False) -> Optional[recorder]: ...

    @abstractmethod
    async def stop(self, timeline:str) -> Optional[recorder]: ...

    @abstractmethod
    async def create_clip(self, session_id:str, time_start:float, time_stop:float, amplify:float=None) -> Optional[clip]: ...

    @abstractmethod
    async def get_timeline_segments(self, timeline:str, time_start:float, time_stop:float) -> List[Tuple[session, List[segment]]]: ...

    @abstractmethod
    async def get_clip_data(self, clip_id:Union[ObjectId, str]) -> np.ndarray: ...

    @abstractmethod
    async def get_segment_data(self, segment_id:Union[ObjectId, str]) -> np.ndarray: ...

    @abstractmethod
    async def get_object(self, object_id:Union[ObjectId, str]) -> Union[session, segment, clip, None]: ...

class parallel_recorder(recorder, Process):
    def __init__(self, uri:str, database:str, collection:str, data_dir:str, timeline:str, device:str):
        super(recorder, self).__init__(name=f"arlib.Recorder[{timeline}]")

        self.uri = uri
        self.database = database
        self.collection = collection
        self.data_dir = data_dir
        self.timeline = timeline
        self.device = device

        self._stop = Event()

        self.exc = multiprocessing.Queue(maxsize=1)

    async def start(self):
        self._stop.clear()
        super(Process, self).start()
    
    async def stop(self):
        self._stop.set()
        self.join()
    
    def _run0(self):
        sd_device_id = int(util.identify_sd_device(self.device))

        sample_rate = sd.query_devices(sd_device_id, 'input')['default_samplerate']

        collection = get_collection(self.uri, self.database, self.collection)

        session_doc = {
            "type": session.type_name,
            "properties": {
                "timeline": self.timeline,
                "device": self.device,
                "sample_rate": sample_rate
            }
        }

        session_doc_result = collection.insert_one(session_doc) ;\
            raise_not_ack(session_doc_result, INSERT_FAILED_SESSION_DOC)

        def save(seg_data:np.ndarray, time_start:float, time_stop:float):
            segment_doc = {
                "type": segment.type_name,
                "properties": {
                    "session_id": session_doc_result.inserted_id,
                    "time_start": time_start,
                    "time_stop": time_stop
                },
                "lookup": {
                    "timeline": self.timeline # Optimization, not strictly required
                }
            }

            segment_doc_result = collection.insert_one(segment_doc) ;\
                raise_not_ack(segment_doc_result, INSERT_FAILED_SEGMENT_DOC)

            filepath = os.path.join(self.data_dir, f"{segment_doc_result.inserted_id}.wav")

            seg_data = np.clip(seg_data, -1, 1)
            seg_data = (seg_data * 32767).astype(np.int16)

            wavfile.write(filepath, int(sample_rate), seg_data)

        if not os.path.isdir(self.data_dir):
            raise data_dir_not_found(self.data_dir)

        with sd.InputStream(device=sd_device_id, channels=1, samplerate=sample_rate) as stream:
            seg_builder:List[np.ndarray] = list()
            first_time_start = None
            
            def construct_and_save(first_time_start, time_stop):
                segment = np.concatenate(seg_builder)

                save(segment, first_time_start, time_stop)

            while not self._stop.is_set():
                chunk, _ol = stream.read(1024)
                time_stop = time()

                approx_time_start = time_stop - (len(chunk) / sample_rate)

                if first_time_start is None:
                    first_time_start = approx_time_start
                
                seg_builder.append(chunk)

                approx_elapsed = time_stop - first_time_start

                if approx_elapsed >= APPROX_SEGMENT_DURATION:
                    construct_and_save(first_time_start, time_stop)

                    seg_builder.clear()
                    first_time_start = None
            
            if first_time_start is not None:
                construct_and_save(first_time_start, time_stop)

    def run(self):
        try:
            return self._run0()
        except Exception as e:
            self.exc.put(e)
            raise

class parallel_library(library):
    def __init__(self, uri:str, database:str, collection:str, data_dir:str):
        self.uri = uri
        self.database = database
        self.collection = collection
        self.data_dir = data_dir

        self._worker_lock = asyncio.Lock()
        self._active:Dict[str, recorder] = dict()
    
    def _get_active(self, timeline:str) -> recorder:
        try:
            worker = self._active[timeline]
        except KeyError:
            return None
    
        if worker.is_alive():
            return worker
        else:
            del self._active[timeline]
            return None

    @property
    async def active(self) -> Tuple[recorder]:
        async with self._worker_lock:
            active_workers = list()

            for timeline, worker in self._active.items():
                if worker.is_alive():
                    active_workers.append(worker)
                else:
                    del self._active[timeline]

            return tuple(active_workers)

    # Return None if a worker is already active
    async def start(self, timeline:str, device:str, force=False) -> Optional[recorder]:
        async with self._worker_lock:
            current = self._get_active(timeline)

            if current is not None:
                if force:
                    current.stop()
                else:
                    return None
            
            worker = recorder(
                self.uri,
                self.database,
                self.collection,
                self.data_dir,
                timeline,
                device
            )

            worker.start()

            self._active[timeline] = worker

            return worker
    
    # Return None if no worker is active for the specified timeline
    async def stop(self, timeline:str) -> Optional[recorder]:
        async with self._worker_lock:
            current = self._get_active(timeline)

            if current is not None:
                current.stop()

            return current
    
    def _construct_single(self, seg_docs:list, sample_rate:int, time_start:float, time_stop:float, amplify:float=None) -> Tuple[np.ndarray, float, float]:
        trimmed_data:np.ndarray = ...
        revised_time_start:float = ...
        revised_time_stop:float = ...

        first_seg_time_start = seg_docs[0]['properties']['time_start']
        last_seg_time_stop = seg_docs[-1]['properties']['time_stop']

        revised_time_start = time_start
        revised_time_stop = time_stop



        if time_start < first_seg_time_start:
            revised_time_start = first_seg_time_start
        
        if time_stop > last_seg_time_stop:
            revised_time_stop = last_seg_time_stop
        


        data_parts = list()

        for seg in seg_docs:
            filepath = f"{self.data_dir}/{seg['_id']}.wav"
            _, data = wavfile.read(filepath)
            data_parts.append(data)
        
        untrimmed_data = np.concatenate(data_parts).astype(np.float32) / 32767

        if amplify is not None:
            untrimmed_data = (untrimmed_data * amplify)
            


        expected_untrimmed_sample_count = sample_rate * (last_seg_time_stop - first_seg_time_start)
        actual_untrimmed_sample_count = len(untrimmed_data)

        if expected_untrimmed_sample_count != actual_untrimmed_sample_count:
            stretch_factor = expected_untrimmed_sample_count / actual_untrimmed_sample_count

            untrimmed_data = librosa.effects.time_stretch(untrimmed_data, rate=stretch_factor)

        

        begin_skip = int(sample_rate * (revised_time_start - first_seg_time_start))
        end_skip = int(sample_rate * (last_seg_time_stop - revised_time_stop)) or 1

        trimmed_data = untrimmed_data[begin_skip:-end_skip]


        
        trimmed_data = np.clip(trimmed_data, -1, 1)
        trimmed_data = (trimmed_data * 32767).astype(np.int16)



        return trimmed_data, revised_time_start, revised_time_stop
    
    async def create_clip(self, session_id:str, time_start:float, time_stop:float, amplify:float=None) -> Optional[clip]:
        collection = get_collection(self.uri, self.database, self.collection)
        sess_id = ObjectId(session_id)

        query_segments = {
            "type": segment.type_name,
            "properties.time_stop": {"$gt": time_start},
            "properties.time_start": {"$lt": time_stop},
            "properties.session": sess_id
        }

        seg_docs = list(collection.find(query_segments, sort=[('properties.time_start', pymongo.ASCENDING)]))

        if len(seg_docs) == 0:
            return None

        sample_rate = collection.find_one({"_id": sess_id})['properties']['sample_rate']

        clip_data, clip_time_start, clip_time_stop = \
            self._construct_single(seg_docs, sample_rate, time_start, time_stop, amplify)
        
        clip_doc = {
            "type": clip.type_name,
            "properties": {
                "session_id": sess_id,
                "time_start": clip_time_start,
                "time_stop": clip_time_stop
            }
        }

        clip_insert_result = collection.insert_one(clip_doc) ;\
            raise_not_ack(clip_insert_result, INSERT_FAILED_SEGMENT_DOC)

        clip_id:ObjectId = clip_insert_result.inserted_id

        filepath = os.path.join(self.data_dir, f"{clip_id}.wav")

        wavfile.write(filepath, int(sample_rate), clip_data)

        return clip(
            clip_id,
            sess_id,
            time_start,
            time_stop
        )
    
    async def get_timeline_segments(self, timeline:str, time_start:float, time_stop:float) -> List[Tuple[session, List[segment]]]:
        collection = get_collection(self.uri, self.database, self.collection)

        query_segments = {
            "type": segment.type_name,
            "properties.time_stop": {"$gt": time_start},
            "properties.time_start": {"$lt": time_stop},
            "lookup.timeline": timeline # Utilize the optimization
        }

        seg_docs = collection.find(query_segments, sort=[('properties.time_start', pymongo.ASCENDING)])

        sess_id_order = list()
        session_groups = dict()

        for seg in seg_docs:
            sess_id = seg['properties']['session']

            if sess_id in session_groups:
                sess_seg_docs = session_groups[sess_id]
            else:
                sess_seg_docs = session_groups[sess_id] = list()
                sess_id_order.append(sess_id)
            
            sess_seg_docs.append(seg)
        
        result = list()

        for sess_id in sess_id_order:
            sess_doc = collection.find_one({"_id": sess_id})
            sess_seg_docs = session_groups[sess_id]

            sess = session(
                sess_id,
                timeline,
                sess_doc['properties']['timeline'], 
                sess_doc['properties']['sample_rate']
            )

            sess_segments = list()

            for seg in sess_seg_docs:
                sess_segments.append(segment(
                    seg['_id'],
                    sess_id,
                    seg['properties']['time_start'],
                    seg['properties']['time_stop']
                ))

            item = sess, sess_segments
            result.append(item)
        
        return result

    async def get_clip_data(self, clip_id:Union[ObjectId, str]) -> np.ndarray:
        file = os.path.join(self.data_dir, f"{clip_id}.wav")
        
        sr, data = wavfile.read(file)
        
        return data

    async def get_segment_data(self, segment_id:Union[ObjectId, str]) -> np.ndarray:
        file = os.path.join(self.data_dir, f"{segment_id}.wav")
        
        sr, data = wavfile.read(file)
        
        return data

    async def get_object(self, object_id:Union[ObjectId, str]) -> Union[session, segment, clip, None]:
        collection = get_collection(self.uri, self.database, self.collection)

        oid = ObjectId(object_id)
        doc = collection.find_one({'_id': oid})

        if doc is None:
            return None
        
        doc_type = doc['type']

        if doc_type == session.type_name:
            return session(
                oid,
                doc['properties']['timeline'],
                doc['properties']['device'],
                doc['properties']['sample_rate']
            )
        elif doc_type == segment.type_name:
            return segment(
                oid,
                doc['properties']['session_id'],
                doc['properties']['time_start'],
                doc['properties']['time_stop']
            )
        elif doc_type == clip.type_name:
            return clip(
                oid,
                doc['properties']['session_id'],
                doc['properties']['time_start'],
                doc['properties']['time_stop']
            )

def _get_segment_ranges(time_start:float, time_stop:float):
    duration = time_stop - time_start
    segment_size_percent = APPROX_SEGMENT_DURATION / duration

    i = 0
    while True:
        elapsed = (i * APPROX_SEGMENT_DURATION)

        seg_start = time_start + elapsed
        seg_stop = seg_start + APPROX_SEGMENT_DURATION

        percent_start = elapsed / duration
        percent_stop = percent_start + segment_size_percent

        if seg_stop >= time_stop:
            yield seg_start, time_stop, percent_start, 1.0
            break

        yield seg_start, seg_stop, percent_start, percent_stop

        i += 1

def record_file(file:str, data_dir:str, col:Collection, timeline:str, time_start:float, time_stop:float=None) -> Tuple[session, List[segment]]:
    sample_rate, data = wavfile.read(file)



    arlib_device_id = f"file:{os.path.normpath(file)}"

    if time_stop is None:
        time_stop = time_start + (len(data) / sample_rate)

    sess_doc = {
        "type": session.type_name,
        "properties": {
            "timeline": timeline,
            "device": arlib_device_id,
            "sample_rate": sample_rate
        }
    }

    sess_result = col.insert_one(sess_doc); \
        raise_not_ack(sess_result, INSERT_FAILED_SESSION_DOC)

    sess_id = sess_result.inserted_id

    sess = session(sess_id, timeline, arlib_device_id, sample_rate)



    segs = list()

    for seg_start, seg_stop, pct_start, pct_stop in _get_segment_ranges(time_start, time_stop):
        sample_start = pct_start * len(data)
        sample_stop = pct_stop * len(data)

        seg_data = data[sample_start, sample_stop]

        seg_doc = {
            "type": segment.type_name,
            "properties": {
                "session_id": sess_id,
                "time_start": seg_start,
                "time_stop": seg_stop
            },
            "lookup": {
                "timeline": timeline # Optimization, not strictly required
            }
        }

        seg_result = col.insert_one(seg_doc); \
            raise_not_ack(seg_result, INSERT_FAILED_SEGMENT_DOC)

        seg_id = seg_result.inserted_id

        dest_file = os.path.join(data_dir, f"{seg_id}.wav")
        wavfile.write(dest_file, sample_rate, seg_data)

        seg = segment(seg_id, sess_id, seg_start, seg_stop)
        segs.append(seg)
    


    return sess, segs

