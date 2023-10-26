# API Functions
Each function uses POST as the HTTP method, and the response code is (supposed to be) 200.
The request body must be a JSON dict, containing the keyword arguments for the function.
The response is a JSON list of three elements:
1. The exit code
2. The exit message
3. The return value (Can be null)

If the function call is successful, it will return something like this:
```json
[0, "Success", "foo return value"]
```

All time floats represent seconds since epoch.


## /api/start-recording
### Description
Starts recording for the specified timeline. Specifying `device` is only optional if a default device is configured for the specified `timeline`. Otherwise, exception #2 will be raised. If a recorder is already running for the specified timeline, exception #1 will be raised unless `force` is `true`. There is no return value.

### Parameters
- timeline (String) (Required)
- device (String) (Semi-Optional)
- force (Boolean) (Optional)

### Exceptions
1. Aready recording for specified timeline
2. Default device not found for specified timeline

### Example Response
```json
[0, "Success", null]
```


## /api/list-active
### Description
List all of the active recorders.

### Example Response
```json
[0, "Success", [{"timeline":"string", "device":"string"}]]
```


## /api/list-default-devices
### Description
List the default devices for the configured timelines.

### Example Response
```json
[0, "Success", [{"timeline":"string", "device":"string"}]]
```


## /api/stop-recording
### Description
Stop recording for the specified timeline. Returns info about the stopped recorder.

### Parameters
- timeline (String) (Required)

### Exceptions
1. Not recording for specified timeline

### Example Response
```json
[0, "Success", {
    "timeline": "string",
    "device": "string"
}]
```


## /api/create-clip
### Description
Create a clip for the specified session and time range, and amplify it by `amplify` if specified.
This function will return info about the clip.

### Parameters
- session_id (String) (Required)
- time_start (Float) (Required)
- time_stop (Float) (Required)
- amplify (Float) (Optional)

### Exceptions
1. No audio found for the specified session and time range

### Example Response
```json
[0, "Success", {
    "clip_id": "string",
    "session_id": "string",
    "time_start": 123,
    "time_stop": 321
}]
```


## /api/get-timeline-segments
### Description
Get a list of audio segments for the specified timeline and time range.
In Python, the return type is "List[Tuple[session, List[segment]]]".

### Parameters
- timeline (String) (Required)
- time_start (Float) (Required)
- time_stop (Float) (Required)

### Example Response
```json
[0, "Success", [
    [
        {
            "type": "arlib.session",
            "session_id": "string",
            "timeline": "string",
            "device": "string",
            "sample_rate": 123
        },
        [
            {
                "type": "arlib.segment",
                "segment_id": "string",
                "session_id": "string",
                "time_start": 123,
                "time_stop": 321
            }
        ]
    ]
]]
```


## /api/get-object
### Description
Get an object by the specified object ID.
If no object was found, this will return null.

Possible object types returned:
- Session (arlib.session)
- Segment (arlib.segment)
- Clip (arlib.clip)

### Parameters
- object_id (String) (Required)

### Example Response
```json
[0, "Success", [
    {
        "type": "arlib.session",
        "session_id": "string",
        "timeline": "string",
        "device": "string",
        "sample_rate": 123
    }
]]
```

