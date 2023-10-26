import argparse
from . import rec, web, util
from aiohttp import web as aioweb



parser = argparse.ArgumentParser()

parser.add_argument('data_dir')
parser.add_argument('mongo_uri')
parser.add_argument('mongo_database')
parser.add_argument('mongo_collection')

parser.add_argument('--http-host', default='localhost')
parser.add_argument('--http-port', type=int, default=80)

parser.add_argument('--device', action='append', nargs=2)
parser.add_argument('--start', action='append')



if __name__ == '__main__':
    args = parser.parse_args()
    
    
    
    lib = rec.library(args.mongo_uri, args.mongo_database, args.mongo_collection, args.data_dir)



    timeline:str
    device:str

    devices = dict()

    for timeline, device in args.device or []:
        devices[timeline] = device



    timeline:str
    device:str

    async def start_recording(app):
        for timeline in args.start or []:
            device = devices[timeline]
            await lib.start(timeline, device)



    app = web.create_aiohttp_app(lib, devices)

    app.on_startup.append(start_recording)

    aioweb.run_app(app, host=args.http_host, port=args.http_port)
