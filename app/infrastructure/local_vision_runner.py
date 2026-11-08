"""JSON-lines worker for CNN inference and deterministic local reference lookup."""
from pathlib import Path
import json, os, socket, sys

def _restore_worker_streams():
    """Windowed Windows executables still receive the pipes created by QProcess."""
    if sys.platform != 'win32' or all((sys.stdin, sys.stdout, sys.stderr)):
        return
    import ctypes
    import msvcrt

    get_handle = ctypes.windll.kernel32.GetStdHandle
    get_handle.argtypes = (ctypes.c_ulong,)
    get_handle.restype = ctypes.c_void_p
    for name, number, mode in (('stdin', -10, 'r'), ('stdout', -11, 'w'), ('stderr', -12, 'w')):
        if getattr(sys, name) is not None:
            continue
        handle = get_handle(number & 0xffffffff)
        if not handle or handle == ctypes.c_void_p(-1).value:
            raise OSError(f'Missing pipe for {name}')
        flags = os.O_BINARY | (os.O_RDONLY if mode == 'r' else os.O_WRONLY)
        descriptor = msvcrt.open_osfhandle(handle, flags)
        setattr(sys, name, os.fdopen(descriptor, mode, encoding='utf-8', errors='replace'))

def main():
    _restore_worker_streams()
    for stream in (sys.stdin,sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')
    root=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root))
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    def offline(*args,**kwargs):raise OSError('CV-1 Vision runs offline; network is disabled')
    socket.create_connection=offline;socket.socket.connect=offline;socket.socket.connect_ex=offline
    from app.infrastructure.local_vision import VisionEngine
    engine=VisionEngine(device=os.environ.get('CV1_VISION_DEVICE') or None)
    for line in sys.stdin:
        request={}
        try:
            if len(line)>2_000_000:raise ValueError('Слишком большой запрос')
            request=json.loads(line)
            if not isinstance(request,dict):raise ValueError('Ожидается объект запроса')
            result=engine.answer(request)
            result['id']=request.get('id')
        except Exception as exc:result={'id':request.get('id') if isinstance(request,dict) else None,'ok':False,'error':f'{type(exc).__name__}: {exc}'}
        print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
