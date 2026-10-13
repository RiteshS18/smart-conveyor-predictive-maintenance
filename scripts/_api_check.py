"""Quick API smoke-test against a running server on localhost:8000."""
import urllib.request, json, time

base = 'http://localhost:8000'

def get(path):
    r = urllib.request.urlopen(base + path)
    return json.loads(r.read())

def post(path, data=None):
    body = json.dumps(data).encode() if data is not None else b'{}'
    req = urllib.request.Request(
        base + path, body,
        {'Content-Type': 'application/json'}, method='POST')
    r = urllib.request.urlopen(req)
    return json.loads(r.read())

def delete(path):
    req = urllib.request.Request(base + path, method='DELETE')
    return urllib.request.urlopen(req)

print('=== API Spot Checks ===')

s = get('/api/settings')
sim_en = s['simulator_enabled']
stream_url = s['stream_url']
print(f'[1] GET /api/settings      OK  simulator_enabled={sim_en}  stream_url={stream_url!r}')

st = get('/api/state')
score = st['health']['score']
running = st['stream']['running']
print(f'[2] GET /api/state         OK  health={score}  stream_running={running}')

r = get('/api/predict/risk')
ready = r['ready']
risk = r.get('risk_probability')
print(f'[3] GET /api/predict/risk  OK  ready={ready}  risk={risk}')

a = get('/api/alerts')
print(f'[4] GET /api/alerts        OK  count={len(a)}')

h = get('/api/sensors/history')
print(f'[5] GET /api/sensors/history  OK  points={len(h["ts"])}')

ph = get('/api/predict/history')
print(f'[6] GET /api/predict/history  OK  points={len(ph["ts"])}')

m = get('/api/models')
demo = m['demo_mode']
n_models = len(m['models'])
print(f'[7] GET /api/models        OK  demo_mode={demo}  trained_models={n_models}')

di = get('/api/dataset/images')
print(f'[8] GET /api/dataset/images  OK  images={len(di["images"])}  labeled={di["labeled"]}')

ts_info = get('/api/train/status')
print(f'[9] GET /api/train/status  OK  state={ts_info["state"]}')

# simulator toggle (on then off)
sim1 = post('/api/simulator/toggle', {})
sim2 = post('/api/simulator/toggle', {})
print(f'[10] POST /api/simulator/toggle  OK  toggled on={sim1["enabled"]} then off={sim2["enabled"]}')

# connect stream
cv = post('/api/stream/connect',
          {'url': 'C:/Users/rites/Downloads/conveyor-monitor/conveyor-monitor/test_belt.avi'})
print(f'[11] POST /api/stream/connect  OK  running={cv["running"]}')
time.sleep(4)

ss = get('/api/stream/status')
print(f'[12] GET /api/stream/status  OK  running={ss["running"]}  connected={ss["connected"]}  fps={ss["fps"]}')

# capture a frame
try:
    cap = post('/api/dataset/capture')
    name = cap['name']
    print(f'[13] POST /api/dataset/capture  OK  name={name}')
    delete('/api/dataset/images/' + name)
    print(f'[14] DELETE /api/dataset/images  OK')
except Exception as e:
    print(f'[13/14] capture/delete  FAIL  {e}')

# train start should error (no labeled images)
try:
    post('/api/train/start', {'base_model':'yolov8n','epochs':1,'imgsz':640,'batch':8,'data_source':'labeled'})
    print('[15] POST /api/train/start  UNEXPECTED OK (should have errored)')
except urllib.error.HTTPError as e:
    body = json.loads(e.read())
    msg = body.get('detail', '')
    print(f'[15] POST /api/train/start  OK (expected error): {msg}')

# disconnect
dc = post('/api/stream/disconnect')
print(f'[16] POST /api/stream/disconnect  OK  running={dc["running"]}')

print()
print('All checks complete.')
