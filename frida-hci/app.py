import os
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ['CURL_CA_BUNDLE'] = 'G:/huggingface.co.crt'
# # os.environ['CURL_CA_BUNDLE'] = ''
# # os.environ['REQUESTS_CA_BUNDLE'] = ''
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from flask import Flask, send_from_directory, jsonify, request, redirect
from torchvision.utils import save_image
from PIL import Image
import io
import base64

### IMPORTS FROM PLAN.PY FOR PRELOADING ###
import os
os.chdir("../src")
import sys
sys.path.append('../src/')
sys.path.append('../frida-hci/')
import api
from api import api_adapt, api_manifest

import os


from painting import *


from paint_utils3 import *
from torchvision.io import write_jpeg
###

app = Flask(__name__)
rootdir = '../frida-hci/public'

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != '' and not '.' in path:
        path = path + '.html'

    if path != "" and path != "css" and os.path.exists(f"{rootdir}/" + path):
        return send_from_directory(rootdir, path)
    elif path == "":
        return send_from_directory(rootdir, 'index.html')
    else:
        return send_from_directory(rootdir, '404.html'), 404
    
@app.route('/manifest')
def manifest():
    prompt = request.args.get('prompt')
    iter = request.args.get('iter')
    api_manifest(prompt, iter)
    return f"done rendering {iter}"

@app.route('/select-objective')
def select_objective():
    objective_id = request.args.get('objective-id')
    img = Image.open(f'{rootdir}/data/stable/{objective_id}.jpg')
    img.save(f'{rootdir}/data/objectives/user.jpg')
    return f"done rendering"

@app.route('/think')
def think():
    task = request.args.get('task')
    poll = request.args.get('poll')
    counter = int(request.args.get('counter'))

    if counter == 0 and task == 'user' and os.path.exists(f'src/caches/cache_6_6_cvpr/[user]_next_brush_strokes.csv'):
        os.remove(f'src/caches/cache_6_6_cvpr/[user]_next_brush_strokes.csv')

    if poll == '1':
        plan_f = os.path.join('src/caches/cache_6_6_cvpr', f"[{task}]_next_brush_strokes.csv")
        return jsonify({'plan_exists': os.path.exists(plan_f)})
    else:
        is_first = request.args.get('is-first') == 'true'
        prompt = request.args.get('prompt')

        global images
        images = api_adapt(task, is_first, counter, prompt)
        return jsonify({'n_strokes': len(images)})

@app.route('/render')
def render():
    counter = int(request.args.get('counter'))

    write_jpeg(images[counter].mul(255).clamp(0, 255).byte()[0], f'{rootdir}/data/camera.jpg', quality=100)
    return f"done rendering {counter}"

@app.route('/save_camera', methods=['POST'])
def save_camera():
    # Get the image data from the request
    image_data = request.form['image_data']

    image = Image.open(io.BytesIO(base64.decodebytes(str.encode(image_data.replace('data:image/jpeg;base64,', '')))))
    # Save the image as a JPG file
    image_path = f'{rootdir}/data/camera.jpg'  # Specify the path to save the image
    image.save(image_path, quality=100)
    return "saved"

if __name__ == '__main__':
    app.run()