from flask import Flask, request, send_file
from flask.views import MethodView
from hparams import hparams, hparams_debug_string
import argparse
import os
from synthesizer import Synthesizer
from flask_cors import CORS
import io

app = Flask(__name__)
CORS(app)

html_body = '''<html><title>Demo</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="40" placeholder="Enter Text">
  <button id="button" name="synthesize">Speak</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = 'Synthesizing...'
    q('#button').disabled = true
    q('#audio').hidden = true
    synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(response.statusText)
      return res.blob()
    }).then(function(blob) {
      q('#message').textContent = ''
      q('#button').disabled = false
      q('#audio').src = URL.createObjectURL(blob)
      q('#audio').hidden = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}
</script></body></html>
'''

synthesizer = Synthesizer()


class Mimic2(MethodView):
    def get(self):
        text = request.args.get('text')
        if text:
            wav, _ = synthesizer.synthesize(text)
            audio = io.BytesIO(wav)
            return send_file(audio, mimetype="audio/wav")


class UI(MethodView):
    def get(self):
        return html_body


ui_view = UI.as_view('ui_view')
app.add_url_rule('/', view_func=ui_view, methods=['GET'])

mimic2_api = Mimic2.as_view('mimic2_api')
app.add_url_rule('/synthesize', view_func=mimic2_api, methods=['GET'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Full path to model checkpoint')
    parser.add_argument('--port', type=int, default=3000)
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument(
        '--gpu_assignment', default='0',
        help='Set the gpu the model should run on')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_assignment
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)
    print(hparams_debug_string())
    synthesizer.load(args.checkpoint)
    app.run(host='0.0.0.0', port=3000)
