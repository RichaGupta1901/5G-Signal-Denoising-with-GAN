from flask import Flask, render_template, request, send_file, redirect, url_for
import torch
import numpy as np
import h5py
from io import BytesIO
from model import Generator
import plotly.graph_objs as go
import base64
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = Generator()
model.load_state_dict(torch.load("generator_denoising_5g.pth", map_location=torch.device('cpu')))
model.eval()

def load_signal(file_path, ext):
    if ext == ".h5":
        with h5py.File(file_path, 'r') as f:
            data = list(f.values())[0][()]  # first dataset
    else:
        data = np.load(file_path)

    if data.ndim == 1 and data.shape[0] == 2048:
        data = data.reshape(1024, 2)
    elif data.shape != (1024, 2):
        raise ValueError("Expected shape (1024, 2)")

    return data

def preprocess_signal(signal_2d):
    signal = signal_2d.T
    signal = (signal - signal.mean()) / signal.std()
    signal = np.clip(signal, -1, 1)
    tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
    return tensor

def denoise_signal(tensor):
    with torch.no_grad():
        denoised = model(tensor).squeeze().numpy()
    return denoised.T

def plot_to_html(data, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data[:, 0], mode='lines', name='I', line=dict(color='#38b1ff')))
    fig.add_trace(go.Scatter(y=data[:, 1], mode='lines', name='Q', line=dict(color='#6843ff')))
    fig.update_layout(title=title, xaxis_title="Sample Index", yaxis_title="Amplitude",
                      template="plotly_dark", height=400)
    return fig.to_html(full_html=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_noisy, plot_denoised, npy_b64, h5_b64, error = None, None, None, None, None

    if request.method == 'POST':
        file = request.files['signal']
        if file and (file.filename.endswith('.npy') or file.filename.endswith('.h5')):
            ext = os.path.splitext(file.filename)[1]
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            try:
                signal = load_signal(file_path, ext)
                tensor = preprocess_signal(signal)
                denoised = denoise_signal(tensor)

                # Plots
                plot_noisy = plot_to_html(signal, "Original (Noisy) Signal")
                plot_denoised = plot_to_html(denoised, "Denoised Signal")

                # .npy
                npy_bytes = BytesIO()
                np.save(npy_bytes, denoised.astype(np.float32))
                npy_b64 = base64.b64encode(npy_bytes.getvalue()).decode()

                # .h5
                h5_bytes = BytesIO()
                with h5py.File(h5_bytes, 'w') as f:
                    f.create_dataset("denoised_signal", data=denoised.astype(np.float32))
                h5_bytes.seek(0)
                h5_b64 = base64.b64encode(h5_bytes.read()).decode()

            except Exception as e:
                error = str(e)

    return render_template('index.html', plot_noisy=plot_noisy, plot_denoised=plot_denoised,
                           npy_b64=npy_b64, h5_b64=h5_b64, error=error)

if __name__ == '__main__':
    app.run(debug=True)
