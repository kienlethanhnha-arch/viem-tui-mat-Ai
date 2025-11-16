import streamlit as st
import pandas as pd
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, render_template_string
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import pickle
import os

MODEL_PATH = '/mnt/data/mo_hinh_du_doan_hoai_tu.pkl'
app = Flask(__name__)

FEATURE_ORDER = [
    'wbc','crp','wall_thickened_1.0','wall_thickened_0.0','age','nlr',
    'alt','systolic_bp','bilirubin_total','neutrophil_pct'
]

# Load model
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    print("⚠️ MODEL_PATH không tồn tại:", MODEL_PATH)

# HTML form
HTML_PAGE = """
<!doctype html>
<html lang='vi'>
<head>
<meta charset='utf-8'/>
<title>Dự đoán hoại tử túi mật</title>
<style>
body{font-family:Arial;max-width:600px;margin:40px auto}
input{width:100%;padding:8px;margin-top:6px}
button{padding:10px;margin-top:15px}
#out{margin-top:20px;padding:10px;background:#eee;display:none}
</style>
</head>
<body>
<h2>Tính tỉ lệ hoại tử túi mật</h2>
<form id='frm'>
{% for f in fields %}
<label>{{loop.index}}. {{f}}</label>
<input name='{{f}}' required type='number' step='any'/>
{% endfor %}
<button type='submit'>Predict</button>
</form>
<div id='out'></div>
<script>
const frm=document.getElementById('frm');
const out=document.getElementById('out');
frm.addEventListener('submit',async e=>{
 e.preventDefault(); out.style.display='block'; out.innerHTML='Đang tính...';
 let data={};
 [...new FormData(frm).entries()].forEach(([k,v])=>data[k]=Number(v));
 let r=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
 r=await r.json();
 if(r.error){out.innerHTML='Lỗi: '+r.error;return;}
 out.innerHTML=`<b>Kết quả:</b> ${r.probability_percent.toFixed(2)}%`;
});
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE, fields=FEATURE_ORDER)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model không tải được'}), 500

    data = request.get_json()
    try:
        row = [data[f] for f in FEATURE_ORDER]
    except:
        return jsonify({'error': 'Thiếu hoặc sai tên tham số'}), 400

    X = pd.DataFrame([row], columns=FEATURE_ORDER).apply(pd.to_numeric, errors='coerce')

    if X.isnull().any().any():
        return jsonify({'error': 'Giá trị không hợp lệ'}), 400

    if hasattr(model, 'predict_proba'):
        p = model.predict_proba(X)[:, 1][0]
    else:
        p = float(model.predict(X)[0])

    return jsonify({
        'probability': float(p),
        'probability_percent': float(p)*100
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
