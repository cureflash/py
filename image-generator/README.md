# Local Image Generator

既存の画像生成モデル `Qwen/Qwen-Image` をHugging Face Diffusersから読み込み、ブラウザUIで画像を生成する最小構成です。

## 必要環境

- Python 3.10以上
- NVIDIA GPU / CUDA
- 十分なRAM・ストレージ

`Qwen/Qwen-Image` は大きなモデルです。標準では `LOW_VRAM=1` としてCPU offloadとVAE tilingを使います。そのぶん生成は遅くなります。

## セットアップ

```bash
cd image-generator
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

起動後、ブラウザで `http://127.0.0.1:7860` を開きます。

初回起動時にモデルがHugging Faceからダウンロードされます。

## 設定

別のQwen-Image互換モデルを使う場合:

Windows PowerShell:

```powershell
$env:MODEL_ID="Qwen/Qwen-Image"
python app.py
```

CPU offloadを無効化してGPUへ直接載せる場合:

```powershell
$env:LOW_VRAM="0"
python app.py
```

## UI

- Prompt
- Negative prompt
- Width / Height
- Steps
- CFG scale
- Seed
- PNGプレビュー

Seedを `-1` にすると毎回ランダムになります。
