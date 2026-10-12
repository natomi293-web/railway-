import os
import uuid
import base64
import requests

from PIL import Image, ImageDraw, ImageOps
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_RESULTS = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_RESULTS, exist_ok=True)

API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")

if not API_KEY:
	raise ValueError("ROBOFLOW_API_KEY が設定されていません")

if not MODEL_ID:
	raise ValueError("MODEL_ID が設定されていません")


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

	if "image" not in request.files:
		return "画像が選択されていません"

	file = request.files["image"]

	if file.filename == "":
		return "画像が選択されていません"

	unique_id = str(uuid.uuid4())

	input_path = os.path.join(
	UPLOAD_FOLDER,
	f"{unique_id}.jpg"
	)

	output_filename = f"{unique_id}_result.jpg"

	output_path = os.path.join(
	STATIC_RESULTS,
	output_filename
	)

	file.save(input_path)

	img = ImageOps.exif_transpose(Image.open(input_path))

	with open(input_path, "rb") as image_file:
		image_bytes = image_file.read()

	image_base64 = base64.b64encode(image_bytes).decode("utf-8")

	url = f"https://detect.roboflow.com/{MODEL_ID}"

	response = requests.post(
		url,
		params={
			"api_key": API_KEY
		},
		data=image_base64,
		headers={
			"Content-Type": "application/x-www-form-urlencoded"
		}
	)

	result = response.json()

	print(result)
	
	img = Image.open(input_path)
	draw = ImageDraw.Draw(img)

	predictions = result.get("predictions", [])

	for pred in predictions:

		x = int(pred["x"])
		y = int(pred["y"])
		w = int(pred["width"])
		h = int(pred["height"])

		class_name = pred["class"]
		confidence = pred["confidence"]

		x1 = int(x - w / 2)
		y1 = int(y - h / 2)

		x2 = int(x + w / 2)
		y2 = int(y + h / 2)

		draw.rectangle(
			[(x1, y1), (x2, y2)],
			outline="lime",
			width=3
		)

		draw.text(
				(x1, max(0, y1 - 20)),
				f"{class_name} {confidence:.2f}",
				fill="lime"
		)

	img.save(output_path)

	return render_template(
		"result.html",
		image_url=url_for(
			"static",

		filename=f"results/{output_filename}"
		),
		detections=len(predictions)
		)


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5000))

	app.run(
		host="0.0.0.0",
		port=port
	)