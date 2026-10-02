import os
import uuid
import cv2

from flask import (
    Flask,
    render_template,
    request,
    url_for
)

from inference_sdk import InferenceHTTPClient


app = Flask(__name__)

# Railway起動時にフォルダ作成
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

# 環境変数
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = os.getenv("MODEL_ID")

CLIENT = InferenceHTTPClient(
api_url="https://detect.roboflow.com",
api_key=API_KEY
)


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

    # 保存
    file.save(input_path)

    # Roboflow推論
    result = CLIENT.infer(
        input_path,
        model_id=MODEL_ID
    )

    # OpenCV読込
    img = cv2.imread(input_path)

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

        # ボックス描画
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        # ラベル
        label = f"{class_name} {confidence:.2f}"

        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imwrite(output_path, img)

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