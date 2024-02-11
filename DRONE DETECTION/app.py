from flask import Flask, render_template, url_for, request
import cv2
import time
import numpy as np
import os

app = Flask(__name__, static_url_path="/static")


# Home page
@app.route("/")
def index():
    return render_template("index.html")


# SignUp page
@app.route("/signup")
def signup():
    return render_template("Signup.html")


# Login page
@app.route("/login")
def login():
    return render_template("Login.html")


@app.route("/detect")
def detect():
    return render_template("detect.html")


@app.route("/contact_us")
def contact_us():
    return render_template("contactus.html")


@app.route("/result", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        value1 = "Drone Present"
        vedio_file = request.files["videoFile"]
        image_file = request.files["imageFile"]
        value2 = "No Drone Present"

        if vedio_file:
            bool1 = False
            # vedio_path = "uploads/" + vedio_file.filename
            # vedio_file.save(vedio_path)
            # print(vedio_path)
            upload_folder = r"./upload"
            os.makedirs(upload_folder, exist_ok=True)
            video_path = os.path.join(upload_folder, vedio_file.filename)
            vedio_file.save(video_path)
            # output_folder = "path/to/output/folder"
            # os.makedirs(output_folder, exist_ok=True)
            # output_video_path = os.path.join(output_folder, "output_video.mp4")
            nm, b_val = processvedio(video_path)
            print(nm)
            bool2 = False
            nm = nm.replace("%20", " ")
            if b_val:
                return render_template(
                    "Result.html",
                    nm2=nm,
                    value1="Drone Detected !!!",
                    bool2=True,
                )
            else:
                return render_template(
                    "Result.html", nm2=nm, value2="No Drone Detected !!!", bool2=False
                )
        elif image_file:
            bool1 = False
            upload_folder = r"./upload1"
            os.makedirs(upload_folder, exist_ok=True)
            img_path = os.path.join(upload_folder, image_file.filename)
            image_file.save(img_path)
            print(image_file)
            nm, b_val = process_image(img_path)
            nm = nm.replace("%20", " ")
            print(nm)
            if b_val:
                return render_template(
                    "Result.html",
                    nm1=nm,
                    value1="Drone Detected !!!",
                    bool1=True,
                )
            else:
                return render_template(
                    "Result.html", nm1=nm, value2="No Drone Exists !!!", bool1=False
                )


def processvedio(vedio_path):
    model = tf.keras.models.load_model(
        r""
    )  # H5 model   drone  CNN Model
    # Load drone classifiers
    drone_classifier1 = cv2.CascadeClassifier(
        r""
    )  # XML classifier 1   drone
    drone_classifier2 = cv2.CascadeClassifier(
        r""
    )  # XML classifier 2   nd
    drone_classifier3 = cv2.CascadeClassifier(
        r""
    )  # XML classifier 3    drone
    drone_classifier4 = cv2.CascadeClassifier(
        r""
    )  # XML classifier 4    d
    s = ""
    vedio_filename = os.path.splitext(os.path.basename(vedio_path))[0]
    if vedio_path.lower().endswith((".mp4", ".avi", ".mov")):
        cap = cv2.VideoCapture(vedio_path)
    elif vedio_path.lower() == "0":  # '0' represents the default webcam
        cap = cv2.VideoCapture(0)
    else:
        # Handle invalid input, e.g., unsupported file format or invalid webcam index
        return "Invalid input for video"

    if not cap.isOpened():
        return "Error opening video capture"

    # Initiate video capture for video file

    x, y = 0, 0
    ans = False
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"H264")  # H264  is not loding

    out = cv2.VideoWriter(
        os.path.join("static/output1", f"{vedio_filename}_output_video.mp4"),
        fourcc,
        10,
        (frame_width, frame_height),
    )
    detected_frames = []

    # Loop once video is successfully loaded
    print("Loading vedio")

    while cap.isOpened():
        # Read the frame
        print("The loop entered")
        ret, frame = cap.read()

        if not ret:
            print("End of Frame  ...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass frame to the drone classifiers
        drones1 = drone_classifier1.detectMultiScale(gray, 1.4, 1)
        drones2 = drone_classifier2.detectMultiScale(gray, 1.4, 1)
        drones3 = drone_classifier3.detectMultiScale(gray, 1.4, 1)
        drones4 = drone_classifier4.detectMultiScale(gray, 1.4, 1)

        # Preprocess the frame for the trained model
        frame_preprocessed = cv2.resize(frame, (258, 258))
        frame_preprocessed = np.array(frame_preprocessed, dtype="float32") / 255.0
        frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)

        # Make a prediction using the trained model
        startX, startY, endX, endY = model.predict(frame_preprocessed)[0]
        print("Processing")

        # Scale the predicted bounding box coordinates based on the frame's size
        startX *= frame.shape[1]
        startY *= frame.shape[0]
        endX *= frame.shape[1]
        endY *= frame.shape[0]

        # Keep track of models predicting a drone
        drone_models = []

        # Check if each classifier predicts a drone
        if len(drones1) > 0:
            drone_models.append((drones1, "Model 1"))
        if len(drones2) > 0:
            drone_models.append((drones2, "Model 2"))
        if len(drones3) > 0:
            drone_models.append((drones3, "Model 3"))
        if len(drones4) > 0:
            drone_models.append((drones4, "Model 4"))

        # If at least two models predict a drone, calculate average coordinates
        if len(drone_models) >= 2:
            avg_startX, avg_startY, avg_endX, avg_endY = 0, 0, 0, 0

            for drones, model_name in drone_models:
                avg_startX += np.mean(drones[:, 0])
                avg_startY += np.mean(drones[:, 1])
                avg_endX += np.mean(drones[:, 0] + drones[:, 2])
                avg_endY += np.mean(drones[:, 1] + drones[:, 3])

            avg_startX /= len(drone_models)
            avg_startY /= len(drone_models)
            avg_endX /= len(drone_models)
            avg_endY /= len(drone_models)

            # Draw the bounding box on the frame in red color
            padding = 0  # You can adjust this value
            cv2.putText(
                frame,
                "Drone",
                (int(avg_startX), int(avg_startY) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.rectangle(
                frame,
                (int(avg_startX) - padding, int(avg_startY) - padding),
                (int(avg_endX) + padding, int(avg_endY) + padding),
                (0, 0, 255),
                2,
            )
            ans = True

            # Display the resulting frame
            # cv2.imshow("frame", frame)
            s = "Drone is Not Detected ...!"
            detected_frames.append(frame)
            out.write(frame)
        else:
            x += 1

        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    if x != 0 and ans == False:
        s = "Drone is Not Detected ...!"
        b_val = False
    if ans:
        s = "There is possibility of Drone !!!"
        b_val = True

    for frame in detected_frames:
        out.write(frame)
    out_path = os.path.join("static/output1", f"{vedio_filename}_output_video.mp4")
    cap.release()
    cv2.destroyAllWindows()
    return out_path, b_val


def process_image(image_path):
    model = tf.keras.models.load_model(
        r"./Model.h5"
    )  # H5 model

    # Load drone classifiers
    drone_classifier1 = cv2.CascadeClassifier(
        r"./drones.xml"
    )
    drone_classifier2 = cv2.CascadeClassifier(
        r""
    )  # XML classifier 2
    drone_classifier3 = cv2.CascadeClassifier(
        r""
    )  # XML classifier 3
    drone_classifier4 = cv2.CascadeClassifier(
        r""
    )  # XML classifier 4
    drone_classifier5 = cv2.CascadeClassifier(
        r""
    )  # XML classifier 5
    drone_classifier6 = cv2.CascadeClassifier(
        r""
    )  # XML classifier 6
    drone_classifier7 = cv2.CascadeClassifier(
        r""
    )
    drone_classifier8 = cv2.CascadeClassifier(
        r""
    )

    # Load bird classifier
    bird_classifier = cv2.CascadeClassifier(
        r""
    )  # XML classifier for birds

    # Read the input image
    img = cv2.imread(image_path)

    # Resize the image to a smaller size
    small_img = cv2.resize(img, (400, 300))  # Adjust the size as needed

    gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

    var = False

    # Initiate a loop for simulating video frames
    for _ in range(20):  # Adjust the number of frames as needed
        # Pass frame to the drone classifiers
        drones1 = drone_classifier1.detectMultiScale(gray, 1.4, 1)
        drones2 = drone_classifier2.detectMultiScale(gray, 1.4, 1)
        drones3 = drone_classifier3.detectMultiScale(gray, 1.4, 1)
        drones4 = drone_classifier4.detectMultiScale(gray, 1.4, 1)
        drones5 = drone_classifier5.detectMultiScale(gray, 1.4, 1)
        drones6 = drone_classifier6.detectMultiScale(gray, 1.4, 1)
        drones7 = drone_classifier7.detectMultiScale(gray, 1.4, 1)
        drones8 = drone_classifier8.detectMultiScale(gray, 1.4, 1)

        if var:
            break

        # Pass frame to the bird classifier
        birds = bird_classifier.detectMultiScale(gray, 1.4, 1)

        # Preprocess the frame for the trained model
        frame_preprocessed = cv2.resize(small_img, (258, 258))
        frame_preprocessed = np.array(frame_preprocessed, dtype="float32") / 255.0
        frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)

        # Make a prediction using the trained model
        startX, startY, endX, endY = model.predict(frame_preprocessed)[0]

        # Scale the predicted bounding box coordinates based on the frame's size
        startX *= small_img.shape[1]
        startY *= small_img.shape[0]
        endX *= small_img.shape[1]
        endY *= small_img.shape[0]

        # Keep track of models predicting a drone or bird
        drone_models = []
        bird_models = []

        # Check if each drone classifier predicts a drone
        if len(drones1) > 0:
            drone_models.append((drones1, "Model 1"))
        if len(drones2) > 0:
            drone_models.append((drones2, "Model 2"))
        if len(drones3) > 0:
            drone_models.append((drones3, "Model 3"))
        if len(drones4) > 0:
            drone_models.append((drones4, "Model 4"))
        if len(drones5) > 0:
            drone_models.append((drones5, "Model 5"))
        if len(drones6) > 0:
            drone_models.append((drones6, "Model 6"))
        if len(drones7) > 0:
            drone_models.append((drones7, "Model 7"))
        if len(drones8) > 0:
            drone_models.append((drones8, "Model 8"))

        # Check if the bird classifier predicts a bird
        if len(birds) > 0:
            bird_models.append(birds)

        # If at least two models predict a drone and the bird classifier does not predict a bird, calculate average coordinates
        if len(drone_models) >= 2 and len(bird_models) == 0:
            avg_startX, avg_startY, avg_endX, avg_endY = 0, 0, 0, 0
            var = True
            for drones, model_name in drone_models:
                avg_startX += np.mean(drones[:, 0])
                avg_startY += np.mean(drones[:, 1])
                avg_endX += np.mean(drones[:, 0] + drones[:, 2])
                avg_endY += np.mean(drones[:, 1] + drones[:, 3])

            avg_startX /= len(drone_models)
            avg_startY /= len(drone_models)
            avg_endX /= len(drone_models)
            avg_endY /= len(drone_models)

            # Draw the bounding box on the frame in red color
            padding = 0  # You can adjust this value
            cv2.putText(
                small_img,
                "Drone",
                (int(avg_startX), int(avg_startY) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.rectangle(
                small_img,
                (int(avg_startX) - padding, int(avg_startY) - padding),
                (int(avg_endX) + padding, int(avg_endY) + padding),
                (0, 0, 255),
                2,
            )
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            # Display the resulting image
            output_img_path = os.path.join(
                "static\output", f"{base_filename}_result.jpg"
            )  # Adjust the path as needed
            cv2.imwrite(output_img_path, small_img)

            # Display the resulting image pputath and boolean value indicating detection
            return output_img_path, True
            # cv2.imshow("Output Image", small_img)
            # cv2.waitKey(8000)  # Adjust the delay as needed

    if var:
        return "Drone Detected !!!", True
    else:
        return image_path, False


@app.route("/home")
def home():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
