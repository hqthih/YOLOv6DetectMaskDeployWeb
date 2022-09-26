import cv2
from flask import Flask, render_template
from flask_cors import CORS, cross_origin
from flask import request
from random import random
import my_yolov6
import os
import pickle

#KHoi tao Flask Server backend
app = Flask(__name__)


#Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = 'static'

my_model = my_yolov6.my_yolov6('weights/best_ckpt.pt', "cpu" ,"data/mydataset.yaml",640, True)

@app.route('/',methods=['GET','POST'])               # POST: input vao bang data duoc gui den tu front-end (co the dung POSTMAN), GET: input qua url hay duong link
#@cross_origin(origin='*')
def home_page():
    #Neu la POST (gui file)
    if request.method == "POST":
        try:
            n_object = 0
            image = request.files['file']
            if image:
                #Luu file
                path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
                print("Save = ",path_to_save)
                image.save(path_to_save)

                frame = cv2.imread(path_to_save)
                #Nhan dien qua model YOLOv6
                frame, n_object = my_model.infer(frame)
                if n_object > 0:
                    cv2.imwrite(path_to_save, frame)
                    return render_template("index.html", user_image = image.filename, rand = str(random()),
                                           msg = "tải file lên thành công", ndet= n_object)
                else:
                    return render_template("index.html", msg="không nhận diện được vật thể",ndet= n_object)
            return render_template("index.html", msg="Hãy chọn file để tải lên",ndet= n_object)
        except Exception as ex:
            print(ex)
            return render_template("index.html", msg="Không nhận diện được vật thể",ndet= n_object)
    else:
        return render_template("index.html")

#Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True  )