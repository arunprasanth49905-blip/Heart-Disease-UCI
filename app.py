from flask import Flask,render_template,request
import pickle
import numpy as np


app = Flask(__name__)



model = pickle.load(
    open(
        "model.pkl",
        "rb"
    )
)



@app.route("/")
def home():

    return render_template(
        "index.html"
    )





@app.route("/predict",
methods=["POST"])

def predict():


    data=[


    float(request.form["age"]),

    float(request.form["sex"]),

    float(request.form["cp"]),

    float(request.form["trestbps"]),

    float(request.form["chol"]),

    float(request.form["fbs"]),

    float(request.form["restecg"]),

    float(request.form["thalach"]),

    float(request.form["exang"]),

    float(request.form["oldpeak"]),

    float(request.form["slope"]),

    float(request.form["ca"]),

    float(request.form["thal"])


    ]



    input_data=np.array(
        data
    ).reshape(
        1,-1
    )



    result=model.predict(
        input_data
    )



    probability=model.predict_proba(
        input_data
    )


    risk=round(
        probability[0][1]*100,
        2
    )



    if result[0]==1:

        prediction="Heart Disease Detected"

        status="HIGH RISK"

    else:

        prediction="No Heart Disease Risk"

        status="LOW RISK"



    return render_template(

        "result.html",

        prediction=prediction,

        risk=risk,

        status=status

    )





if __name__=="__main__":

    app.run(debug=True)
