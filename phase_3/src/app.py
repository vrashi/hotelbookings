from flask import Flask, render_template, request
from main import dataCleaningCallingTrain

app = Flask(__name__)
app.debug = True

@app.route("/")
def index():
    return render_template("home.html")
    
@app.route('/', methods=['POST'])
def requestResults():
    text = request.form['sname']
    resultFor = text.upper()
    print(resultFor)
    return predict(resultFor)

def predict(modelName):
    modelData=dataCleaningCallingTrain(modelName)
    if modelData[0] == 'EDA':
        return render_template("EDA.html", modelName = modelName, modelData=modelData)
    else:
        return render_template("training.html", modelName = modelName, modelData=modelData)


if __name__ == "__main__":
    app.run()
