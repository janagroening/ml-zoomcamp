import pickle
from flask import Flask, request, jsonify

modelfile = 'model1.bin'
dvfile = 'dv.bin'

with open(modelfile, 'rb') as m:
    model = pickle.load(m)

with open(dvfile, 'rb') as d:
    dv = pickle.load(d)


app = Flask('churn')

@app.route("/predict", methods = ["POST"])

def predict():

    client = request.get_json()

    X = dv.transform(client)
    y_pred = model.predict_proba(X)[0, 1]
    y_pred_round = round(y_pred, 3)
    churn = (y_pred_round >= 0.5)

    result = {
        "churn_prob": float(y_pred_round),
        "churn_decision": bool(churn)
    }

    result = jsonify(result)

    return result

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)