from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_pipeline.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get input values from form
            crop = float(request.form['item_ag'])
            production = float(request.form['production'])
            cpi = float(request.form['cpi'])
            food_inflation = float(request.form['food_inflation'])
            temperature = float(request.form['temperature'])
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['food_inflation'])
            fungicides = float(request.form['fungicides'])
            herbicides = float(request.form['herbicides'])
            insecticides = float(request.form['insecticides'])
            rodenticides = float(request.form['rodenticides'])

            # Predict using model
            prediction = model.predict([[crop,
                                         production, 
                                         cpi, 
                                         food_inflation,
                                         temperature,
                                         nitrogen,
                                         phosphorus,
                                         potassium,
                                         fungicides,
                                         herbicides,
                                         insecticides,
                                         rodenticides]])
            # Square predicition due to squared transformation on original data
            loss_prediction = np.power(round(prediction[0], 2), 2)

            formatted_pred = "{:,.2f}".format(loss_prediction)

            return render_template('index.html', prediction=formatted_pred)
        except Exception as e:
            return f"Error: {str(e)}"

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)