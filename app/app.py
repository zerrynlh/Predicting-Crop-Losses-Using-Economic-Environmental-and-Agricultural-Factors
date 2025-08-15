from flask import Flask, render_template, request
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import io
import base64

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

@app.route('/overtheyears', methods=['GET', 'POST'])
def oty():
    if request.method == 'POST':
        try:
            country = request.form['country_oty']

            data = pd.read_csv('data/merged_data.csv')
            data = data[data['Area_FB'] == country]
            data = data.groupby('YEAR', as_index=False)['Losses'].sum().rename(columns={'Losses': 'SUM'})

            fig, ax = plt.subplots(figsize=(15, 15))
            ax.plot(data['YEAR'].tolist(), data['SUM'].tolist())
            ax.set_xlabel('Year')
            ax.set_xticks(data['YEAR'].tolist())
            ax.set_ylabel('Food Loss (tonnes)')
            ax.set_title(f'Food Loss 2010 to 2022 - {country}')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=70)
            plt.close(fig)
            buf.seek(0)

            plot_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

            return render_template('overtheyears.html', plot_uri=plot_uri)

        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('overtheyears.html', plot_uri=None)

@app.route('/corr', methods=['GET', 'POST'])
def corr():
    if request.method == 'POST':
        try:
            data = pd.read_csv('data/merged_data.csv')
            data = data.iloc[:, 6:].corr()

            fig, ax = plt.subplots(figsize=(15, 15))
            sns.heatmap(data, ax=ax, annot=True, cmap='BuPu', center=0, vmin=-1, vmax=1)
            ax.set_title('Correlation Heatmap of All Variables')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            fig.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=70)
            plt.close(fig)
            buf.seek(0)

            plot_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

            return render_template('corr.html', plot_uri=plot_uri)

        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('corr.html', plot_uri=None)

@app.route('/total', methods=['GET', 'POST'])
def avg():
    if request.method == 'POST':
        try:
            country = request.form['country_total']

            data = pd.read_csv('data/merged_data.csv')
            data = data[data['Area_FB'] == country]

            new_total = round(data['Losses'].sum(), 2)
            total = "{:,.2f}".format(new_total)

            fig, ax = plt.subplots(figsize=(15, 15))

            bars = ax.bar(country, new_total)
            ax.bar_label(bars, labels=[total], padding=3)
            ax.set_xlabel('Country')
            ax.set_ylabel('Food Loss (tonnes)')
            ax.set_title(f'Total Food Loss Since 2010 for {country}')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=70)
            plt.close(fig)
            buf.seek(0)

            plot_uri = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

            return render_template('total.html', plot_uri=plot_uri)

        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('total.html', plot_uri=None)

if __name__ == '__main__':
    app.run(debug=True)