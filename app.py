from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.joinpath('src').resolve()))

from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('performance.html')
    else:
        # Read all the data from the performance.html form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        # Convert the input form data to a dataframe
        pred_df = data.get_data_as_df()

        # Call the prediction function to get the predictions
        pred_pipeline = PredictPipeline()
        results = pred_pipeline.predict(pred_df)
        results = round(results[0], 2)

        # Return the results to form
        return render_template('performance.html', results=results)
    

if __name__ == '__main__':
    app.run(host='0.0.0.0')