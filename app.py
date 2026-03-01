from flask import Flask, render_template, request,url_for
import os 
import numpy as np
import pandas as pd
import joblib 
from src.IPL.pipeline.prediction import PredictionPipeline
from pathlib import Path
app = Flask(__name__)

team_image_map = {
    'Chennai Super Kings': 'chennai_super_kings.png',
    'Mumbai Indians': 'mumbai_indians.png',
    'Delhi Capitals': 'delhi_capitals.png',
    'Kolkata Knight Riders': 'kolkata_knight_riders.png',
    'Punjab Kings': 'punjab_kings.png',
    'Rajasthan Royals': 'rajasthan_royals.png',
    'Royal Challengers Bangalore': 'royal_challengers_bangalore.png',
    'Sunrisers Hyderabad': 'sunrisers_hyderabad.png'
}

@app.route('/', methods=['GET'])
def homePage():
    return render_template("index.html")

@app.route('/train', methods=['GET'])
def training():
    os.system("python main.py")
    return "Training Successful!" 


model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
transformer = joblib.load(Path('artifacts/data_transformation/transformer.joblib'))


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            batting_team = request.form['batting_team']
            bowling_team = request.form['bowling_team']
            city = request.form['city']
            
            # 1. Basic Range Validation
            target = float(request.form['target'])
            score = float(request.form['score'])
            overs = float(request.form['overs'])
            wickets_out = float(request.form['wickets'])

            if score > 1000 or target > 1000:
                return "Error: Score or Target cannot exceed 1000 runs."
            
            if wickets_out > 10:
                return "Error: Wickets cannot exceed 10."

            if overs > 20:
                return "Error: Overs cannot exceed 20."

            # 2. Cricket Over-Format Validation (n.0 to n.5)
            # Logic: Extract the decimal part. If it's > 0.5, it's an invalid ball count.
            over_str = request.form['overs']
            if '.' in over_str:
                decimal_part = int(over_str.split('.')[1])
                if decimal_part > 5:
                    return f"Error: Invalid over format '{overs}'. Balls in an over cannot exceed 5 (use {int(overs)+1}.0 instead)."

            # 3. Calculation Logic
            runs_left = target - score
            
            # Convert overs to total balls correctly (e.g., 5.2 overs = 5*6 + 2 = 32 balls)
            over_full = int(overs)
            ball_count = int(round((overs - over_full) * 10))
            total_balls_bowled = (over_full * 6) + ball_count
            balls_left = 120 - total_balls_bowled
            
            wickets = 10 - wickets_out
            
            # CRR/RRR logic with safety checks
            crr = score / overs if overs > 0 else 0
            rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

            # 4. Dataframe and Prediction
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'runs_left': [runs_left],
                'balls_left': [balls_left],
                'wickets': [wickets],
                'total_runs_x': [target],
                'crr': [crr],
                'rrr': [rrr]
            })

            numeric_input = transformer.transform(input_df)
            result = model.predict_proba(numeric_input)
            
            win_prob = round(result[0][1] * 100)
            loss_prob = round(result[0][0] * 100)
            
            bat_team = request.form['batting_team']
            bowl_team = request.form['bowling_team']

            batting_logo = url_for('static', filename='images/' + team_image_map[bat_team])
            bowling_logo = url_for('static', filename='images/' + team_image_map[bowl_team])

            # Passing team names to the template so your "Classy" UI shows them
            return render_template('results.html', 
                                win=win_prob, 
                                loss=loss_prob, 
                                batting_team=batting_team, 
                                bowling_team=bowling_team,
                                batting_logo=batting_logo, 
                                bowling_logo=bowling_logo, 
                                batting_team1=bat_team, 
                                bowling_team1=bowl_team,)

        except Exception as e:
            return f'Error: {str(e)}'

if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)