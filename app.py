from flask import Flask, render_template, request
import numpy as np
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load your model
with open('model/lead_scoring_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Classification logic (assuming you have defined it somewhere)
def classify_lead_score(score):
    if score > 0.8:
        return "Hot Lead"
    elif score > 0.5:
        return "Warm Lead"
    elif score > 0.3:
        return "Medium Lead"
    elif score > 0.1:
        return "Dormant Lead"
    else:
        return "Cold Lead"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form and set unused features to 0 by default
    
    # Lead Origin
    lead_origin_add_form = 1 if request.form['lead_origin'] == 'Add Form' else 0
    lead_origin_import = 1 if request.form['lead_origin'] == 'Lead Import' else 0
    
    # Last Activity
    last_activity_converted = 1 if request.form['last_activity'] == 'Converted to Lead' else 0
    last_activity_email_bounced = 1 if request.form['last_activity'] == 'Email Bounced' else 0
    last_activity_olark_chat = 1 if request.form['last_activity'] == 'Olark Chat Conversation' else 0
    last_activity_sms_sent = 1 if request.form['last_activity'] == 'SMS Sent' else 0
    
    # Country Unknown
    country_unknown = 1 if request.form['country_known'] == 'No' else 0
    
    # Specialization
    specialization_travel = 1 if request.form['specialization'] == 'Travel and Tourism' else 0
    
    # Current Occupation
    current_occupation_undisclosed = 1 if request.form['current_occupation'] == 'Undisclosed' else 0
    
    # Tags
    tags_lost = 1 if request.form['tags'] == 'Lost' else 0
    tags_ongoing = 1 if request.form['tags'] == 'Ongoing' else 0
    tags_unable_to_reach = 1 if request.form['tags'] == 'Unable to Reach' else 0
    
    # Lead Quality
    lead_quality_might_be = 1 if request.form['lead_quality'] == 'Might be' else 0
    lead_quality_worst = 1 if request.form['lead_quality'] == 'Worst' else 0
    
    # Asymmetrique Activity Index
    asym_activity_index_low = 1 if request.form['asym_activity_index'] == 'Low' else 0

    # Prepare the input with all 16 features, including the constant term
    features = np.array([
        1.837250,                  # constant term
        lead_origin_add_form,
        lead_origin_import,
        last_activity_converted,
        last_activity_email_bounced,
        last_activity_olark_chat,
        last_activity_sms_sent,
        country_unknown,
        specialization_travel,
        current_occupation_undisclosed,
        tags_lost,
        tags_ongoing,
        tags_unable_to_reach,
        lead_quality_might_be,
        lead_quality_worst,
        asym_activity_index_low
    ]).reshape(1, -1)

    # Make prediction
    lead_score = model.predict(features)[0]
    lead_category = classify_lead_score(lead_score)

    return render_template('index.html', score=lead_score, category=lead_category)




if __name__ == '__main__':
    app.run(debug=True)
