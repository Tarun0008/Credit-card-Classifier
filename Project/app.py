from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.offline as opy
import numpy as np
import os

app = Flask(__name__, static_url_path='/static', template_folder=os.path.abspath("templates"))

valid_credentials = {'Admin': 'Admin'}
excel_file = 'credit_card_data.xlsx'
success_message = 'Data added successfully!'

# Load the data
excel_file = 'credit_card_data.xlsx'
credit_cards = pd.read_excel(excel_file, engine='openpyxl')

credit_cards = credit_cards.dropna()
# Create the clusters
clustering_data = credit_cards[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
scaler = MinMaxScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)

# Run KMeans for different K values
distortions = []
for k in range(1, 50):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(clustering_data_scaled)
    distortions.append(kmeans.inertia_)

# Calculate the second derivative of the distortion values
second_derivative = np.diff(np.diff(distortions))

optimal_k_index = np.where(second_derivative > 0)[0][0] + 1
optimal_k = optimal_k_index + 1  # Convert index to K value

# Apply KMeans with the optimal K value
kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=0)
clusters_optimal = kmeans_optimal.fit_predict(clustering_data_scaled)
credit_cards["CREDIT_CARD_SEGMENTS_OPTIMAL"] = clusters_optimal

# Transform names of clusters for easy interpretation
credit_cards["CREDIT_CARD_SEGMENTS_OPTIMAL"] = credit_cards["CREDIT_CARD_SEGMENTS_OPTIMAL"].map({
    i: f"Cluster {i+1}" for i in range(optimal_k)
})

# Plot the clusters using Plotly Express
fig_optimal = px.scatter_3d(credit_cards, x='BALANCE', y='PURCHASES', z='CREDIT_LIMIT',
                            color='CREDIT_CARD_SEGMENTS_OPTIMAL', symbol='CREDIT_CARD_SEGMENTS_OPTIMAL',
                            size_max=0, opacity=1, width=950, height=750,
                            title=f'Credit Card Segmentation (Optimal K = {optimal_k})')

# Adjusting aspect ratio
fig_optimal.update_layout(scene=dict(aspectmode="cube"))

plot_html = opy.plot(fig_optimal, auto_open=False, output_type='div')



@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in valid_credentials and valid_credentials[username] == password:
            return render_template('index.html')
        return 'Invalid credentials. Please try again.'
    return render_template('login.html')

@app.route('/Input')
def Input():
    return render_template('Input.html')

@app.route('/Visualize', methods=['GET', 'POST'])
def Visualize():
    if request.method == 'POST':
        pass
    return render_template('Visualize.html', plot_html=plot_html)

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        field_names = [
            'CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES',
            'CASH_ADVANCE', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY',
            'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
            'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'TENURE'
        ]
        form_data = {name: request.form[name] for name in field_names}
        df = pd.DataFrame([form_data])
        try:
            existing_data = pd.read_excel(excel_file, engine='openpyxl')
            updated_data = pd.concat([existing_data, df], ignore_index=True)
            updated_data.to_excel(excel_file, index=False, engine='openpyxl')
            return render_template('Input.html', success_message=success_message)
        except FileNotFoundError:
            df.to_excel(excel_file, index=False, engine='openpyxl')
            return 'Excel file created with form data!'
        except Exception as e:
            return f'An error occurred: {str(e)}'

if __name__ == '__main__':
    app.run(debug=True)