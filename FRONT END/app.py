from flask import Flask,render_template,redirect,request,url_for, send_file
import mysql.connector, joblib
import joblib, re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import io
import base64
import matplotlib.pyplot as plt
import numpy as np
import lime
from lime import lime_tabular
import numpy as np
import pandas as pd


app = Flask(__name__)

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    port="3306",
    database='website'
)

mycursor = mydb.cursor()

def executionquery(query,values):
    mycursor.execute(query,values)
    mydb.commit()
    return

def retrivequery1(query,values):
    mycursor.execute(query,values)
    data = mycursor.fetchall()
    return data

def retrivequery2(query):
    mycursor.execute(query)
    data = mycursor.fetchall()
    return data


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        c_password = request.form['c_password']
        if password == c_password:
            query = "SELECT UPPER(email) FROM users"
            email_data = retrivequery2(query)
            email_data_list = []
            for i in email_data:
                email_data_list.append(i[0])
            if email.upper() not in email_data_list:
                query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
                values = (name, email, password)
                executionquery(query, values)
                return render_template('login.html', message="Successfully Registered! Please go to login section")
            return render_template('register.html', message="This email ID is already exists!")
        return render_template('register.html', message="Conform password is not match!")
    return render_template('register.html')


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        
        query = "SELECT UPPER(email) FROM users"
        email_data = retrivequery2(query)
        email_data_list = []
        for i in email_data:
            email_data_list.append(i[0])

        if email.upper() in email_data_list:
            query = "SELECT UPPER(password) FROM users WHERE email = %s"
            values = (email,)
            password__data = retrivequery1(query, values)
            if password.upper() == password__data[0][0]:
                global user_email
                user_email = email

                return redirect("/home")
            return render_template('login.html', message= "Invalid Password!!")
        return render_template('login.html', message= "This email ID does not exist!")
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        Response_Time = float(request.form['Response_Time'])
        Availability = float(request.form['Availability'])
        Throughput = float(request.form['Throughput'])
        Successability = float(request.form['Successability'])
        Reliability = float(request.form['Reliability'])
        Compliance = float(request.form['Compliance'])
        Best_Practices = float(request.form['Best_Practices'])
        Latency = float(request.form['Latency'])
        Documentation = float(request.form['Documentation'])

        inputs = [Response_Time, Availability, Throughput, Successability, Reliability, Compliance, Best_Practices, Latency, Documentation]

        # Load the trained model and scaler and dataset
        knn_model = joblib.load(r'models\knn_recommender_model.joblib')
        scaler = joblib.load(r'models\scaler.joblib')
        data = pd.read_csv(r"dataset\Final_data.csv")
        data.columns = data.columns.str.replace(' ', '_')  # Fix column names

        def recommendations(input):
            # Step 1: Scale the user input
            user_input_scaled = scaler.transform([input])

            # Step 2: Get the 10 nearest neighbors
            distances, indices = knn_model.kneighbors(user_input_scaled)

            URLs = []
            # Step 3: Retrieve the recommended WSDL Addresses
            for idx in indices[0]:
                url = data.iloc[idx]
                URLs.append(url)
            return URLs, distances
        
        URLs, distances = recommendations(inputs)
        recommendations_data = zip(URLs, distances[0])

        return render_template('recommendation.html', recommendations_data=recommendations_data)
    return render_template('recommendation.html')






### Classification

# Load the dataset
file_path = r'dataset\Final_dataset.csv'
df = pd.read_csv(file_path)

# Drop duplicates
duplicates = df[df.duplicated(keep=False)]

#label encoding the object columns.
# Store object column names
original_columns = df.select_dtypes(include='object').columns

# Initialize LabelEncoder
label_encoders = {}

# Apply LabelEncoder to each categorical variable
for col in original_columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Features and target
X = df.drop('Service Classification', axis=1)
y = df['Service Classification']

# Split labeled data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



model = joblib.load(r'models\DecisionTree.joblib')
prediction_dic = {
    0 : 'Bronze', 
    1 : 'Gold',
    2 : 'Platinum',
    3 : 'Silver' 
}

# creating an instance of the lime tabular explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
                                        training_data=np.array(X_train), 
                                        feature_names=X_train.columns, 
                                        class_names=['Bronze', 'Gold', 'Platinum', 'Silver'], 
                                        mode='classification')





def predict_func(input_data):
    result = model.predict([input_data])
    predicted_class = prediction_dic[result[0]]
    
    input_series = pd.Series(input_data, index=X_train.columns)

    # Create the explanation using LIME
    explanation = lime_explainer.explain_instance(data_row=input_series, 
                                                 predict_fn=model.predict_proba, 
                                                 top_labels=4, num_features=10)
    
    return predicted_class, explanation




@app.route('/classification', methods=['GET', 'POST'])
def classification():
    prediction = None
    explanation = None
    if request.method == 'POST':
        Response_Time = float(request.form['Response_Time'])
        Availability = float(request.form['Availability'])
        Throughput = float(request.form['Throughput'])
        Successability = float(request.form['Successability'])
        Reliability = float(request.form['Reliability'])
        Compliance = float(request.form['Compliance'])
        Best_Practices = float(request.form['Best_Practices'])
        Latency = float(request.form['Latency'])
        Documentation = float(request.form['Documentation'])
        Service_Name = int(request.form['Service_Name'])

        inputs = [Response_Time, Availability, Throughput, Successability, Reliability, Compliance, Best_Practices, Latency, Documentation, Service_Name]
        
        # Make prediction and explain with LIME
        predicted_class, explanation = predict_func(inputs)
        prediction = predicted_class
        explanation = explanation.as_html()

    
    # Load the dataset
    df = pd.read_csv(r"Dataset\Final_dataset.csv")
    
    # Drop columns
    columns_to_drop = ['Service Classification']
    df = df.drop(columns=columns_to_drop)
    
    # Replace spaces in column names with underscores
    df.columns = [re.sub(r'\s+', '_', col) for col in df.columns]
    
    # Define object columns to be encoded
    object_columns = df.select_dtypes(include=['object']).columns
    
    # Store label counts before encoding
    labels = {col: df[col].value_counts().to_dict() for col in object_columns}
    
    # Initialize LabelEncoder
    le = LabelEncoder()
    
    # Encode categorical columns and store the encoded value counts
    encodes = {}
    for col in object_columns:
        df[col] = le.fit_transform(df[col])
        value_counts = df[col].value_counts().to_dict()
        encodes[col] = value_counts
    
    dic = {}
    
    for key in labels.keys():
        dic[key] = []
        for sub_key, value in labels[key].items():
            for id_key, id_value in encodes[key].items():
                if value == id_value:
                    dic[key].append((sub_key, id_key))
                    break
    
    return render_template('classification.html', 
                           data=dic, 
                           prediction = prediction, 
                           explanation = explanation)





if __name__ == '__main__':
    app.run(debug = True)