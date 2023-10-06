PART I, model transcription into the "model.py" file.
The model selected in this case is Logistic Regression with Feature Importante and with Balance.
By using this method makes the model simpler an easier to interpret, which can be beneficial in terms of
understanding and explaining the model's decisions.
Also, class balancing improves the performance by increasing the recall of class "1".
Since there is no noticeable difference between XGBoost and Logistic Regression in terms of results, its better 
to opt for the simplicity of Logistic Regression.
PART II.
I developed the API using fastapi with the integration of joblib, this way i could import the machine learning
algorithm. used uvicorn for testing and proceeded to cloud development.
PART III.
I personally decided to implement my app using amazon web services, since i was already familiar with it.
I used an EC2 host for the app implementation. The app is currently accesable using the following url
url= http://ec2-34-229-40-55.compute-1.amazonaws.com:8000/
