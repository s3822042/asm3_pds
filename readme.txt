All code file for task 3 and 4 inside 'movie-rating-prediction' folder

System using:
Windows 11 Pro
Python 3.9

General Instruction:

Please do this first before all of the below
- Install all required packages inside requirements.txt using pip install -r requirements.txt in root directory

For starting the visualization dashboard:
 - Go inside folder "movie-rating-prediction'
 - Run python dashboard.py
 - Wait until  this line "Running on http://127.0.0.1:5000/" appear
 - Navigate to this URL to see the visualization dashboard

Task 3 API:
Please use Postman in order to do below step:
 - Create new POST request to http://127.0.0.1:5000/evaluate or http://127.0.0.1:5000/predict
  - Body:
    - formdata:
      - key : file
      - type: file
      - value: choose file data.csv in "test" folder
 - Evaluate API: http://127.0.0.1:5000/evaluate to output the evaluation metric
 - PRedict API http://127.0.0.1:5000/predict to output the predicted rating value
