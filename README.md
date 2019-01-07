# Cuisine Prediction Service


- Make sure you have python3, flask, scikit-learn, pandas, scipy, csv, numpy, pickle, seaborn, matplotlib and some python packages like request, json, os, io etc. installed

- Run the webservice `python webservice.py`
- In a different terminal run this curl command
`curl --header "Content-Type: application/json"   --request POST   --data @test2.json   http://127.0.0.1:5000/cuisine/api/json`
- You should see below output
`{"predictions": [{"id": 18009, "cuisine": "italian", "probability": 0.17846838753494407}, {"id": 28583, "cuisine": "italian", "probability": 0.1918703125538735}]}`
