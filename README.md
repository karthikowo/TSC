# dataGenie-hackathon

# Dependancies to be installed:

statsmodels

pandas

scikit-learn

fastapi

matplotlib

Tensorflow

Initial Setup of hackathon

# To run : uvicorn api_endpoints:app --reload

Check localhost:8000/docs and use uploadfile to upload dataset. 

Specify exact fields in the dataset, the default values are used here.

![image](https://user-images.githubusercontent.com/76225835/224533004-e9ccf005-20fb-400f-bdcb-333bad501038.png)


Then the plot of the dataset is displayed.


![image](https://user-images.githubusercontent.com/76225835/224532975-918d80b9-90e9-48d7-96da-de3525dd8e54.png)


Then Model is built based on the features of the dataset and the predictions are made for the test set.


![image](https://user-images.githubusercontent.com/76225835/224533111-2a7f2cb0-627b-45f5-9391-1737e9c2a76d.png)


The model used and the MAPE of the model is returned as a response.


![image](https://user-images.githubusercontent.com/76225835/224533163-30a47944-c565-488c-a1c4-900cae1ac854.png)


# Time Series data decomposition visualization.


In bash, type streamlit run dashboard.py


![image](https://user-images.githubusercontent.com/76225835/224566290-e08fbdac-21cf-4f44-a6f4-88364f2fb279.png)

![image](https://user-images.githubusercontent.com/76225835/224566317-13d9158b-54fc-4e65-a8a4-12c2583151fc.png)



# Implementing the Model classifier

The initial thought process was to create a variant of nbc or kernel SVM classifier which classifies the best model to take based on the 

time series extracted features like trend seasonality etc. But I was unable to carry out the implementation and chose to pick the model based on

the extracted features instead ,also one of the reason for this was more than 1 model comes under multiple use case sometimes. The models chosen 

are :
