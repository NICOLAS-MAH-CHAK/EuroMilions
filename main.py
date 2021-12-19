import euroMillions
from fastapi import Body,FastAPI
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from pydantic.fields import Field

euroMillions.trainingModel()
best_to_play = euroMillions.best_to_play()
print(best_to_play)


app= FastAPI()

class Tirage():
    boule1:int
    boule2:int
    boule3:int
    boule4:int
    boule5:int
    etoile1:int
    etoile2:int

class Prediction():
    gain:float

    
@app.get("/api/best")
async def get_good_tirage(best: Tirage):
        best.boule1==best_to_play()[0]
        best.boule2==best_to_play[1]
        best.boule3==best_to_play[2]
        best.boule4==best_to_play[3]
        best.boule5==best_to_play[4]
        best.etoile1==best_to_play[5]
        best.etoile2==best_to_play[6]
        return {"meilleur_tirage": best, "message": "Voici une combinaison  avec une forte probabilité de gagner"}



@app.post("/api/predict/{tirage}")
async def predict_tirage(prediction: Prediction,tirage:Tirage):
    prediction.gain=euroMillions.predictionRF(tirage)
    return {"model_name":prediction, "message": "Voila la probabilité de gagner de la combinaison entrée"}

"""
@app.put("")
async def model_read(model_name:BaseModel):
    return {"Modèle":model_name}

@app.post("/model/predict")
async def rooget_model(model_name: ModelName):
    if(model_name == ModelName.alexnet):
        return {"model_name":model_name, "message": "Deep Learning FTW!"}
    if(model_name.value == "lenet"):
        return {"model_name":model_name, "message": "LeCNN all iamges"}
    return {"model_name":model_name, "message": "autre"}
"""
