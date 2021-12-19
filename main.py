import euroMillions
from fastapi import Body,FastAPI
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from pydantic.fields import Field

predictions_proba,predictions,accuracy_model,X_test = euroMillions.training_model()

app= FastAPI()


@app.get("/api/best")
async def best():
    """Adds a combination in the file where the model tooks his data.


    Args:
        predictions_proba: An array with the probabilities to win about X_test combinations, predicted by the model.
        X_test: An array with the combinations of tests


    Returns:
      The list of the combination which has the most probability to win with.
    

    Raises:
      None  

    """
    best_to_play = euroMillions.best_to_play(predictions_proba, X_test)

    mystring = ""

    for digit in best_to_play:
        mystring += str(digit) + " "

    return{"La meilleur combinaison est " : mystring}


class combi(BaseModel):
    B1: int
    B2: int
    B3: int
    B4: int
    B5: int
    E1: int
    E2: int



# @app.post("/api/predict")
# async def predict(combi : combi):
#     """Print the probability to win for a combination according with the model.


#     Args:
#       x: a list meaning a combination of a lotto draw.


#     Returns:
#       The probability to win with the combination in enter.
    

#     Raises:
#       None  
#     """
#     res = euroMillions.prediction_RF([combi.B1,combi.B2,combi.B3,combi.B4,combi.B5,combi.E1,combi.E2])
#     return{"La probabilité de cette combinaison est de ": res}


# @app.get("/api/model")
# async def model():
#     """Indicates information about the model.


#     Args:
#       accuracy_model : the accuracy of the model


#     Returns:
#       A tuple with the name of the model, the number of estimators in parameters of the model and the accuracy of the model calculated on the test data.
    

#     Raises:
#       None  
#     """
#     infos_model = euroMillions.infos_model(accuracy_model)

#     return infos_model


# @app.put("/api/model")
# async def add_base(combi : combi):
#     """Adds a combination in the file where the model tooks his data.


#     Args:
#       combination: a list meaning a combination which won a lotto draw.


#     Returns:
#       None
    

#     Raises:
#       None  
#     """
#     euroMillions.add_base(combi)
#     return {"Combi added":combi}


# @app.post("/api/model/retrain")
# async def retrain():
#     """Formats a dataset from the file and with tools using random, and  trains a random forest with it.


#     Args:
#       None


#     Returns:
#       The status and the accuracy of the model
    

#     Raises:
#       None  
#     """
#     accuracy = euroMillions.training_model()[2]
#     return {"Status":"Modele retrained","Accuracy":accuracy}