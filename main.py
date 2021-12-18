#!/usr/bin/env python3
from fastapi import FastAPI, File, UploadFile, Header, Body
from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Set, List


class ModelName(str,Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"
    
app = FastAPI()

# @app.get("/items/{items_id}")
# async def read_item(item_id: int):
#     return {"item_id": item_id}
  
    
@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    
    if model_name == ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}
    
    return {"model_name": model_name, "message": "Have some residuals"}

class Item(BaseModel):
    name: str
    description: Optional[str] = Field(
        None, title = "The description of the item", max_length=300
        )
    price: float = Field(...,gt=0, description="The price must be greater than 0")
    tax: Optional[float] = None
    
# @app.post("/items/")
# async def create_item(item: Item):
#     return item

# @app.put("/items/{item_id}")
# async def create_item(item_id: int, item: Item, q: Optional[str] = None):
#     result = {"item_id": item_id, **item.dict()}
#     if q: result.update({"q": q})
#     return result

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item = Body(...,embed=True)):
    results = {"item_id": item_id, "item": Item}
    return results

# @app.post("/items/",status_code = status.HTTP_201_CREATED)
# async def create_item(name: str):
#     return {"name": name}