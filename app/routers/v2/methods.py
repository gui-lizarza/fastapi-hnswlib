import time
import os

import numpy as np
import hnswlib

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router = APIRouter()

graph = hnswlib.Index(space='cosine', dim=3)

if os.path.isfile('app/graph.bin'):
    graph.load_index('app/graph.bin')
else:
    graph.init_index(max_elements=1000, ef_construction=200, M=50)
    graph.set_ef(50)

class QueryRequest(BaseModel):
    vec: List[float]
    k: int

class QueryResponse(BaseModel):
    time_passed: float
    labels: List[int]
    distances: List[float]

class AddResponse(BaseModel):
    time_passed: float
    item_count: int

class SaveResponse(BaseModel):
    time_passed: float
    message: str

@router.put('/query', response_model=QueryResponse)
async def make_query(request: QueryRequest):
    
    vec = request.vec
    k = request.k
    
    try:
        tic = time.time()
        labels, distances = graph.knn_query(vec, k=k)
        toc = time.time()
    except RuntimeError:
        raise HTTPException(status_code=400, detail="Menos do que {} nós no grafo".format(k))
    
    labels = labels.squeeze(axis=0)
    distances = distances.squeeze(axis=0)

    time_passed = toc - tic

    return {
        'time_passed': time_passed,
        'labels': labels.tolist(),
        'distances': distances.tolist()
    }

@router.put('/add', response_model=AddResponse)
async def add_item(vec: List[float]):

    label = graph.get_current_count()
    tic = time.time()
    graph.add_items(vec, label)
    toc = time.time()

    time_passed = toc - tic

    item_count = label + 1
    return {
        'time_passed': time_passed,
        'item_count': item_count
    }

@router.put('/save', response_model=SaveResponse)
async def save_index():

    tic = time.time()
    graph.save_index('app/graph.bin')
    toc = time.time()

    time_passed = toc - tic

    return {
        'time_passed': time_passed,
        'message': 'Index saved!'
    }