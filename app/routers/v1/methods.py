import time

import numpy as np
import hnswlib

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

data = np.random.normal(loc=0.0, scale=1.0, size=(1000, 512))

item_count = len(data)
data_labels = np.arange(item_count)

graph = hnswlib.Index(space='cosine', dim=512)
graph.init_index(max_elements=len(data)*2, ef_construction=200, M=16)
graph.add_items(data, data_labels)
graph.set_ef(50)

class QueryResponse(BaseModel):
    time_passed: float
    labels: List[int]
    distances: List[float]

class AddResponse(BaseModel):
    time_passed: float
    item_count: int

@router.get('/query', response_model=QueryResponse)
async def make_query():
    query = np.random.normal(loc=0.0, scale=1.0, size=(1, 512))

    tic = time.time()
    labels, distances = graph.knn_query(query, k=3)
    toc = time.time()
    
    labels = labels.squeeze(axis=0)
    distances = distances.squeeze(axis=0)

    time_passed = toc - tic

    return {
        'time_passed': time_passed,
        'labels': labels.tolist(),
        'distances': distances.tolist()
    }

@router.put('/add', response_model=AddResponse)
async def add_item():
    item = np.random.normal(loc=0.0, scale=1.0, size=(1, 512))

    global item_count
    curr_count = item_count

    tic = time.time()
    graph.add_items(item, item_count)
    toc = time.time()

    item_count += 1
    time_passed = toc - tic

    return {
        'time_passed': time_passed,
        'item_count': curr_count
    }