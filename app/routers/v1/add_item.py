import time
import numpy as np

from fastapi import APIRouter
from pydantic import BaseModel

from ...main import graph, item_count

router = APIRouter()

class AddResponse(BaseModel):
    time_passed: float
    item_count: int

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