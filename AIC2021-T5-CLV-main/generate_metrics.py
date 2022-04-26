import json
import heapq
from PIL import Image, ImageDraw
import os
import random
import numpy as np
import sys

json_file = sys.argv[1]

def recall(restuls, at=5):
    c = 0
    RR = 0
    for k in results:
        resultK = results[k]
        for i in range(at):
            if resultK[i] == k:
                RR += 1
                break
        c += 1
    return RR/c

def MRR(results):
    c = 0
    RR = 0
    for k in results:
        resultK = results[k]
        rank = 1
        for i in range(len(resultK)):
            if resultK[i] == k:
                break
            rank += 1
        RR += 1/rank
        c += 1
    return RR/c

def best(results, n=20):
    q = []
    heapq.heapify(q)

    for k in results:
        resultK = results[k]
        rank = 1
        for i in range(len(resultK)):
            if resultK[i] == k:
                break
            rank += 1
        heapq.heappush(q, (-rank, k))
        if len(q) > n:
            heapq.heappop(q)
    return q

def worst(results, n=20):
    q = []
    heapq.heapify(q)

    for k in results:
        resultK = results[k]
        rank = 1
        for i in range(len(resultK)):
            if resultK[i] == k:
                break
            rank += 1
        heapq.heappush(q, (rank, k))
        if len(q) > n:
            heapq.heappop(q)
    return q


results = json.load(open(json_file))
print("MRR -> {0}".format(MRR(results)))
print("Recall@5 -> {0}".format(recall(results, 5)))
print("Recall@10 -> {0}".format(recall(results, 10)))
