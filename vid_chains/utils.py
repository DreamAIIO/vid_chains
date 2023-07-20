# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_utils.ipynb.

# %% auto 0
__all__ = ['load_obj_model', 'detect_objects', 'centroid', 'list_centroids', 'inter_dist']

# %% ../nbs/00_utils.ipynb 2
from .imports import *
import math

# %% ../nbs/00_utils.ipynb 4
def load_obj_model(name="yolov8n.pt"):
    return YOLO(name)


def detect_objects(model, img):
    res = model(img, stream=True)
    return [{"boxes": r.boxes.data.detach().cpu().tolist()} for r in res]


def centroid (l):
    t = []
    cx = (l[0]+l[2])/2.0
    cy = (l[1]+l[3])/2.0
    t.append(cx)
    t.append(cy)
    return t

def list_centroids (objects):
    c=[]
    for i in range(0,11):
      l=[]
      for j in range(0,4):
        l.append(objects[0].get('boxes')[i][j])
      centre = centroid(l)
      c.append(centre)
    return c


def inter_dist(objects):
   
    c = list_centroids(objects)
    dis = []
    st = []
    for i in range(0,11):
        for j in range(i+1,11):
            #st.append("Distance b/w object "+str(i)+" and object "+str(j))
            #st.append("D("+str(i)+","+str(j)+")")
            dis.append(math.dist(c[i],c[j]))
    #return st,dis
    return dis
