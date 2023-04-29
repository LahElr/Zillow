from sklearn.cluster import KMeans
import pandas
import gc
import numpy
import pickle
import time

print(f'{time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))}: Started...')
p16 = pandas.read_csv("./data/properties_2016.csv")
p17 = pandas.read_csv("./data/properties_2017.csv")

df16 = p16[["latitude","longitude"]]
df17 = p17[["latitude","longitude"]]

del p16,p17
gc.collect()

print(f'{time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))}: Data collected...')

df = pandas.concat([df16,df17],axis = 0,ignore_index=True,join="outer")
df = df.dropna(axis=0,how="any")
data = df.to_numpy()

print(data.shape)

clusterer = KMeans(n_clusters=300,random_state=42,n_init="auto")
print(f'{time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))}: Model initialized...')

clusterer.fit(data)
with open("./models/position_clusterer.pkl","wb") as pkl_file:
    pickle.dump(clusterer,pkl_file)
print(f'{time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))}: Finished!')
