import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template
app=Flask(__name__)
def myfunc(a, b):
    return {"x":a,"y":b}
@app.route('/')
def root():   
    markers={}
    dataset = pd.read_csv("raw.csv")
    x = dataset.iloc[:, [5, 7]].values
    kota = dataset.iloc[:,[4]].values
    kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(x)
    markers['tinggi'] = list(map(myfunc,x[y_kmeans == 1, 0],x[y_kmeans == 1, 1]))
    markers['sedang'] = list(map(myfunc,x[y_kmeans == 2, 0],x[y_kmeans == 2, 1]))
    markers['rendah'] = list(map(myfunc,x[y_kmeans == 0, 0],x[y_kmeans == 0, 1]))
    markers['centroid'] = list(map(myfunc,kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1]))
    return render_template('index.html',markers=markers )
if __name__ == '__main__':
    app.run(host="localhost", port=8080, debug=True)