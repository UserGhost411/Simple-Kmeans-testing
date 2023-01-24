import pandas as pd
from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template
app=Flask(__name__)
def myfunc(a, b):
    return {"x":a,"y":b}
@app.route('/')
def home():  
    return render_template('index.html')
@app.route('/data')
def data():  
    dataset = pd.read_csv("raw.csv")
    return render_template('data.html', tables=[dataset.to_html(classes='table',border = 0, header="true")], titles=[''])
@app.route('/elbow')
def elbow():   
    markers={}
    dataset = pd.read_csv("raw_gud.csv")
    x = dataset.iloc[:, [1,2,3,4,5,6,7,8,9]].values
    wcss = []
    for i in range (1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    markers['elbow'] = wcss 
    return render_template('elbow.html',markers=markers )

@app.route('/chart')
def chart():   
    markers={}
    dataset = pd.read_csv("raw_gud.csv")
    x = dataset.iloc[:, [1,2,3,4,5,6,7,8,9]].values
    kota = dataset.iloc[:,[0]].values
    kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(x)
    markers['tinggi'] = list(map(myfunc,x[y_kmeans == 2, 0],x[y_kmeans == 2, 1]))
    markers['sedang'] = list(map(myfunc,x[y_kmeans == 1, 0],x[y_kmeans == 1, 1]))
    markers['rendah'] = list(map(myfunc,x[y_kmeans == 0, 0],x[y_kmeans == 0, 1]))
    markers['centroid'] = list(map(myfunc,kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1]))
    return render_template('chart.html',markers=markers )
if __name__ == '__main__':
    app.run(host="localhost", port=8080, debug=True)