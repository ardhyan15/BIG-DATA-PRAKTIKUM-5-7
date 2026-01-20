# Tugas 6
## Implementasi ML sederhana
Algoritma yang akan diimplementasikan:
- Linear Regression
- K-Means

## Persiapan
1. Buka google colab
2. Install pyspark `!pip install pyspark`
3. Buat spark session
```py
from pyspark.sql import SparkSession
# Membuat sesi
spark = SparkSession.builder \
 .appName("Latihan_MLlib") \
 .getOrCreate()
print("Spark Session berhasil dibuat!")
```
<br>

## Linear Regression
1. Buat dataset dummy!
```py
data_gaji = [
 (1.0, 20, 5000),
 (2.0, 22, 6000),
 (3.0, 25, 7000),
 (4.0, 26, 8500),
 (5.0, 30, 10000),
 (6.0, 31, 11500)
]
columns = ["pengalaman", "umur", "gaji"]
df_regresi = spark.createDataFrame(data_gaji, columns)
print("Data Awal:")
df_regresi.show()
```
2. Tambahkan data baru
```py
data_baru = [(10.0, 40, None)]  # Gaji diisi None karena akan diprediksi
df_baru = spark.createDataFrame(data_baru, ["pengalaman", "umur", "gaji"])
```
3. Gabungkan jadi satu dataframe
```py
df_regresi_full = df_regresi.union(df_baru)
```
4. Gabungkan fitur menjadi satu vektor dengan vector assembler
```py
from pyspark.ml.feature import VectorAssembler
# Menggabungkan kolom input 'pengalaman' dan 'umur' menjadi satu kolom 'features'
assembler = VectorAssembler(
 inputCols=["pengalaman", "umur"],
 outputCol="features"
)
# Transformasi data
data_siap_reg = assembler.transform(df_regresi_full).select("features", "gaji")
print("Data dalam format Vector:")
data_siap_reg.show(truncate=False)
```
output yang diharapkan
```txt
+----------+-------+
|features  |gaji   |
+----------+-------+
|[1.0,20.0]|5000.0 |
|[2.0,22.0]|6000.0 |
|[3.0,25.0]|7000.0 |
|[4.0,26.0]|8500.0 |
|[5.0,30.0]|10000.0|
|[6.0,31.0]|11500.0|
|[8.0,35.0]|NULL   |
+----------+-------+
```
5. Pisahkan data lama untuk training dan data baru untuk prediksi lalu latih model
```py
from pyspark.ml.regression import LinearRegression
train_data = data_siap_reg.filter("gaji IS NOT NULL")
new_data = data_siap_reg.filter("gaji IS NULL")
lr = LinearRegression(featuresCol="features", labelCol="gaji")

# Latih Model
model_lr = lr.fit(train_data)

# Prediksi pada data baru
hasil_prediksi = model_lr.transform(new_data)
hasil_train = model_lr.transform(train_data)
print("Hasil Prediksi Gaji:")
hasil_prediksi.select("features", "gaji", "prediction").show()
hasil_train.select("features","gaji","prediction").show()

# Menampilkan koefisien (kemiringan garis)
print(f"Koefisien: {model_lr.coefficients}")
print(f"Intercept: {model_lr.intercept}")
```
Hasil Dari linear regression:

```txt
# Prediksi Gaji untuk data baru
+----------+----+------------------+
|  features|gaji|        prediction|
+----------+----+------------------+
|[8.0,35.0]|NULL|13949.999999999844|
+----------+----+------------------+

# Prediksi gaji untuk train data
+----------+-------+------------------+
|  features|   gaji|        prediction|
+----------+-------+------------------+
|[1.0,20.0]| 5000.0| 4712.500000000006|
|[2.0,22.0]| 6000.0|  6037.49999999996|
|[3.0,25.0]| 7000.0| 7325.000000000075|
|[4.0,26.0]| 8500.0| 8687.499999999867|
|[5.0,30.0]|10000.0| 9937.500000000146|
|[6.0,31.0]|11500.0|11299.999999999936|
+----------+-------+------------------+

Koefisien: [1399.9999999996287,-37.49999999983759]
Intercept: 4062.499999997129
```
## K-Means
1. Buat data 
```py
from pyspark.ml.clustering import KMeans

data_mall = [
 (15, 39), (16, 81), (17, 6), (18, 77), (19, 40), # Grup Acak
 (50, 50), (55, 55), (60, 60), # Grup Menengah
 (100, 90), (110, 95), (120, 88) # Grup Kaya & Boros
]
df_mall = spark.createDataFrame(data_mall, ["pendapatan", "skor"])
```
2. Buat juga vector assemblernya
```py
assembler_cluster = VectorAssembler(inputCols=["pendapatan", "skor"],
outputCol="features")
data_siap_cluster = assembler_cluster.transform(df_mall)
```
3. Buat K-means dengan K=3
```py
kmeans = KMeans().setK(3).setSeed(1)
model_km = kmeans.fit(data_siap_cluster)
```
4. Buat K-means dengan K=2
```py
kmeans_2 = KMeans().setK(2).setSeed(1)
model_km_2 = kmeans_2.fit(data_siap_cluster)
```
5. Prediksi untuk masing-masing setK
```py
# K=3
prediksi_cluster = model_km.transform(data_siap_cluster)
print("Hasil Pengelompokan (Prediction adalah nomor cluster):")
prediksi_cluster.show()

#K=2
prediksi_cluster_2 = model_km_2.transform(data_siap_cluster)
print("Hasil Pengelompokan dengan K=2:")
prediksi_cluster_2.show()
```
6. Lihat dan bandingkan pusat clusternya
```py
#K=3
centers = model_km.clusterCenters()
print("Pusat Cluster (Centroids):")
for i, center in enumerate(centers):
 print(f"Cluster {i}: {center}")

#K=2
centers_2 = model_km_2.clusterCenters()
print("Pusat Cluster (K=2):")
for i, center in enumerate(centers_2):
    print(f"Cluster {i}: {center}")
```
7. Hasil
```txt
K=3
Pusat Cluster (Centroids):
Cluster 0: [110.  91.]
Cluster 1: [17.  48.6]
Cluster 2: [55. 55.]

K=2
Pusat Cluster (Centroids):
Cluster 0: [110.  91.]
Cluster 1: [31.25 51.  ]
```
Dari perbandingan ini dapat disimpulkan bahwa cluster K=3 sebelumnya kira-kira terbagi menjadi:
- Kelompok 0: Pendapatan rendah, skor rendah
- Kelompok 1: Pendapatan tinggi, skor tinggi
- Kelompok 2: Pendapatan menengah, skor menengah

Sedangkan K=2, dua cluster kemungkinan besar akan membagi data menjadi:
- Cluster 0: Data dengan pendapatan rendah–menengah + skor rendah–menengah
- Cluster 1: Data dengan pendapatan tinggi + skor tinggi

Artinya, tidak persis "Kaya" vs "Miskin", karena ada data pendapatan menengah yang mungkin masuk ke salah satu cluster tergantung kedekatannya dengan centroid.
