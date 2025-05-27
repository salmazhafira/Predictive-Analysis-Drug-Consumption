# Laporan Proyek Machine Learning - Salma Zhafira Muchtar

## Project Overview

### Latar Belakang

Penggunaan zat psikoaktif, baik yang legal maupun ilegal, merupakan isu yang kompleks dan multidimensional yang dipengaruhi oleh berbagai faktor psikologis dan demografi. Karakteristik kepribadian, impulsivitas, dan sensasi mencari pengalaman (sensasi mencari) adalah komponen penting yang harus dikaji secara menyeluruh untuk memahami perilaku konsumsi zat tersebut ([Fernández-Suárez et al., 2024](https://www.mdpi.com/2076-3425/14/5/449); [Sagepub, 2024)](https://journals.sagepub.com/doi/10.1177/09727531241274098)). Profil kepribadian tertentu, seperti tingkat neurotisisme yang tinggi, extraversion, dan kesadaran (conscientiousness) yang rendah, telah dikaitkan dengan risiko lebih besar dalam penggunaan zat terlarang maupun alkohol  ([Volkow et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10168177/)).

Metode komputasi seperti Machine Learning semakin banyak digunakan untuk memahami dan memprediksi perilaku konsumsi zat berdasarkan demografi dan psikologi ([Castaneda et al., 2022](https://www.idescat.cat/sort/sort471/47.1.1.Castaneda-etal.pdf)). Analisis data dapat digunakan untuk mengungkap berbagai pola dan hubungan yang kompleks secara lebih sistematis dan objektif. Hal ini berkontribusi signifikan dalam pengambilan keputusan, khususnya dalam strategi pencegahan dan intervensi dini ([Tan & Fauzi, 2023](https://www.svedbergopen.com/files/1698407561_3_IJDSBDA202313011740NZ_(p_45-57).pdf))

Sumber data yang lengkap untuk menganalisis hubungan tersebut adalah Dataset Konsumsi Obat (Quantified), yang dikumpulkan oleh Fehrman et al. (2017) dan tersedia di Perpustakaan Pembelajaran Mesin UCI ([Fehrman et al., 2017](https://arxiv.org/abs/1506.06297)). Dataset ini berisi data dari 1885 responden yang mencakup 12 atribut utama, antara lain skor kepribadian berdasarkan model lima faktor (NEO-FFI-R), skor impulsivitas (BIS-11), sensasi mencari pengalaman (ImpSS), serta variabel demografi seperti usia, jenis kelamin, tingkat pendidikan, negara, dan etnisitas ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)). Selain itu, data juga mencakup informasi tentang frekuensi penggunaan delapan belas jenis psikoaktif, termasuk alkohol, amfetamin, benzodiazepin, ganja, kokain, heroin, dan satu substansi yang tidak nyata untuk mengidentifikasi klaim palsu ([Fehrman et al., 2017](https://arxiv.org/abs/1506.06297)).

Dengan tujuh kategori mulai dari "Never Used" hingga "Used in Last Day", setiap responden melaporkan berapa kali mereka menggunakan zat tersebut. Dataset ini memungkinkan klasifikasi status pengguna dan analisis risiko konsumsi berdasarkan karakteristik psikologis dan demografis ([Fehrman et al., 2017](https://arxiv.org/abs/1506.06297)). Dengan demikian, dataset ini sangat berguna untuk mengembangkan model prediktif dan memahami faktor-faktor yang memengaruhi perilaku penggunaan zat psikoaktif, termasuk menemukan faktor kepribadian dan demografi yang paling berpengaruh terhadap risiko konsumsi alkohol dan zat lain ([Fehrman et al., 2017](https://arxiv.org/abs/1506.06297)).

Melalui penelitian berbasis data ini, diharapkan diperoleh pemahaman yang lebih baik mengenai faktor risiko yang memengaruhi konsumsi zat, serta kontribusi penting untuk bidang psikologi klinis, kebijakan kesehatan masyarakat, dan upaya rehabilitasi.

## Business Understanding

### Problem Statements

Berdasarkan latar belakang dan kajian pustaka mengenai konsumsi zat serta hubungan dengan faktor demografi dan kepribadian, maka rumusan masalah dalam penelitian ini adalah sebagai berikut:

1. Bagaimana hubungan antara tingkat impulsivitas dan sensasi mencari pengalaman (sensation seeking) dengan frekuensi konsumsi zat psikoaktif?

2. Bagaimana model prediktif dapat digunakan untuk memprediksi frekuensi konsumsi zat psikoaktif berdasarkan karakteristik psikologis (big five personality traits) dan demografis?

### Goals

Berdasarkan rumusan masalah yang telah dijelaskan, penelitian ini bertujuan untuk:

1. Menganalisis hubungan antara tingkat impulsivitas dan sensasi mencari pengalaman dengan frekuensi konsumsi zat psikoaktif.

2. Membangun model prediksi untuk memprediksi frekuensi konsumsi zat psikoaktif berdasarkan variabel psikologis (big five personality traits) dan demografis.

### Solution Statements

Berdasarkan tujuan yang telah dirumuskan, solusi yang diajukan untuk mencapai goals tersebut meliputi dua pendekatan utama sebagai berikut:

1. Membangun model prediksi menggunakan algoritma regresi seperti Linear Regression dan Random Forest Regressor untuk memprediksi frekuensi konsumsi zat berdasarkan tingkat impulsivitas, sensasi mencari pengalaman (sensation seeking), fitur psikologis (big five personality traits) dan demografi.

2. Melakukan hyperparameter tuning pada model terbaik serta mengevaluasi performa model dengan metrik seperti MAE, RMSE, dan R².

## Data Understanding

### Data Loading
Penelitian ini menggunakan dataset Drug Consumption (Quantified) yang diperoleh dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified), dan pertama kali dipublikasikan oleh Fehrman et al. (2017). Dataset ini dikumpulkan melalui survei daring terhadap 1.885 responden dewasa, yang berasal dari berbagai negara, terutama kawasan Eropa dan Amerika Utara.

Dataset Drug Consumption (Quantified) dimuat langsung dari repositori UCI Machine Learning menggunakan fungsi pd.read_csv() dari pustaka pandas, dengan URL sumber data yang mengarah ke file `.data` berformat plaintext. Karena file tersebut tidak memiliki header, nama-nama kolom ditentukan secara manual menggunakan parameter columns, berdasarkan dokumentasi resmi UCI. Konversi ini dilakukan agar dataset lebih mudah dibaca, dianalisis, dan diolah dalam ekosistem Python, khususnya untuk keperluan data preprocessing, visualisasi, maupun pengembangan model machine learning.

Setelah kolom diberi label, langkah selanjutnya dalam pengolahan data melibatkan transformasi nilai-nilai numerik pada variabel demografis dan kepribadian menjadi label kategori yang lebih representatif. Misalnya, nilai -0.95197 pada kolom *Age* diubah menjadi *“18-24”*, dan nilai 0.48246 pada kolom *Gender* dikonversi menjadi *“Female”*. Transformasi semacam ini dilakukan untuk meningkatkan keterbacaan data serta mendukung interpretasi hasil analisis, terutama saat visualisasi data atau eksplorasi hubungan antar variabel.

Secara keseluruhan, dataset terdiri dari 1.885 baris, di mana setiap baris merepresentasikan satu individu. Terdapat 32 kolom yang mencakup atribut psikologis, demografis, serta informasi terkait penggunaan 18 jenis zat psikoaktif. Setiap entri dalam dataset ini menggambarkan kombinasi dari karakteristik kepribadian, latar belakang demografi, dan pola konsumsi zat yang dilaporkan oleh masing-masing responden.

Sumber Data : [Drug Consumption (Quantified)](https://archive.ics.uci.edu/dataset/373/drug+consumption+quantified)

---
**Berikut penjelasan fitur berdasarkan kategori data dalam dataset drug consumption:**

1. **ID**
   Variabel unik yang digunakan untuk mengidentifikasi setiap responden.

2. **Age (Usia)**
   Kelompok umur peserta dalam kategori 18-24, 25-34, 35-44, 45-54, 55-64, dan 65+.

3. **Gender (Jenis Kelamin)**
   Jenis kelamin peserta, yaitu laki-laki dan perempuan.

4. **Education (Tingkat Pendidikan)**
   Tingkat pendidikan terakhir yang ditempuh peserta dalam kategori: sekolah dasar, sekolah menengah pertama, sekolah menengah atas, diploma, sarjana, pascasarjana.

5. **Country (Negara Tempat Tinggal)**
   Negara tempat tinggal peserta.

6. **Ethnicity (Etnis)**
   Kelompok etnis peserta.

---

**Big Five Personality Traits**
Kelima dimensi kepribadian ini digunakan untuk menggambarkan sifat dan karakter seseorang secara umum:

7. **Neuroticism (Nscore)**
   Mengukur tingkat kecenderungan mengalami emosi negatif seperti cemas, marah, dan depresi.

8. **Extraversion (Escore)**
   Mengukur tingkat keaktifan sosial, energi, dan kecenderungan mencari interaksi sosial.

9. **Openness to Experience (Oscore)**
   Mengukur tingkat rasa ingin tahu, imajinasi, dan keterbukaan terhadap pengalaman baru.

10. **Agreeableness (Ascore)**
    Mengukur tingkat sifat ramah, empati, dan kepercayaan kepada orang lain.

11. **Conscientiousness (Cscore)**
    Mengukur tingkat disiplin, tanggung jawab, dan ketelitian.

---

**Skor Tambahan Psikologis**

12. **Impulsivity (Impulsive)**
    Mengukur kecenderungan bertindak tanpa berpikir matang terlebih dahulu.

13. **Sensation Seeking (SS)**
    Mengukur kecenderungan mencari pengalaman dan sensasi baru yang intens.

---

**Variabel konsumsi zat (setiap variabel mewakili jenis zat):**
Nilai variabel ini bersifat kategorikal dan menggambarkan riwayat penggunaan zat oleh responden dengan kategori:

* CL0: Never (Tidak pernah menggunakan)
* CL1: Used over a decade ago (Pernah menggunakan lebih dari 10 tahun yang lalu)
* CL2: Used in last decade (Pernah menggunakan dalam 10 tahun terakhir)
* CL3: Used in last year (Pernah menggunakan dalam 1 tahun terakhir)
* CL4: Used in last month (Pernah menggunakan dalam 1 bulan terakhir)
* CL5: Used in last week (Pernah menggunakan dalam 1 minggu terakhir)
* CL6: Used in last day (Pernah menggunakan dalam 1 hari terakhir)

14. **Alcohol**
    Status penggunaan alkohol oleh peserta.

15. **Amphetamines**
    Status penggunaan amfetamin.

16. **Amyl Nitrite**
    Status penggunaan amil nitrit.

17. **Benzodiazepines**
    Status penggunaan benzodiazepin.

18. **Caffeine**
    Status konsumsi kafein.

19. **Cannabis**
    Status penggunaan ganja.

20. **Chlorpromazine**
    Status penggunaan klorpromazin.

21. **Cocaine**
    Status penggunaan kokain.

22. **Crack**
    Status penggunaan crack.

23. **Ecstasy**
    Status penggunaan ekstasi.

24. **Heroin**
    Status penggunaan heroin.

25. **Ketamine**
    Status penggunaan ketamin.

26. **Legal Highs**
    Status penggunaan zat psikoaktif legal.

27. **LSD**
    Status penggunaan LSD.

28. **Methadone**
    Status penggunaan metadon.

29. **Magic Mushrooms**
    Status penggunaan jamur psikedelik.

30. **Nicotine**
    Status konsumsi nikotin.

31. **Semer**
    Status penggunaan semer.

32. **VSA (Volatile Substance Abuse)**
    Status penyalahgunaan zat volatil.

--- 
**Tabel. Tipe Data**
| #   | Column           | Non-Null Count | Dtype   |
|-----|------------------|----------------|---------|
| 0   | Age              | 1885           | object  |
| 1   | Gender           | 1885           | object  |
| 2   | Education        | 1885           | object  |
| 3   | Country          | 1885           | object  |
| 4   | Ethnicity        | 1885           | object  |
| 5   | Nscore           | 1885           | float64 |
| 6   | Escore           | 1885           | float64 |
| 7   | Oscore           | 1885           | float64 |
| 8   | Ascore           | 1885           | float64 |
| 9   | Cscore           | 1885           | float64 |
| 10  | Impulsive        | 1885           | float64 |
| 11  | SS               | 1885           | float64 |
| 12  | Alcohol          | 1885           | object  |
| 13  | Amphetamines     | 1885           | object  |
| 14  | Amyl Nitrite     | 1885           | object  |
| 15  | Benzodiazepines  | 1885           | object  |
| 16  | Caffeine         | 1885           | object  |
| 17  | Cannabis         | 1885           | object  |
| 18  | Chlorpromazine   | 1885           | object  |
| 19  | Cocaine          | 1885           | object  |
| 20  | Crack            | 1885           | object  |
| 21  | Ecstasy          | 1885           | object  |
| 22  | Heroin           | 1885           | object  |
| 23  | Ketamine         | 1885           | object  |
| 24  | Legal Highs      | 1885           | object  |
| 25  | LSD              | 1885           | object  |
| 26  | Methadone        | 1885           | object  |
| 27  | Magic Mushrooms  | 1885           | object  |
| 28  | Nicotine         | 1885           | object  |
| 29  | Semer            | 1885           | object  |
| 30  | VSA              | 1885           | object  |

* Diketahui bahwa tidak terdapat missing values dalam dataset ini, karena seluruh kolom memiliki 1.885 nilai non-null.

* Tipe data yang digunakan terdiri dari float64 untuk variabel numerik (seperti skor kepribadian,  impulsivitas, dan Sensation seeking), serta object untuk variabel kategorikal (seperti variabel demografi dan status penggunaan zat).

Hal ini menunjukkan bahwa data tidak memerlukan penanganan nilai hilang.

---
**Tabel. Deskriptif Data**
| Statistic | Nscore    | Escore    | Oscore    | Ascore    | Cscore    | Impulsive | SS        |
|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| count     | 1885.0000 | 1885.0000 | 1885.0000 | 1885.0000 | 1885.0000 | 1885.0000 | 1885.0000 |
| mean      | 0.0000    | -0.0002   | -0.0005   | -0.0002   | -0.0004   | 0.0072    | -0.0033   |
| std       | 0.9981    | 0.9974    | 0.9962    | 0.9974    | 0.9975    | 0.9544    | 0.9637    |
| min       | -3.4644   | -3.2739   | -3.2739   | -3.4644   | -3.4644   | -2.5552   | -2.0785   |
| 25%       | -0.6783   | -0.6951   | -0.7173   | -0.6063   | -0.6525   | -0.7113   | -0.5259   |
| 50%       | 0.0426    | 0.0033    | -0.0193   | -0.0173   | -0.0067   | -0.2171   | 0.0799    |
| 75%       | 0.6297    | 0.6378    | 0.7233    | 0.7610    | 0.5849    | 0.5298    | 0.7654    |
| max       | 3.2739    | 3.2739    | 2.9016    | 3.4644    | 3.4644    | 2.9016    | 1.9217    |

Hasil deskriptif data menunjukkan bahwa seluruh variabel numerik memiliki 1.885 data, tanpa missing values. Skor kepribadian, impulsivitas, dan sensation seeking sudah dalam bentuk z-score (mean ≈ 0, std ≈ 1), menandakan data telah dinormalisasi. Nilai minimum dan maksimum yang cukup ekstrim mengindikasikan potensi outlier. Distribusi terlihat simetris (median ≈ 0)

### **Univariate Analysis**

![KDE variabel numerik](https://github.com/user-attachments/assets/dbbdb7b2-a49d-48f0-af9d-0bd84072db75)

Gambar di atas menampilkan histogram dengan kurva Kernel Density Estimation (KDE) untuk setiap variabel numerik dalam dataset, sebagian besar variabel kepribadian (Nscore hingga Cscore) menunjukkan pola distribusi yang cukup simetris dan menyerupai distribusi normal, yang sesuai dengan karakteristik data yang telah dinormalisasi. Sedangkan variabel Impulsive dan Sensation Seeking (SS) terlihat sedikit menyimpang dari normal, dengan bentuk yang lebih miring (skewed), menunjukkan adanya variasi atau ketidakseimbangan dalam persebaran nilai responden.

![Distribusi variabel kategorik](https://github.com/user-attachments/assets/00986d98-f1d2-4725-84e7-f04d38ce5f18)

Visualisasi barplot di atas menampilkan distribusi frekuensi dari seluruh variabel kategorikal yang ada dalam dataset.
1. Age

- Mayoritas responden berusia 18–24 tahun, diikuti oleh 25–34 tahun.
- Kelompok usia di atas 55 tahun jauh lebih sedikit.

2. Gender

- Komposisi hampir seimbang antara laki-laki dan perempuan.

3. Education

- Mayoritas berpendidikan tinggi, terutama some college dan university degree.
- Pendidikan rendah (<18 tahun) dan doktoral jumlahnya kecil.

4. Country

- Responden paling banyak berasal dari UK dan USA
- Negara lain seperti Australia, Ireland, dan New Zealand cukup kecil proporsinya.

5. Ethnicity

- Mayoritas adalah etnis White, kelompok etnis lain relatif sedikit.

6. Penggunaan Obat (per Zat)

- Caffeine dan Alcohol adalah dua zat yang paling sering digunakan oleh responden.

  - Caffeine didominasi oleh kategori CL6 (Used in last day) sebanyak 1385 orang dan CL5 (Used in last week) sebanyak 273 orang.
  - Alcohol juga tinggi, dengan dominasi di CL5 (Used in last week) sebanyak 759 orang dan CL6 (Used in last day) sebanyak 505 orang.

- Cannabis dan Nicotine menunjukkan pola distribusi yang relatif merata:

  - Keduanya memiliki pengguna yang tersebar di hampir semua kategori dari CL0 sampai CL6.
  - Cannabis tertinggi di CL6 (463) dan CL0 (413), menandakan sebagian besar responden rutin menggunakan, sebagian lainnya tidak pernah.
  - Nicotine tertinggi di CL6 (610) dan CL0 (428), menunjukkan tren serupa.

- Beberapa zat seperti Amphetamines, Benzodiazepines, Ecstasy, LSD, Legal Highs, dan Magic Mushrooms juga cukup bervariasi, namun jumlah terbesar tetap berada di CL0 (tidak pernah menggunakan).

- Heroin, Semer, dan Crack menunjukkan dominasi kuat di CL0:

  - Heroin: 1605 responden di CL0
  - Semer: 1877 responden di CL0
  - Crack: 1627 responden di CL0
    
    Hal ini mengindikasikan bahwa sebagian besar responden tidak memiliki pengalaman dengan zat-zat tersebut.

### **Multivariate Analysis**
#### **Korelasi antar fitur numerik**

![Korelasi antar fitur numerik](https://github.com/user-attachments/assets/1a30a834-c32c-44f4-83bb-9aee6f1892e2)

Visualisasi Heatmap di atas menunjukkan korelasi antara variabel numerik sebagai berikut.

**1. Korelasi Tinggi:**
- Impulsivity (Impulsive) & Sensation Seeking (SS): korelasi 0.62 → makin impulsif seseorang, makin tinggi Sensation Seeking-nya.

- Openness to Experience (Oscore) & Sensation Seeking (SS): korelasi 0.42 → orang dengan openness tinggi cenderung lebih sensation seeking.

- Extraversion (Escore) & Conscientiousness (Cscore): korelasi 0.30 → ekstrovert sedikit cenderung lebih conscientious.

**2. Korelasi Negatif:**
- Neuroticism (Nscore) & Extraversion (Escore): -0.43 → neurotik cenderung kurang ekstrovert.

- Neuroticism (Nscore) & Conscientiousness (Cscore): -0.38 → neurotik cenderung tidak conscientious.

- Conscientiousness (Cscore) & Impulsivity (Impulsive): -0.33 → makin conscientious, makin tidak impulsif.

- Conscientiousness (Cscore) & Sensation Seeking (SS): -0.24 → conscientious rendah cenderung lebih sensation seeking.

**3. Korelasi Lemah:**
- Korelasi antar sebagian besar skor (Openness to Experience (Oscore) dan Agreeableness (Ascore)) tergolong lemah atau tidak signifikan secara praktis.

#### **Personality Traits vs Drug Use Alcohol**

![Personality traits VS Alcohol use categories](https://github.com/user-attachments/assets/d878f85f-0956-4ab6-9d81-b19a1a987157)

Berdasarkan visualisasi pointplot antara skor kepribadian (Big Five) dengan kategori penggunaan alkohol (CL0–CL6), ditemukan beberapa pola yang cukup konsisten:

1. Nscore (Neuroticism)

- Titik cenderung di bawah 0 untuk CL0–CL1 (tidak atau jarang konsumsi), naik sedikit di CL2–CL4, lalu datar.
- Pengguna alkohol ringan cenderung punya skor neurotik lebih rendah, tapi tidak ada tren jelas meningkat untuk pengguna berat.

2. Escore (Extraversion)

- Skor cenderung negatif di CL2–CL3, lalu kembali naik di CL5.
- Ada penurunan sementara untuk peminum sedang, tapi pengguna rutin cenderung lebih ekstrovert.

3. Oscore (Openness)

- Rata-rata skor relatif stabil, sedikit naik di CL4–CL6.
- Tidak ada pola ekstrim, tapi peminum sering/harian sedikit lebih terbuka terhadap pengalaman baru.

4. Ascore (Agreeableness)

- Titik tertinggi di CL1, turun di CL2-CL6.
- Pengguna alkohol cenderung punya skor Agreeableness lebih rendah, terutama yang makin rutin konsumsi.

5. Cscore (Conscientiousness)

- Jelas terlihat tinggi di CL0 dan CL1. menurun di CL2-CL6.
- Artinya: Semakin sering konsumsi alkohol, cenderung semakin rendah skor kedisiplinan/tanggung jawabnya.

**Kesimpulan**

- Neuroticism & Extraversion: Fluktuatif, tapi tidak ada pola kuat yang signifikan.
- Openness: Agak naik pada peminum berat.
- Agreeableness & Conscientiousness*: Jelas lebih rendah pada pengguna alkohol, terutama rutin (CL4 ke atas).

#### **Distribusi dan hubungan variabel demografi terhadap status penggunaan zat Alkohol**

![Distribusi variabel demografi terhadap penggunaan alkohol](https://github.com/user-attachments/assets/e0aed2f8-8f51-4d53-b8e2-f2751461a6b9)

Berdasarkan visualisasi barplot demografi terhadap status penggunaan alcohol, dapat diketahui sebagai berikut

1. Distribusi Gender terhadap Alkohol

  Grafik ini menunjukkan bahwa baik pria maupun wanita paling banyak berada pada kategori konsumsi alkohol CL5 (tertinggi), disusul CL6. Pria cenderung sedikit lebih banyak pada kategori CL6, sedangkan wanita lebih dominan pada CL5. Ini mengindikasikan bahwa konsumsi alkohol tinggi terjadi pada kedua gender, dengan kecenderungan lebih tinggi pada wanita di kategori tertinggi.

2. Distribusi Umur terhadap Alkohol

  Kelompok umur 18–24 tahun adalah yang paling dominan dalam konsumsi alkohol tinggi (CL5 dan CL6), disusul kelompok 25–34 tahun. Semakin tua kelompok umur, konsumsi alkohol cenderung menurun. Ini menandakan bahwa konsumsi alkohol tinggi lebih umum pada usia muda.

3. Distribusi Pendidikan terhadap Alkohol

  Orang dengan pendidikan “University degree” paling banyak dalam kategori konsumsi alkohol tinggi (CL5), disusul oleh yang hanya memiliki “Some college/university, no degree.” Sebaliknya, mereka yang berhenti sekolah di usia muda cenderung lebih sedikit mengonsumsi alkohol. Artinya, semakin tinggi pendidikan, kecenderungan konsumsi alkohol juga meningkat.
  
#### **Personality vs Drug Use**

![Personality VS use drugs](https://github.com/user-attachments/assets/27e7cdc7-32d8-4ae3-81d5-73755c1f8cf3)

Visualisasi Heatmap di atas menunjukkan hubungan (korelasi) antara tujuh dimensi kepribadian dan penggunaan berbagai jenis zat (obat-obatan atau narkoba).
1. Oscore (Openness to experience)
   Korelasi paling tinggi di antara semua dimensi kepribadian, terutama dengan LSD (0.37), Magic Mushrooms (0.37), dan Cannabis (0.41). Hal ini menunjukkan bahwa individu yang terbuka pada pengalaman baru cenderung lebih mungkin menggunakan zat-zat ini.

2. SS (Sensation Seeking)
   Korelasi positif kuat dengan berbagai zat, terutama Cannabis (0.46), LSD (0.41), dan Ecstasy (0.39). Artinya, individu yang mencari sensasi lebih tinggi lebih rentan menggunakan berbagai jenis zat.

3. Impulsive
   Juga memiliki korelasi cukup positif terhadap zat-zat seperti Cannabis (0.31), Crack (0.26), dan LSD (0.27), menunjukkan bahwa sifat impulsif bisa menjadi faktor risiko penggunaan zat.

4. Cscore (Conscientiousness) dan Ascore (Agreeableness)
   Cenderung berkorelasi negatif secara konsisten terhadap hampir semua zat. Artinya, orang yang teliti dan mudah bekerja sama cenderung lebih rendah risikonya dalam menggunakan zat.

5. Nscore (Neuroticism) dan Escore (Extraversion)
   Hubungannya lemah atau tidak konsisten terhadap sebagian besar zat, menunjukkan pengaruh yang lebih kecil atau bervariasi tergantung jenis zatnya.

**Kesimpulan:**
Sifat kepribadian seperti keterbukaan terhadap pengalaman baru, pencarian sensasi, dan impulsivitas memiliki hubungan yang paling kuat terhadap penggunaan zat. Sedangkan, sifat kehati-hatian dan keramahan menunjukkan efek protektif. Visualisasi ini penting untuk memahami faktor psikologis dalam pencegahan dan penanganan penyalahgunaan zat.


## Data Preparation
### **1. Menghapus fitur ID**
Pada tahap awal pemuatan data (data loading), kolom identifikasi (ID) dihapus karena tidak memiliki kontribusi informasi terhadap proses analisis maupun pemodelan. Kolom ini hanya berfungsi sebagai penanda unik untuk setiap entri dan tidak mengandung nilai yang relevan untuk dianalisis secara statistik atau digunakan dalam model prediktif. 

Alasan: Penghapusan kolom ini adalah karena ID bersifat unik untuk setiap individu, sehingga tidak merepresentasikan pola atau hubungan apa pun dengan variabel target. Bahkan, menyertakan kolom seperti ini dapat menyebabkan model machine learning salah mengenali pola dan meningkatkan risiko overfitting, terutama pada algoritma berbasis pohon keputusan. Oleh karena itu, penghapusan dilakukan sejak awal agar **proses eksplorasi data** menjadi lebih efisien dan fokus hanya pada fitur-fitur yang bersifat informatif.

### **2. Pemeriksaan Outlier dan Penanganannya**

Outlier adalah nilai-nilai ekstrem pada beberapa variabel numerik yang dapat memengaruhi distribusi data dan hasil analisis. Penanganan outlier dilakukan untuk memastikan kualitas data tetap terjaga dan model prediktif dapat bekerja secara optimal.

Alasan:

1. Outlier perlu dianalisis secara menyeluruh sebelum dilakukan penanganan agar tidak menghapus data yang sebenarnya valid dan penting.
2. Penanganan outlier yang terlalu dini berisiko menghilangkan informasi yang relevan dan mengubah distribusi data secara signifikan.
3. Outlier yang tidak ditangani dapat menyebabkan distorsi distribusi data dan menurunkan performa model, terutama pada algoritma yang sensitif terhadap nilai ekstrem.

![Sebelum penanganan outlier](https://github.com/user-attachments/assets/4ebe046f-d5b7-4f15-8245-762b0d29a69d)

Sebelum dilakukan penanganan, visualisasi boxplot pada lima dimensi skor kepribadian (Nscore, Escore, Oscore, Ascore, dan Cscore) dan impulsivitas menunjukkan adanya sejumlah outlier. Hal ini terlihat dari adanya titik-titik di luar rentang whisker pada boxplot. Meskipun data telah dinormalisasi, nilai-nilai ekstrem masih tetap dapat terdeteksi karena proses normalisasi tidak menghilangkan outlier.

Penanganan dilakukan menggunakan metode Z-score, di mana data dengan nilai Z lebih dari 3 dianggap sebagai outlier. Alih-alih menghapus data tersebut, nilai outlier digantikan dengan nilai median pada masing-masing kolom. Pendekatan ini dipilih karena:

- Jumlah data cukup terbatas (n = 1885) sehingga menghapus baris yang mengandung outlier berisiko mengurangi variasi dan informasi dalam data.

- Mengganti outlier dengan median tetap menjaga ukuran sampel dan mencegah hilangnya pola yang mungkin penting untuk analisis selanjutnya.

- Median lebih robust terhadap outlier, sehingga dapat mengurangi distorsi tanpa menggeser distribusi secara ekstrem.

![Setelah penanganan outlier](https://github.com/user-attachments/assets/c34dcc73-39e4-435d-a19b-053e73e28ab4)

Setelah penanganan, boxplot menunjukkan distribusi yang lebih merata dan jumlah outlier yang jauh berkurang, terkhusus untuk Impulsive yang tidaak terdapat outlier lagi. Hal ini menunjukkan bahwa proses penggantian berhasil mereduksi nilai-nilai ekstrem tanpa mengubah pola data secara signifikan.


### **3. Mengubah Age Menjadi Format Numerik**

Kolom Age awalnya berisi data dalam bentuk rentang usia (misalnya "25-34") atau kategori berbentuk string. Karena model machine learning, khususnya regresi, membutuhkan data numerik untuk melakukan perhitungan, maka rentang tersebut dikonversi menjadi nilai tengah dari masing-masing rentang agar bisa direpresentasikan sebagai angka.

Alasan:
Model seperti regresi tidak bisa langsung memproses data dalam bentuk rentang kategori (string), karena tidak merepresentasikan hubungan numerik yang jelas. Dengan mengubah rentang usia menjadi angka tengahnya, didapatkan estimasi usia yang lebih realistis dan bisa digunakan untuk analisis numerik terhadap target variabel.

---
Pada bagian ini, digunakan fungsi convert_age digunakan untuk membersihkan data usia dalam bentuk string, seperti '25-34', '65+', atau '30', menjadi nilai numerik tunggal, seperti berikut:

- Jika kosong (NaN) → dikembalikan sebagai np.nan.

- Jika rentang usia seperti '25-34' → dihitung rata-ratanya (misal: 29).

- Jika format '65+' → diambil angkanya saja (misal: 65).

- Jika sudah angka biasa (misal '30') → langsung dikonversi ke integer.

- Jika gagal diproses → dikembalikan sebagai np.nan.

Fungsi ini diterapkan ke kolom 'Age' menggunakan df['Age'].apply(convert_age) untuk memastikan semua nilai usia dalam bentuk numerik.

### **4. Mapping Target ke Bentuk Numerik**


Target awal jenis obat berupa kategori teks, seperti berikut:
- CL0: Never (Tidak pernah menggunakan)
- CL1: Used over a decade ago (Pernah menggunakan lebih dari 10 tahun yang lalu)
- CL2: Used in last decade (Pernah menggunakan dalam 10 tahun terakhir)
- CL3: Used in last year (Pernah menggunakan dalam 1 tahun terakhir)
- CL4: Used in last month (Pernah menggunakan dalam 1 bulan terakhir)
- CL5: Used in last week (Pernah menggunakan dalam 1 minggu terakhir)
- CL6: Used in last day (Pernah menggunakan dalam 1 hari terakhir).

karena pemodelan regresi membutuhkan variabel numerik, maka kategori-kategori tersebut diubah menjadi angka ordinal yang mencerminkan tingkat frekuensi penggunaan.

Alasan:
Model regresi tidak dapat memproses data target dalam bentuk teks. Oleh karena itu, diperlukan konversi ke angka ordinal agar model bisa mengenali urutan tingkat frekuensi konsumsi, dari yang paling jarang (misalnya tidak pernah) hingga yang paling sering. Pendekatan ini membantu model memahami pola hubungan antara fitur-fitur input dengan tingkat konsumsi target secara lebih tepat.


### **5. Pemiilihan Fitur dan Target**
Fitur yang digunakan disesuaikan dengan rumusan masalah, yaitu terdiri dari:

- Fitur demografi: Age, Gender, Education, Country, Ethnicity

- Fitur psikologis: Nscore, Escore, Oscore, Ascore, Cscore, Impulsive, dan SS

Target yang diprediksi adalah frekuensi penggunaan untuk setiap jenis obat.

Alasan:
Pemilihan fitur dilakukan secara selektif untuk memastikan bahwa hanya variabel yang relevan dengan perilaku konsumsi obat yang digunakan. Fitur demografi dan psikologis dianggap memiliki pengaruh terhadap kebiasaan tersebut, sesuai dengan dasar teori dan rumusan masalah.
Selain itu, target juga harus dalam bentuk numerik agar dapat digunakan dalam model regresi. Hal ini memungkinkan model untuk belajar pola hubungan antara karakteristik individu dengan tingkat frekuensi penggunaan obat secara efektif.

### **6. Encoding Fitur Kategori**


Beberapa fitur seperti Gender, Education, Country, dan Ethnicity merupakan data kategorikal dalam bentuk string. Agar bisa digunakan dalam model machine learning, fitur-fitur ini perlu diubah ke format numerik menggunakan
OneHotEncoder. OneHotEncoder mengubah setiap kategori menjadi kolom biner (0 atau 1), sehingga tidak ada asumsi urutan di antara kategori tersebut. Misalnya, kategori Gender dengan nilai Male dan Female akan diubah menjadi dua kolom: Gender_Male dan Gender_Female.

Alasan:
Model machine learning, terutama yang berbasis numerik seperti regresi, tidak dapat memproses string secara langsung. Selain itu, jika kategori dikodekan sebagai angka biasa (seperti 1, 2, 3), model bisa keliru menganggap ada urutan atau jarak antar nilai. Dengan menggunakan OneHotEncoder:

- Tidak ada asumsi urutan antar kategori
- Menghindari bias model akibat interpretasi angka yang keliru
- Cocok untuk fitur kategorikal nominal
---
Pada bagian ini, dilakukan encoding fitur kategorikal (Gender, Education, Country, Ethnicity) menjadi vektor one-hot, dan menggabungkannya dengan fitur numerik lain (seperti Age, skor kepribadian, dll) menjadi satu DataFrame numerik.

### **7. Split Data Train-Test**

Data dibagi menjadi dua bagian:

- 80% untuk data latih (training set)
- 20% untuk data uji (test set)

Pembagian ini dilakukan untuk setiap target prediksi yang digunakan dalam pemodelan., dalam penelitian ini adalah setiap jenis obat.

Alasan:
Tujuan dari membagi data adalah agar model bisa dilatih pada satu bagian data latih (train set), lalu diuji performanya pada data yang belum pernah dilihat sebelumnya (test set). Manfaatnya adalah:

- Menghindari overfitting
- Memastikan model bisa beradaptasi dengan data baru yang belum pernah muncul sebelumnya
### **8. Transform Data dengan Preprocessing Pipeline**

Digunakan preprocessing pipeline untuk mengatur proses transformasi data, terutama pada fitur kategorikal.

- Fitur kategorikal seperti Gender, Education, Country, dan Ethnicity di-encode menggunakan OneHotEncoder.
- Fitur numerik langsung digunakan karena data sudah dinormalisasi sejak awal.

Alasan:
Menggunakan pipeline mempunyai banyak kelebihan:

- Mengotomatisasi proses encoding pada data train dan test, tanpa perlu transformasi manual.
- Menjaga konsistensi: transformasi yang dilakukan pada data latih akan secara otomatis diterapkan juga ke data uji, sehingga tidak terjadi mismatch.
- Memudahkan proses modeling, baik saat pelatihan maupun saat prediksi di data baru.
- Fitur numerik tetap utuh, karena pipeline bisa diatur hanya memproses kolom yang dibutuhkan saja, tanpa mengubah yang sudah siap pakai.

## Modeling
### **Linear Regression**
Linear Regression adalah model statistik sederhana dan mudah diinterpretasikan. Model ini mengasumsikan adanya hubungan linear antara fitur (X) dan target (y).

Kelebihan:

- Mudah diinterpretasikan: bisa tahu seberapa besar pengaruh tiap fitur ke target.
- Cepat dilatih, bahkan pada dataset besar.
- Cocok untuk baseline model karena sederhana.

Kekurangan:

- Tidak cocok untuk hubungan non-linear.
- Sangat sensitif terhadap outlier.
- Asumsi multikolinearitas dan homoskedastisitas bisa sulit dipenuhi.


#### **Tahapan dan Parameter Pemodelan**
1. `LinearRegression()` dari `sklearn.linear_model`

  - Fungsi:

  Membuat model regresi linear yang digunakan untuk memprediksi nilai kontinu (numerik) berdasarkan fitur-fitur input.

  - Parameter default yang digunakan:

  ```python
  LinearRegression(
      fit_intercept=True, 
      copy_X=True, 
      n_jobs=None, 
      positive=False
  )
  ```

  Keterangan :

| Parameter       | Nilai Default | Penjelasan                                                                                                              |
| --------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `fit_intercept` | `True`        | Jika `True`, model akan menghitung intersep (bias). Jika `False`, model mengasumsikan bahwa data sudah berpusat di nol. |
| `copy_X`        | `True`        | Jika `True`, data X akan disalin sebelum diproses. Jika `False`, dapat mempercepat komputasi tapi berisiko mengubah X.  |
| `n_jobs`        | `None`        | Jumlah core CPU yang digunakan untuk komputasi. `None` berarti hanya satu core yang digunakan.                          |
| `positive`      | `False`       | Jika `True`, model membatasi koefisien agar bernilai positif saja. Digunakan untuk kasus khusus seperti prediksi harga. |


2. `model.fit(X_train_processed, y_train_i)`

  - Fungsi:

  Melatih (training) model Linear Regression menggunakan data fitur (`X_train_processed`) dan target (`y_train_i`).

3. `model.predict(X_test_processed)`

  - Fungsi:

  Menggunakan model yang sudah dilatih untuk memprediksi nilai target dari data uji (`X_test_processed`).


4. Metrik Evaluasi:

  Ketiga metrik ini digunakan untuk mengukur performa model pada data uji (`y_test_i` vs `y_pred`):

  a)  `mean_absolute_error(y_test_i, y_pred)`

  * Fungsi: Mengukur rata-rata dari selisih absolut antara nilai prediksi dan nilai aktual.
  * Interpretasi: Semakin kecil MAE, semakin akurat prediksi model.
  * Rumus:

  $$
  MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
  $$


b) `mean_squared_error(y_test_i, y_pred)`

  * Fungsi: Mengukur rata-rata dari kuadrat selisih antara prediksi dan aktual.
  * Interpretasi: Lebih sensitif terhadap outlier dibanding MAE.
  * Rumus:

  $$
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$

c)  `r2_score(y_test_i, y_pred)`

  * Fungsi: Koefisien determinasi (R²), mengukur seberapa baik model menjelaskan variabilitas target.

    * Nilai mendekati 1: Model menjelaskan variabilitas dengan baik.
    * Nilai 0: Model tidak lebih baik dari rata-rata.
    * Nilai negatif: Model lebih buruk dari model rata-rata.
  * Rumus:

  $$
  R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
  $$

5. `model.coef_`

* Fungsi: Menyimpan nilai koefisien (bobot) dari masing-masing fitur pada model.

  * Positif: Fitur berkontribusi menaikkan nilai prediksi.
  * Negatif: Fitur berkontribusi menurunkan nilai prediksi.
  * Nilai mendekati 0: Pengaruh fitur kecil atau tidak signifikan.

6. `pd.DataFrame(coefs_lr, columns=all_cols)`

  * Fungsi: Menyusun semua koefisien dari model ke dalam DataFrame untuk setiap target (misalnya, jenis zat tertentu).
  * Ditambahkan kolom:

    * `"Drug"`: Nama target/zat yang diprediksi.
    * `"R²"`: Skor R² dari masing-masing model.
  * Reorganisasi kolom dilakukan untuk menyusun dataframe agar `Drug`, `R²`, dan semua koefisien fitur tampil berurutan.

7. Looping (`for i in range(1, 20):`)

  Fungsi:

  * Melakukan proses pelatihan dan evaluasi untuk **19 jenis zat (drug)** yang berbeda.
  * Asumsi: Anda memiliki 19 pasang `y_train_i` dan `y_test_i` untuk masing-masing zat.
  * Setiap iterasi menyimpan hasil evaluasi dan koefisien dari model Linear Regression tersebut.

---

#### Ringkasan

* Tujuan kode: Melatih 19 model regresi linear, satu untuk setiap zat, mengevaluasi performanya, dan menyimpan koefisien model.
* Parameter penting: `fit_intercept=True` untuk menangkap bias/intersep, `model.coef_` untuk analisis fitur, `r2_score` untuk kualitas model.
* Evaluasi dilakukan dengan MAE, MSE, dan R².
* Output akhir: Tabel koefisien yang lengkap untuk semua zat dengan skor R², digunakan untuk interpretasi model.

---
**Tabel. Ringkasan Koefisien Model Regresi Linear**

| Variabel                  | Alcohol | Amphetamines | Amyl Nitrite | Benzodiazepines | Caffeine | Cannabis | Chlorpromazine | Cocaine | Crack | Ecstasy | Heroin | Ketamine | Legal Highs | LSD   | Methadone | Magic Mushrooms | Nicotine | Semer | VSA   |
|---------------------------|---------|--------------|--------------|------------------|----------|----------|-----------------|---------|-------|---------|--------|----------|--------------|-------|-----------|------------------|----------|-------|-------|
| **Nscore**                | 0.0351  | 0.0660       | 0.0189       | 0.3878           | 0.0161   | -0.1036  | 0.0053          | 0.0767  | 0.0455| -0.0512 | 0.0820 | 0.0031   | 0.0052        | -0.0298| 0.1058    | -0.0855          | 0.0970   | 0.0014| 0.0234|
| **Escore**                | 0.1163  | 0.0672       | 0.0469       | 0.0309           | 0.0591   | -0.1069  | -0.0087         | 0.1117  | -0.0038| 0.1375 | -0.0335| 0.0148   | -0.0543       | 0.0003 | -0.0816   | -0.0038          | 0.0302   | 0.0029| -0.0191|
| **Oscore**                | 0.0172  | 0.0365       | 0.0070       | 0.1212           | -0.0163  | 0.4616   | 0.0853          | 0.0160  | -0.0010| 0.1546 | 0.0449 | 0.1219   | 0.2220        | 0.2596 | 0.0822    | 0.2282           | 0.2038   | -0.0013| 0.0517|
| **Ascore**                | -0.0017 | -0.0532      | -0.0327      | -0.0741          | -0.0156  | -0.0077  | -0.0225         | -0.1143 | -0.0004| -0.0150| -0.0686| -0.0286  | -0.0297       | 0.0198 | -0.1006   | 0.0204           | 0.0263   | 0.0052| -0.0306|
| **Cscore**                | -0.0527 | -0.1658      | -0.1011      | -0.0328          | -0.0379  | -0.2101  | -0.0352         | -0.1176 | -0.0380| -0.2027| -0.0087| -0.1282  | -0.1544       | -0.0427| -0.0445   | -0.1346          | -0.2156  | 0.0034| -0.0597|
| **Impulsivity**           | -0.0384 | 0.1611       | -0.0200      | 0.0585           | 0.0459   | 0.0794   | 0.0098          | 0.0357  | 0.0566 | 0.0272 | 0.0466 | 0.0053   | -0.0202       | -0.0135| 0.0461    | 0.0593           | 0.1935   | -0.0072| 0.0139|
| **Sensation Seeking (SS)**| 0.1748  | 0.1403       | 0.2043       | 0.1368           | 0.0270   | 0.3430   | -0.0279         | 0.3027  | 0.0458 | 0.2574 | 0.0642 | 0.1202   | 0.3074        | 0.1554 | 0.0368    | 0.1275           | 0.2560   | 0.0085| 0.0914|
| **Gender_Male**           | 0.0447  | 0.1931       | 0.1740       | 0.1230           | 0.0257   | 0.2811   | -0.0461         | 0.1157  | 0.0779 | 0.1929 | 0.0522 | 0.1650   | 0.3034        | 0.2221 | 0.1365    | 0.2091           | 0.2340   | -0.0039| 0.0227|
| **Gender_Female**         | -0.0447 | -0.1931      | -0.1740      | -0.1230          | -0.0257  | -0.2811  | 0.0461          | -0.1157 | -0.0779| -0.1929| -0.0522| -0.1650  | -0.3034       | -0.2221| -0.1365   | -0.2091          | -0.2340  | 0.0039 | -0.0227|
| **Age**                   | -0.0036 | -0.0108      | -0.0070      | 0.0087           | 0.0064   | -0.0374  | 0.0003          | -0.0130 | 0.0032 | -0.0312| 0.0009 | -0.0148  | -0.0309       | -0.0124| -0.0064   | -0.0137          | -0.0226  | -0.0002| -0.0080|
| **Education_Left<16**     | -0.6433 | 0.1473       | -0.1494      | 0.1389           | -0.4049  | 0.9481   | 0.1890          | 0.1346  | 0.1380 | 0.1239 | 0.0124 | -0.0361  | 0.0899        | 0.2851 | 0.2590    | 0.3804           | 1.0041   | -0.0071| -0.0946|
| **Education_Masters**     | 0.3876  | -0.0611      | 0.1084       | -0.0354          | 0.1673   | -0.5020  | -0.0001         | -0.0866 | -0.1467| -0.2281| -0.1109| -0.0433  | -0.1376       | -0.1576| -0.2049   | -0.1575          | -0.5756  | -0.0102| -0.0765|
| **Ethnicity_Other**       | 0.2623  | 0.5466       | 0.3539       | -0.0250          | 0.2164   | 0.2852   | -0.0359         | 0.5176  | 0.0950 | 0.6960 | -0.0282| 0.3542   | 0.4226        | -0.1584| -0.3221   | 0.0822           | 0.5977   | 0.0312 | -0.0617|
| **Ethnicity_White**       | 0.3943  | 0.3563       | 0.2699       | -0.1864          | 0.2412   | 0.2701   | 0.0444          | 0.1582  | -0.0040| 0.4311 | -0.0193| 0.1035   | 0.2125        | -0.1456| 0.1292    | 0.0677           | 0.4108   | -0.0342| -0.1258|

Setiap model regresi dilakukan untuk masing-masing dari 19 jenis zat, dan koefisien menunjukkan arah serta kekuatan pengaruh masing-masing variabel terhadap frekuensi penggunaan zat.

#### **Variabel Psikologis Kepribadian**

**1. Neuroticism (Nscore)**

Cenderung positif pada zat-zat seperti Benzodiazepines (0.3878), Heroin (0.0820), dan Nicotine (0.0970), menunjukkan bahwa individu dengan tingkat neurotisisme tinggi lebih rentan menggunakan zat sebagai pelarian dari stres atau tekanan emosional.

**2. Extraversion (Escore)**

Bervariasi, tapi terlihat positif pada Cocaine (0.1117), Ecstasy (0.1375), dan Nicotine (0.0302), menunjukkan bahwa individu ekstrovert yang suka stimulasi sosial lebih cenderung menggunakan zat yang berkaitan dengan euforia atau aktivitas sosial.

**3. Openness to Experience (Oscore)**

Sangat positif pada Cannabis (0.4616), LSD (0.2596), dan Magic Mushrooms (0.2282), memperkuat hubungan antara keterbukaan terhadap pengalaman baru dengan eksplorasi penggunaan zat-zat psikedelik atau alternatif.

**4. Agreeableness (Ascore)**

Negatif untuk banyak zat seperti Cocaine (-0.1143), Benzodiazepines (-0.0741), dan Heroin (-0.0686), menandakan bahwa orang yang cenderung kooperatif dan penuh empati kurang tertarik terhadap penggunaan zat-zat tersebut.

**5. Conscientiousness (Cscore)**

Konsisten negatif, misalnya pada Cannabis (-0.2101), Ecstasy (-0.2027), dan Nicotine (-0.2156). Ini menunjukkan bahwa orang yang terorganisir dan berhati-hati cenderung menghindari perilaku berisiko seperti penggunaan zat.


#### **Impulsivity & Sensation Seeking**

**1. Impulsivity**

Positif dan cukup kuat pada zat-zat seperti Amphetamines (0.1611), Cannabis (0.0794), dan Nicotine (0.1935), menunjukkan bahwa individu yang lebih impulsif cenderung lebih sering menggunakan zat karena dorongan sesaat atau kontrol diri yang rendah.

**2. Sensation Seeking (SS)**
Sangat tinggi pada Cannabis (0
.3430), Cocaine (0.3027), Legal Highs (0.3074), dan Ecstasy (0.2574), menunjukkan bahwa mereka yang haus pengalaman intens dan baru lebih rentan terhadap penggunaan zat dengan efek kuat atau menantang.


#### **Variabel Demografis**

**1. Jenis Kelamin**

- Pria (Gender\_Male): Positif untuk hampir semua zat, misalnya Cannabis (0.2811), Legal Highs (0.3034), dan Amphetamines (0.1931)
- Wanita (Gender\_Female): Negatif (karena dummy variabel), artinya pria lebih cenderung menggunakan zat daripada wanita

**2. Usia (Age)**

Umumnya negatif pada zat-zat seperti Cannabis (-0.0374), Ecstasy (-0.0312), dan Legal Highs (-0.0309), mengindikasikan bahwa kelompok usia muda memiliki kecenderungan lebih tinggi untuk mengonsumsi zat tersebut.

**3. Pendidikan (Education)**

- **Left school before 16 years**: positif besar pada Cannabis (0.9481) dan Nicotine (1.0041), memperlihatkan bahwa pendidikan rendah merupakan faktor risiko kuat
- **Masters degree**: efek cenderung negatif, seperti Cannabis (-0.5020) dan Nicotine (-0.5756), menunjukkan bahwa pendidikan tinggi dapat menjadi faktor pelindung dari penggunaan zat

**4. Etnisitas (Ethnicity)**

- **Ethnicity\_Other** positif besar pada zat seperti Cocaine (0.5176) dan Ecstasy (0.6960), artinya kelompok etnis tertentu mungkin memiliki pola penggunaan yang lebih tinggi, meskipun bisa juga dipengaruhi faktor sosial dan budaya
- **Ethnicity\_White** juga memiliki beberapa pengaruh positif, seperti pada Alcohol (0.3943), tapi umumnya lebih kecil dibandingkan kelompok *Other*


#### **Kesimpulan**

- Openness, sensation seeking, dan impulsivity jadi prediktor psikologis yang paling kuat untuk berbagai jenis zat, terutama yang bersifat eksploratif dan stimulatif
- Neuroticism berhubungan dengan zat yang dipakai sebagai coping mechanism, seperti benzodiazepine dan heroin
- Conscientiousness dan agreeableness tampak sebagai faktor pelindung, karena punya arah negatif hampir di semua jenis zat
- Dari sisi demografi, pria, usia muda, dan pendidikan rendah adalah kelompok yang punya risiko lebih tinggi terhadap penggunaan zat

### **Random Forest Regressor**
Random Forest adalah ensemble model berbasis decision tree yang membentuk banyak pohon (trees) dan menggabungkan hasilnya untuk prediksi akhir.

Kelebihan:

- Menangani hubungan non-linear dan interaksi antar fitur secara otomatis.
- Tidak sensitif terhadap multikolinearitas.
- Tidak memerlukan scaling atau normalisasi data.
- Dapat memberikan feature importance → membantu memahami pengaruh tiap variabel.

Kekurangan:

- Interpretasi model lebih sulit dibanding Linear Regression.
- Bisa overfit jika jumlah pohon terlalu banyak atau kedalaman pohon terlalu dalam.
- Waktu komputasi lebih tinggi dibanding model linear.


#### **Tahapan Pemodelan dan Parameter**
1. `RandomForestRegressor()` dari `sklearn.ensemble`

  - Fungsi:

  Membuat model regresi ensambel berbasis decision tree yang menggabungkan banyak pohon untuk meningkatkan akurasi dan mengurangi overfitting.

  - Parameter yang digunakan:

  ```python
  RandomForestRegressor(random_state=42)
  ```

| Parameter      | Nilai | Fungsi                                                                                      |
| -------------- | ----- | ------------------------------------------------------------------------------------------- |
| `random_state` | `42`  | Menetapkan seed agar hasil eksperimen bisa direproduksi (konsisten setiap kali dijalankan). |

> Catatan:hanya `random_state` yang ditentukan, parameter lainnya akan menggunakan **default value**.

- Parameter Default:

| Parameter           | Default           | Penjelasan                                                                                                          |
| ------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `n_estimators`      | `100`             | Jumlah pohon (trees) dalam forest. Lebih banyak biasanya meningkatkan performa, tapi juga menambah waktu komputasi. |
| `criterion`         | `"squared_error"` | Ukuran untuk membagi node (untuk regresi: MSE).                                                                     |
| `max_depth`         | `None`            | Tidak ada batasan kedalaman pohon, memungkinkan pertumbuhan hingga daun murni atau terlalu kecil.                   |
| `min_samples_split` | `2`               | Minimum sampel untuk membagi node internal.                                                                         |
| `min_samples_leaf`  | `1`               | Minimum sampel pada setiap daun.                                                                                    |
| `max_features`      | `"auto"`          | Jumlah fitur yang dipertimbangkan saat mencari split terbaik. Untuk regresi, default-nya adalah `n_features`.       |
| `bootstrap`         | `True`            | Menggunakan bootstrapping (sampling dengan pengembalian) untuk membangun setiap pohon.                              |
| `n_jobs`            | `None`            | Menentukan jumlah CPU core. `None` artinya hanya 1 core yang digunakan.                                             |
| `oob_score`         | `False`           | Jika `True`, model akan menghitung skor "Out-of-Bag" sebagai validasi internal.                                     |
| `verbose`           | `0`               | Tidak menampilkan log saat training.                                                                                |

2. `rf_model.fit(X_train_processed, y_train_i)`

  Fungsi:

  Melatih model Random Forest menggunakan fitur (`X_train_processed`) dan target (`y_train_i`).

  > Setiap iterasi dilakukan untuk 1 target zat (misalnya alkohol, heroin, dsb).

3. `rf_model.predict(X_test_processed)`

  Fungsi:

  Menggunakan model yang telah dilatih untuk memprediksi nilai pada data uji.

4. Metrik Evaluasi

  Digunakan untuk mengevaluasi akurasi model pada data uji.

  a) `mean_absolute_error(y_test_i, y_pred_rf)`

  * Mengukur rata-rata selisih absolut antara nilai prediksi dan aktual.
  * Semakin kecil nilainya, semakin baik prediksi model.

  b) `mean_squared_error(y_test_i, y_pred_rf)`

  * Mengukur rata-rata kuadrat dari selisih antara prediksi dan aktual.
  * Lebih sensitif terhadap kesalahan besar (outlier).

  c) `r2_score(y_test_i, y_pred_rf)`

  * Mengukur seberapa baik model menjelaskan variasi target.
  * Nilai berkisar dari -∞ hingga 1. Nilai lebih dekat ke 1 berarti prediksi bagus.

5. `rf_model.feature_importances_`

  Fungsi:

  Mengembalikan pentingnya setiap fitur dalam prediksi model (semakin besar nilainya, semakin penting fitur tersebut).

  > Metode ini berdasarkan berapa banyak fitur digunakan untuk memecah node dalam semua pohon dan seberapa besar pengurangan impurity yang terjadi.

6. `pd.DataFrame(importances_rf, columns=all_cols)`

  - Fungsi:

  Membuat tabel yang berisi:

  * Feature importances (importance setiap fitur dalam model)
  * Nama zat (`Drug`) sebagai target
  * Nilai R² (`R²`) sebagai indikator performa

7. Loop `for i in range(1, 20):`

* Melakukan iterasi 19 kali, masing-masing untuk satu jenis target (zat).
* Setiap model dilatih dan diuji secara terpisah.

---
**Tabel. Ringkasan Feature Importance**
#### **1. Faktor Psikologis Kepribadian**

| Trait Kepribadian | Zat Terkait                          | Nilai Importance |
|-------------------|--------------------------------------|------------------|
| Neuroticism       | Benzodiazepines                      | **0.1408**       |
|                   | Caffeine                             | **0.1283**       |
|                   | Chlorpromazine                       | **0.1274**       |
| Extraversion      | Cannabis                             | 0.0653           |
|                   | Ecstasy                              | 0.0884           |
|                   | LSD                                  | 0.0852           |
| Openness          | Cannabis                             | 0.0580           |
|                   | Ecstasy                              | 0.0974           |
|                   | LSD                                  | 0.0789           |
| Agreeableness     | Caffeine                             | 0.1155           |
|                   | Chlorpromazine                       | 0.1161           |
| Conscientiousness | Ketamine                             | 0.1327           |
|                   | Amyl Nitrite                         | 0.1237           |

Beberapa traits berperan penting dalam memprediksi perilaku konsumsi, di antaranya:

- Neuroticism (Nscore): Sangat tinggi untuk Benzodiazepines (0.1408), Caffeine (1283), dan Chlorpromazine (1274), mengindikasikan individu dengan neuroticism tinggi cenderung menggunakan zat yang bersifat penenang.

- Extraversion (Escore) & Openness (Oscore): Punya kontribusi besar untuk Cannabis, Ecstasy, dan LSD, zat-zat yang biasanya berhubungan dengan eksplorasi dan sosial.

- Agreeableness (Ascore) & Conscientiousness (Cscore): Umumnya menurunkan kemungkinan penggunaan, tapi nilainya tetap signifikan dalam memodelkan variasi perilaku.

---

#### **2. Sensation Seeking & Impulsiveness**

| Trait              | Zat Terkait                          | Nilai Importance |
|--------------------|--------------------------------------|------------------|
| Sensation Seeking  | Ecstasy                              | **0.1696**       |
|                    | Cocaine                              | **0.1599**       |
|                    | Cannabis                             | **0.1222**       |
|                    | Nicotine                             | **0.1199**       |
|                    | Legal Highs                          | **0.1170**       |
| Impulsiveness      | Amphetamines                         | 0.0622           |
|                    | Crack                                | 0.0680           |
|                    | LSD                                  | 0.0468           |

- Sensation Seeking (SS): Tinggi untuk Ecstasy (0.1696), cocaine (0.1599), Cannabis (0.1222), Nicotine (0.1199), dan Lagel Highs (0.1170) memperkuat temuan bahwa individu pencari sensasi cenderung mengeksplorasi berbagai jenis zat.

- Impulsiveness: Penting untuk zat-zat seperti Amphetamines , Crack, dan LSD, di mana keputusan impulsif bisa jadi pendorong utama konsumsi.

---

#### **3. Variabel Demografis**

| Variabel      | Zat Terkait         | Catatan Penting |
|---------------|---------------------|-----------------|
| Gender        | Semua zat           | Pengaruh kecil dan merata |
| Education     | Legal Highs, LSD, Ecstasy | Kontribusi kecil tapi konsisten |
| Ethnicity     | Semer               | Ethnicity\_Other sangat tinggi (0.1060) |
| Age           | Legal Highs         | **0.1544** (tinggi) |
|               | Ecstasy             | 0.1082 |

- Gender: Tidak terlalu berpengaruh secara signifikan untuk sebagian besar prediksi, namun tetap punya nilai kontribusi kecil yang merata.

- Education: Beberapa level pendidikan memberi kontribusi kecil, terutama untuk prediksi zat-zat seperti Ecstasy, LSD, dan Legal Highs.

- Ethnicity: Pengaruhnya rendah, namun Ethnicity\_Other cukup tinggi untuk Semer (0.1060), menandakan adanya pola konsumsi unik pada kelompok etnis minoritas.

- Age: Termasuk salah satu fitur penting untuk beberapa zat, terutama Legal Highs (0.1544) dan Ecstasy (0.1082), menunjukkan bahwa umur mempengaruhi kecenderungan konsumsi.

#### **Kesimpulan**

Model Random Forest Regressor berhasil mengidentifikasi fitur-fitur yang relevan dalam memprediksi konsumsi zat tertentu, terutama:

* Cannabis, Ecstasy, LSD, Legal Highs, dan Amphetamines: diprediksi cukup baik.
* Umur, traits psikologis (terutama Openness to Experience (Oscore), Neuroticism (Nscore), Sensation Seeking (SS)), dan pendidikan: jadi prediktor penting.

## Evaluation
### 1. **Metrik Evaluasi yang Digunakan**

Untuk mengukur performa dari model regresi yang dibangun (Linear Regression dan Random Forest), digunakan tiga metrik evaluasi utama, yaitu:

- **Mean Absolute Error (MAE):**
  
![MAE](https://github.com/user-attachments/assets/aa813013-358c-43ad-a417-94270e270999)
  
  Metrik ini mengukur rata-rata selisih absolut antara nilai aktual dan nilai prediksi. MAE mudah dipahami karena menggunakan satuan yang sama dengan target, dan tidak terlalu sensitif terhadap outlier.

- **Mean Squared Error (MSE):**

![MSE](https://github.com/user-attachments/assets/7cd5179f-7749-4937-8a81-0575ad67380a)

  MSE mengkuadratkan selisih prediksi agar penalti terhadap kesalahan besar menjadi lebih tinggi. Metrik ini lebih sensitif terhadap outlier dibanding MAE.

- **R-squared (R²):**
  
![R Squared](https://github.com/user-attachments/assets/10b23f4d-1d04-4e32-a213-96b915dca8ae)

  Metrik ini menunjukkan seberapa besar proporsi variasi dalam data target yang bisa dijelaskan oleh model. Nilai R² berkisar antara minus tak hingga sampai 1, dengan nilai mendekati 1 menandakan model yang lebih baik.


### 2. **Kesesuaian Metrik dengan Problem Statement**

Karena tujuan dari proyek ini adalah memprediksi tingkat penggunaan obat berdasarkan data demografi dan kepribadian, maka:

- MAE & MSE cocok digunakan untuk mengetahui seberapa besar prediksi meleset dari nilai sebenarnya.
- R² penting untuk melihat apakah model benar-benar menangkap pola dari data. Ini membantu mengevaluasi apakah model punya daya prediksi yang kuat.


### 3. **Penjelasan Hasil Proyek Berdasarkan Metrik**

Evaluasi dilakukan untuk setiap target zat (obat), yaitu dari y₁ sampai y₁₉. Untuk setiap target, dua model dibandingkan: Linear Regression dan Random Forest.

Setiap iterasi menyimpan hasil evaluasi ke dalam list results, yang kemudian dikonversi menjadi DataFrame eval_df. Struktur tabel mencakup:

Target: Nama variabel target (misalnya, y_1, y_2, ..., y_19)

- Drug: Nama zat terkait target

- Model: Tipe model regresi yang digunakan

- MAE, MSE, R²: Metrik evaluasi

### 4. **Penjelasan Kode Tambahan**

a) `results = []`

* List kosong yang akan menyimpan hasil evaluasi dari masing-masing model dan target.

b) Loop `for i in range(1, 20):`

* Iterasi sebanyak 19 kali, sesuai dengan jumlah target zat (`y_1` sampai `y_19`).

c) dalam setiap iterasi:

```python
mae_lr_i = mae_lr[i-1]
mse_lr_i = mse_lr[i-1]
r2_lr_i = r2_lr[i-1]
```

* Mengambil hasil evaluasi model Linear Regression dari list sebelumnya (hasil evaluasi yang sudah dihitung di script sebelumnya).

```python
mae_rf_i = mae_rf[i-1]
mse_rf_i = mse_rf[i-1]
r2_rf_i = r2_rf[i-1]
```

* Mengambil hasil evaluasi model Random Forest dari list sebelumnya.

d) Menambahkan hasil ke `results`

```python
results.append({...})
```

* Dua dictionary ditambahkan per iterasi: satu untuk Linear Regression, satu untuk Random Forest.

e) Membuat DataFrame:

```python
eval_df = pd.DataFrame(results)
```

* DataFrame `eval_df` adalah hasil akhir evaluasi yang berisi seluruh metrik untuk semua model dan target.

---

**Tabel. Hasil Evaluasi**
| Target | Drug            | MAE (LR) | MAE (RF) | MSE (LR) | MSE (RF) | R² (LR) | R² (RF) |
| ------ | --------------- | -------- | -------- | -------- | -------- | ------- | ------- |
| y\_1   | Alcohol         | 0.9645   | 1.0256   | 1.6855   | 1.9118   | 0.0420  | -0.0866 |
| y\_2   | Amphetamines    | 1.1287   | 1.1472   | 2.2274   | 2.3507   | 0.2211  | 0.1779  |
| y\_3   | Amyl Nitrite    | 0.7389   | 0.7386   | 0.9341   | 1.0037   | 0.1000  | 0.0329  |
| y\_4   | Benzodiazepines | 1.3445   | 1.3555   | 2.9198   | 3.0028   | 0.2048  | 0.1822  |
| y\_5   | Caffeine        | 0.7730   | 0.8085   | 1.3070   | 1.4225   | 0.0290  | -0.0569 |
| y\_6   | Cannabis        | 1.3019   | 1.3432   | 2.6100   | 2.7578   | 0.4886  | 0.4596  |
| y\_7   | Chlorpromazine  | 0.7207   | 0.7618   | 0.9506   | 1.0368   | 0.0052  | -0.0850 |
| y\_8   | Cocaine         | 1.1108   | 1.1475   | 1.8426   | 2.0862   | 0.1978  | 0.0917  |
| y\_9   | Crack           | 0.4908   | 0.4814   | 0.6212   | 0.6792   | 0.0423  | -0.0472 |
| y\_10  | Ecstasy         | 1.1111   | 1.0975   | 1.9406   | 2.0339   | 0.2391  | 0.2025  |
| y\_11  | Heroin          | 0.6071   | 0.6074   | 0.9770   | 1.0707   | 0.0967  | 0.0100  |
| y\_12  | Ketamine        | 0.8550   | 0.8638   | 1.4759   | 1.5331   | 0.1040  | 0.0693  |
| y\_13  | Legal Highs     | 1.0660   | 1.0210   | 1.9425   | 2.0792   | 0.3905  | 0.3476  |
| y\_14  | LSD             | 0.9399   | 0.8835   | 1.5475   | 1.5577   | 0.3016  | 0.2970  |
| y\_15  | Methadone       | 1.0305   | 1.0162   | 2.1287   | 2.1808   | 0.2161  | 0.1969  |
| y\_16  | Magic Mushrooms | 0.9603   | 0.9563   | 1.5673   | 1.6324   | 0.3086  | 0.2799  |
| y\_17  | Nicotine        | 1.8936   | 1.9726   | 4.9908   | 5.2177   | 0.1599  | 0.1217  |
| y\_18  | Semer           | 0.0243   | 0.0153   | 0.0109   | 0.0189   | -0.0294 | -0.7836 |
| y\_19  | VSA             | 0.6373   | 0.6689   | 1.0056   | 1.1373   | 0.1016  | -0.0160 |


### **Berikut ini adalah ringkasan informasi evaluasi model berdasarkan hasil di atas:**
### **Model Linear Regression**

Model ini menunjukkan performa paling baik dibandingkan Random Forest dalam mayoritas prediksi, terutama ditandai dengan nilai R² yang konsisten lebih tinggi.

Hasil terbaik ditunjukkan pada prediksi:

- Cannabis: R² = 0.4886 → hampir 49% variasi frekuensi penggunaan bisa dijelaskan model.
- Legal Highs: R² = 0.3905
- Magic Mushrooms: R² = 0.3086
- LSD: R² = 0.3016
- Ecstasy: R² = 0.2391
- Amphetamines: R² = 0.2211
- Methadone: R² = 0.2161
- Benzodiazepines: R² = 0.2048

Hasil ini menunjukkan bahwa Linear Regression mampu menangkap pola konsumsi pada zat yang lebih umum digunakan.

Namun, model ini kurang efektif untuk beberapa zat, seperti:

- Semer: R² = -0.0294
- Chlorpromazine: R² = 0.0052
- Caffeine: R² = 0.0290

Hal ini mengindikasikan bahwa model hampir tidak mampu menjelaskan variasi data untuk zat-zat tersebut.


### **Model Random Forest Regressor**

Performa cenderung lebih rendah dibandingkan Linear Regression pada sebagian besar target.


R² tertinggi tetap pada:

- Cannabis: R² = 0.4596, namun masih lebih rendah dibandingkan Linear Regression.
- Legal Highs: R² = 0.3476
- Magic Mushrooms: R² = 0.2799
- LSD: R² = 0.2970
- Ecstasy: R² = 0.2025

Namun, sering menghasilkan R² negatif, misalnya pada:

- Semer: R² = -0.7836 → terburuk, menunjukkan model benar-benar gagal mempelajari pola dari data.
- Alcohol: R² = -0.0866
- Caffeine: R² = -0.0569
- Chlorpromazine: R² = -0.0850
- Crack: R² = -0.0472
- VSA: R² = -0.0160

Model Random Forest kemungkinan mengalami overfitting atau tidak mampu menangkap pola konsisten.

### **Kesimpulan Evaluasi**

* Linear Regression lebih unggul dalam memodelkan pola konsumsi sebagian besar zat, khususnya yang konsumsinya lebih umum (seperti Cannabis, Ecstasy, Legal Highs).
* Random Forest tidak memberikan peningkatan performa berarti, bahkan cenderung lebih buruk, terutama pada zat dengan distribusi langka atau outlier.
* Nilai MAE dan MSE di kedua model umumnya berada di kisaran menengah, artinya masih ada potensi peningkatan akurasi prediksi, misalnya dengan tuning parameter atau penggunaan model lain seperti XGBoost, SVR, atau ensemble methods.
* Model juga dapat ditingkatkan dengan penyeimbangan data, terutama untuk zat yang kurang umum digunakan.


## Comparison & Model Seletion
### **Perbandingan Model (Model Comparison)**

Berdasarkan hasil evaluasi terhadap 19 jenis zat psikoaktif dengan dua model regresi (Linear Regression dan Random Forest Regressor), dapat disimpulkan bahwa:

1. Linear Regression menunjukkan performa yang lebih stabil dan unggul dalam banyak kasus, khususnya pada zat-zat yang memiliki distribusi penggunaan yang lebih umum. Ini tercermin dari nilai R² yang konsisten lebih tinggi di hampir seluruh prediksi.

2. Random Forest Regressor seringkali memberikan performa lebih rendah, bahkan negatif pada beberapa target. Hal ini menandakan bahwa model ini:

   - Tidak mampu menggeneralisasi dengan baik,
   - Kemungkinan overfitting terhadap data latih,
   - Kurang optimal pada data dengan distribusi tidak seimbang.

3. Dari segi MAE dan MSE, Linear Regression juga relatif lebih baik untuk sebagian besar zat. Random Forest memang memiliki beberapa MAE yang sedikit lebih rendah, tapi diiringi dengan R² yang buruk, sehingga tidak cukup representatif sebagai model yang baik.

### **Pemilihan Model Terbaik (Model Selection)**

Berdasarkan perbandingan performa, Linear Regression dipilih sebagai model terbaik dalam analisis ini, dengan alasan berikut:

* R² tertinggi secara konsisten, khususnya pada target utama seperti Cannabis (0.4886), Legal Highs (0.3905), dan Magic Mushrooms (0.3086).
* Lebih stabil dan tidak menunjukkan R² negatif ekstrem seperti yang terjadi pada Random Forest.
* Interpretasi model lebih mudah, cocok dengan tujuan analisis yang ingin memahami hubungan antar variabel kepribadian/demografi terhadap frekuensi penggunaan zat.

## **Hyperparameter Tuning**
**Proses Improvement dengan Ridge Regression (Hyperparameter Tuning)**

Untuk mengatasi kelemahan Linear Regression, khususnya masalah overfitting dan multikolinearitas, dilakukan improvement dengan menggunakan Ridge Regression. Ridge Regression adalah varian Linear Regression yang menambahkan regularisasi L2 pada fungsi loss, yang membantu mengontrol kompleksitas model dengan menekan bobot koefisien agar tidak terlalu besar.

**Langkah-langkah improvement yang dilakukan:**

- Pencarian Nilai Optimal Hyperparameter (alpha)
  - Dilakukan tuning hyperparameter alpha, yang mengontrol kekuatan regularisasi.
  - Nilai alpha memengaruhi kompleksitas model:
    - Alpha kecil → model lebih fleksibel, risiko overfitting.
    - Alpha besar → model lebih sederhana, risiko underfitting.
  - Pencarian nilai optimal dilakukan dengan GridSearchCV, yang menguji kombinasi nilai alpha dari alpha_list = [0.01, 0.1, 1, 10, 100].

- Menggunakan Cross-Validation dalam Grid Search

  - GridSearchCV menggunakan cross-validation (cv=5) untuk memastikan model tidak hanya fit terhadap satu subset data.
  - Cross-validation mengurangi risiko overfitting saat memilih hyperparameter, karena mengevaluasi performa rata-rata dari beberapa fold data.

- Iterasi untuk Tiap Target Zat (Multi-Target Regression)

  - Model di-loop sebanyak 19 kali, masing-masing untuk satu target (zat psikoaktif berbeda).
  - Setiap iterasi melakukan tuning dan fitting model yang independen.
  - Hal ini memungkinkan model menangkap karakteristik unik dari masing-masing zat, karena efek prediktor bisa berbeda-beda.

**Tahapan Pemodelan dan Parameter**
1. `Ridge()`

  Inisialisasi model Ridge Regression:

  * `alpha`: Parameter regularisasi L2. Dicari secara otomatis melalui `GridSearchCV` dari daftar `[0.01, 0.1, 1, 10, 100]`.
  * `fit_intercept=True`: Model menghitung nilai intercept secara otomatis.
  * `solver='auto'` *(default)*: Scikit-learn akan memilih algoritma terbaik berdasarkan data.

2. `alpha_list = [0.01, 0.1, 1, 10, 100]`

  Daftar nilai alpha (hyperparameter) yang diuji:

  * `0.01`: Hampir tanpa regularisasi, mirip Linear Regression.
  * `0.1`–`1`: Regularisasi ringan–sedang.
  * `10`–`100`: Regularisasi kuat.

| Parameter       | Default  | Fungsi                                            |
| --------------- | -------- | ------------------------------------------------- |
| `alpha`         | `1.0`    | Kekuatan regularisasi L2                          |
| `fit_intercept` | `True`   | Menentukan apakah akan menghitung intercept       |
| `solver`        | `'auto'` | Algoritma optimisasi, default akan pilih otomatis |
| `max_iter`      | `None`   | Iterasi maksimum (jika solver iteratif)           |
| `tol`           | `1e-3`   | Toleransi konvergensi                             |


3. `GridSearchCV(estimator=ridge, param_grid={'alpha': alpha_list}, cv=5)`

  Melakukan pencarian alpha terbaik dengan Cross-Validation:

  * `param_grid`: Dictionary berisi daftar parameter (`alpha`) yang diuji.
  * `cv=5`: 5-fold cross-validation. Data pelatihan dibagi menjadi 5 bagian, model dilatih pada 4 bagian dan divalidasi pada 1 bagian secara bergiliran.
  * `scoring`: Default-nya menggunakan R² untuk regresi. Dapat diubah ke MSE atau MAE jika diinginkan.

| Parameter    | Fungsi                                                         |
| ------------ | -------------------------------------------------------------- |
| `param_grid` | Daftar nilai hyperparameter yang akan dicoba                   |
| `cv`         | Jumlah fold untuk cross-validation                             |
| `scoring`    | Metrik penilaian selama tuning (default: R² untuk regresi)     |
| `n_jobs`     | Jumlah core CPU yang digunakan (gunakan `-1` untuk semua core) |


---
**Tabel. Ringkasan Hasil Terbaik Ridge Regression untuk Prediksi Konsumsi Zat Psikoaktif**

| **Kategori**                | **Zat Psikoaktif** | **R² Score** | **Alpha Optimal** | **Keterangan**                                                           |
| --------------------------- | ------------------ | ------------ | ----------------- | ------------------------------------------------------------------------ |
| **R² Tertinggi**         | Cannabis           | 0.4905       | 10                | Paling tinggi; menunjukkan hubungan kuat antara kepribadian & konsumsi   |
| Cukup tinggi             | Legal Highs        | 0.3882       | 100               | Sangat dipengaruhi oleh Sensation Seeking dan Impulsiveness              |
|                             | Magic Mushrooms    | 0.3137       | 100               | Asosiasi kuat dengan openness dan sensation seeking                      |
|                             | LSD                | 0.3080       | 10                | Hubungan stabil dengan kepribadian, khususnya openness dan agreeableness |
|                             | Ecstasy            | 0.2510       | 100               | Kuat dengan sensation seeking dan agreeableness                          |
| R² Rendah (< 0.25)       | Amphetamines       | 0.2312       | 10                | Dipengaruhi impulsiveness, tapi variasinya masih cukup tinggi            |
|                             | Benzodiazepines    | 0.2038       | 100               | Cenderung dipengaruhi neuroticism                                        |
|                             | Methadone          | 0.2236       | 10                | Faktor neuroticism dominan                                               |
|                             | Cocaine            | 0.1927       | 100               | Sensation Seeking dominan tapi noise tinggi                              |
|                             | Nicotine           | 0.1637       | 100               | Ada korelasi impulsiveness & neuroticism, tapi tidak terlalu kuat        |
| R² Sangat Rendah (\~0.0) | Alcohol            | 0.0412       | 10                | Sangat umum, variabel kepribadian kurang menjelaskan variasi konsumsi    |
|                             | Caffeine           | 0.0175       | 100               | Hampir semua orang konsumsi → variasi terlalu kecil                      |
|                             | Chlorpromazine     | 0.0152       | 100               | Obat medis; tidak terpengaruh faktor psikologis                          |
|                             | Crack              | 0.0459       | 100               | Konsumsi sangat jarang, data imbalance tinggi                            |
|                             | Semer              | -0.0178      | 100               | Tidak informatif; mungkin noise atau tidak digunakan oleh peserta survei |

Hasil untuk hyperparameter tuning, sebagai berikut:
- Nilai R² tertinggi diperoleh pada prediksi Cannabis dengan R² sebesar 0.4905 dan alpha 10.
- Beberapa zat lain yang juga menunjukkan R² cukup tinggi adalah:

  - Magic Mushrooms (R² = 0.3137, alpha = 100)
  - LSD (R² = 0.3080, alpha = 10)
  - Legal Highs (R² = 0.3882, alpha = 100)
  - Ecstasy (R² = 0.2510, alpha = 100)

Namun, untuk sebagian besar zat lain, nilai R² tergolong rendah (di bawah 0.25), bahkan mendekati nol. Sebagai contoh:

- Alcohol (R² = 0.0412),
- Caffeine (R² = 0.0175),
- Chlorpromazine (R² = 0.0152),
- Semer (R² = -0.0178).

Catatan:
- Model Ridge dapat menurunkan overfitting dengan menambahkan penalti terhadap besar koefisien, sehingga dapat mengurangi varians walau dengan sedikit penurunan R².
- Efek dari variabel Openness, Sensation Seeking, dan Impulsiveness cenderung dominan dan konsisten berkontribusi positif terhadap prediksi penggunaan beberapa jenis zat.

### **Performa Model**:

Rendahnya nilai R² pada sebagian besar zat menunjukkan bahwa fitur-fitur kepribadian dan demografis tidak cukup kuat dalam menjelaskan variasi perilaku konsumsi zat tersebut, kemungkinan karena pengaruh kuat dari faktor eksternal lain seperti budaya, lingkungan sosial, atau ketersediaan zat.

Meskipun demikian, performa cukup tinggi pada zat seperti Cannabis, Magic Mushrooms, dan LSD menunjukkan bahwa konsumsi zat-zat tersebut memiliki keterkaitan yang lebih konsisten dengan karakteristik psikologis individu.

## Kesimpulan
Berdasarkan hasil analisis yang dilakukan, penelitian ini berusaha menjawab dua rumusan masalah utama, yaitu:
1. bagaimana hubungan antara tingkat impulsivitas dan sensation seeking dengan frekuensi konsumsi zat psikoaktif
2. bagaimana model prediktif dapat digunakan untuk memprediksi frekuensi konsumsi zat berdasarkan karakteristik psikologis (big five personality traits) dan demografis.

Hasil analisis menunjukkan bahwa terdapat kecenderungan positif antara impulsiveness dan sensation seeking dengan frekuensi konsumsi berbagai jenis zat psikoaktif. Artinya, semakin tinggi tingkat impulsivitas dan pencarian sensasi seseorang, semakin tinggi pula kecenderungan mereka untuk lebih sering mengonsumsi zat. Selain itu, beberapa dimensi dari Big Five Personality Traits, seperti Neuroticism (N) dan Openness (O), juga menunjukkan kontribusi yang berarti dalam model prediktif, tergantung pada jenis zat yang dianalisis. Misalnya, individu dengan skor tinggi pada openness atau neuroticism cenderung memiliki frekuensi konsumsi zat tertentu yang lebih tinggi.

Akan tetapi, ketika dibangun model prediksi menggunakan Linear Regression dan dibandingkan dengan Random Forest Regressor, diperoleh bahwa Linear Regression memberikan hasil yang lebih baik secara umum. Selanjutnya, model ini ditingkatkan performanya dengan menggunakan Ridge Regression dan dilakukan tuning terhadap hyperparameter alpha untuk mendapatkan hasil terbaik. Akan tetapi, nilai R² tertinggi yang diperoleh tetap hanya sekitar 0.49 (Cannabis), sementara sebagian besar zat lainnya memiliki R² yang rendah, bahkan mendekati nol.

Performa model yang cenderung rendah ini bukan semata-mata disebabkan oleh kelemahan model itu sendiri, melainkan juga mencerminkan bahwa fitur-fitur yang tersedia, yaitu skor kepribadian (Big Five, sensation seeking, impulsiveness) dan variabel demografis (usia, jenis kelamin, pendidikan, dan etnis) kemungkinan tidak cukup kuat untuk menjelaskan variasi perilaku penggunaan zat secara signifikan.

Beberapa hal yang dapat menjelaskan kondisi ini:
1. Karakteristik psikologis bersifat umum, bukan spesifik terhadap penggunaan zat

   Meskipun Big Five dan konstruk lain seperti impulsivitas memiliki hubungan dengan perilaku adiktif, namun hubungan tersebut bersifat umum dan tidak spesifik terhadap zat tertentu, sehingga daya prediksinya secara frekuensi cenderung lemah.

2. Faktor eksternal yang tidak terukur dalam data

   Banyak faktor penting lain yang berperan dalam konsumsi zat seperti tekanan lingkungan, pengaruh sosial, stres, trauma masa lalu, atau kondisi kesehatan mental yang tidak terekam dalam dataset ini, sehingga model tidak mampu menangkapnya.

3. Distribusi target yang tidak merata dan noise yang tinggi

   Beberapa zat memiliki distribusi yang sangat skewed (mayoritas responden tidak mengonsumsi), sehingga mempersulit model dalam mengenali pola yang bermakna, apalagi jika tidak cukup data dari pengguna aktif.

3. Hasil tuning Ridge Regression sudah optimal, namun tetap tidak signifikan peningkatannya, menandakan bahwa batasan utama bukan dari teknik modelling, melainkan dari kapasitas informasi yang bisa disediakan oleh fitur yang digunakan.

Dengan kata lain, meskipun model sudah dikembangkan secara sistematis dan dilakukan tuning dengan benar, informasi yang tersedia dalam fitur-fitur yang digunakan belum cukup kuat untuk membangun model prediktif yang presisi terhadap frekuensi konsumsi zat.

## Saran
Berdasarkan keterbatasan dan hasil penelitian ini, beberapa saran untuk penelitian selanjutnya adalah sebagai berikut:

1. Penambahan Variabel yang Lebih Komprehensif

   Untuk meningkatkan akurasi model prediksi, disarankan memasukkan variabel tambahan yang berkaitan dengan faktor lingkungan sosial, dukungan keluarga, kondisi psikologis yang lebih spesifik (seperti stres, trauma, gangguan mental), dan faktor perilaku lain yang dapat memengaruhi konsumsi zat psikoaktif.

2. Penggunaan Data Longitudinal

   Mengumpulkan data secara longitudinal dapat membantu menangkap perubahan perilaku konsumsi zat seiring waktu dan faktor-faktor yang memengaruhinya, sehingga model dapat mempelajari pola temporal yang lebih kompleks.

3. Eksplorasi Model Prediktif yang Lebih Kompleks

   Walaupun model linear memberikan hasil yang relatif baik, penggunaan metode lain seperti model ensemble, deep learning, atau model yang dapat menangkap interaksi non-linear antar fitur dapat dieksplorasi, terutama jika didukung dengan data yang lebih kaya.

4. Pendekatan Kualitatif

   Penelitian kualitatif untuk menggali alasan dan motivasi individu dalam mengonsumsi zat dapat memberikan wawasan yang lebih dalam untuk memperbaiki pemilihan variabel dan model prediktif.
   
## Daftar Pustaka

[Castaneda, J., Calvet, L., Benito, S., Tondar, A., & Juan, A. A. (2022). Data science, analytics and artificial intelligence in e-health: Trends, applications and challenges. Idescat.](https://www.idescat.cat/sort/sort471/47.1.1.Castaneda-etal.pdf)

[Fehrman, E., Egan, V., & Mirkes, E. (2015). Drug consumption (quantified) [Data set]. UCI Machine Learning Repository. https://archive.ics.uci.edu/ml/datasets/Drug+consumption+(quantified)](https://arxiv.org/abs/1506.06297)

[Fernández-Suárez, Á., Pérez, V., & Martínez, F. (2024). Substance addiction in adolescents: Influence of parenting and personality traits. Brain Sciences, 14(5), 449. https://doi.org/10.3390/brainsci14050449](https://www.mdpi.com/2076-3425/14/5/449)

[Personality in alcohol and opioid use disorder: A multiphasic approach. (2024). Annals of Neurosciences. https://doi.org/10.1177/09727531241274098](https://journals.sagepub.com/doi/10.1177/09727531241274098)

[Tan, C. N.-L., & Fauzi, M. A. (2023). The significance of Big Data Analytics in Healthcare Information Systems. International Journal of Data Science and Big Data Analytics, 3(1), 45–57.](https://www.svedbergopen.com/files/1698407561_3_IJDSBDA202313011740NZ_(p_45-57).pdf)

[Volkow, N. D., Koob, G. F., & McLellan, A. T. (2023). Substance use disorders: A comprehensive update of classification, epidemiology, neurobiology, and treatment. European Journal of Neuroscience. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10168177/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10168177/)
