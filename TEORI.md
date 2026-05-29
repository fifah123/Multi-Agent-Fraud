# 1. Fraud Detection Model

## 3 Hal yang kurang tepat

### a. Langsung deploy hanya karena metric naik
PR-AUC test naik dari 0.68 → 0.74 setelah SMOTE memang improvement, tetapi belum cukup untuk langsung deploy.

Masalah:
- Belum ada analisis business impact
- Belum tahu precision/recall di threshold tertentu
- Fraud detection biasanya highly imbalanced → metric global saja tidak cukup

Yang saya lakukan:
- Evaluate recall pada fraud high-risk
- Hitung expected fraud loss saved vs false positive cost

---

### b. Threshold 0.5 dipakai default
Threshold 0.5 hampir tidak pernah optimal untuk fraud detection.

Masalah:
- Fraud detection cost-sensitive
- False negative jauh lebih mahal dibanding false positive
- Threshold harus mengikuti business objective

Yang saya lakukan:
- Tune threshold berdasarkan:
  - recall target
  - investigation capacity
  - expected monetary loss
- Gunakan PR curve untuk memilih operating point optimal

---

### c. Oversampling bisa menyebabkan leakage / unrealistic distribution
SMOTE bisa membuat synthetic samples yang tidak realistis pada fraud patterns.

Masalah:
- Fraud data bersifat temporal & evolving
- Jika split salah → leakage antar synthetic samples
- Bisa overfit ke synthetic fraud pattern

Yang saya lakukan:
- Split train/test secara temporal
- Apply SMOTE hanya di training fold
- Compare dengan:
  - class_weight
  - focal loss
  - balanced random forest
  - XGBoost scale_pos_weight

---

## Tambahan yang Akan Saya Lakukan

- Calibrate probability
- Monitor drift
- Analyze feature importance

---

# 2. Kenapa RAG di Banking Lebih Sulit

## a. Regulasi & Compliance

Masalah:
- Jawaban tidak boleh melanggar regulasi OJK
- Dokumen policy sering berubah
- Harus ada auditability

Mitigasi:
- Versioned document indexing
- Metadata filtering:
  - effective_date
  - document_version
  - department
- Citation mandatory
- Human approval untuk critical response

---

## b. Hallucination Sangat Berbahaya

Masalah:
- Jawaban salah bisa menyebabkan:
  - financial loss
  - compliance violation
  - customer misinformation

Mitigasi:
- Retrieval-first prompting
- Strict grounded generation
- Low temperature
- Confidence scoring
- “I don't know” or "I am sorry" fallback
- kategorikan user question kedalam tags dalam knowledge untuk improve retrieval precision

---

## c. Sensitive Data / PII

Masalah:
- KYC document mengandung:
  - NIK
  - alamat
  - income
  - biometric info

Mitigasi:
- PII masking sebelum indexing
- Role-based access control
- Private deployment (VPC/on-prem)
- Encryption at rest & in transit
- Audit logging semua query user

---

# 3. Credit Scoring Production Diagnosis

## Kondisi

- AUC tetap 0.85
- Default rate naik 30%

Artinya:
Ranking model masih bagus, tetapi business outcome berubah.

---

## Hipotesis Awal

### a. Population drift
Karakter customer berubah.

Contoh:
- ekonomi memburuk
- customer baru lebih risky

---

### b. Policy drift
Approval policy berubah.

Contoh:
- threshold approval diturunkan
- acquisition agresif

---

### c. Label delay / macroeconomic shift
Default dipengaruhi faktor eksternal.

Contoh:
- inflasi
- PHK
- kenaikan suku bunga

---

## Urutan Pengecekan

### 1. Data quality check
- missing value
- schema drift
- feature distribution

---

### 2. Population Stability Index (PSI)
Cek drift feature utama:
- income
- utilization
- delinquency

---

### 3. Segment analysis
Apakah default naik di:
- region tertentu
- acquisition channel tertentu
- customer segment tertentu

---

### 4. Calibration check
Mungkin ranking masih bagus tetapi probability prediction tidak calibrated.

Cek:
- calibration curve
- Brier score

---

### 5. Business policy changes
Cek:
- approval threshold
- underwriting rules
- campaign acquisition

---

## Kapan Retrain?

Retrain jika:
- feature drift moderat
- relationship masih relevan
- architecture masih valid

Contoh:
- macro shift ringan
- seasonal change

---

## Kapan Rebuild dari Awal?

Rebuild jika:
- major concept drift
- new fraud/customer behavior
- feature set obsolete
- new data sources tersedia

Contoh:
- sekarang ada transaction graph data
- digital behavior data
- open banking data

---

# 4. Model A vs Model B

## Saya pilih: Model A (XGBoost)

Alasan utama:
- regulator friendliness
- explainability
- operational practicality

---

## Kenapa bukan DNN?

Walaupun AUC naik:
0.88 → 0.91

Tetapi:
- latency 4x lebih lambat
- black-box
- sulit explain adverse action

---

## Dalam konteks OJK

Bank harus bisa menjelaskan:
- kenapa aplikasi ditolak
- faktor utama keputusan

SHAP pada XGBoost jauh lebih acceptable.

Contoh adverse action:
- utilization terlalu tinggi
- income instability
- delinquency history

DNN sulit memberi explanation yang stabil & regulator-friendly.

---

## Trade-off yang Saya Pertimbangkan

### Model A
Pros:
- explainable
- latency rendah
- mudah audit
- stabil di tabular data

Cons:
- sedikit kalah akurasi

---

### Model B
Pros:
- higher predictive power

Cons:
- black-box
- difficult governance
- harder monitoring
- costly inference

---

## Kesimpulan

Dalam banking:
“slightly lower AUC but explainable”
sering lebih valuable dibanding
“higher AUC but opaque”.

---

# 5. Design D.A.R.A (Data Assistant for Rapid Awareness)

## Target

- 100 concurrent users
- metadata QA
- business glossary QA
- cost-efficient
- high accuracy

---

# Recommended Architecture

## LLM

### Recommendation:
- GPT-4o-mini untuk production

---

## Kenapa GPT-4o-mini?

Pros:
- strong reasoning
- fast latency
- cheap enough
- excellent tool calling



## Embedding Model

### Recommendation:
- text-embedding-3-large
---

## Kenapa?

- Semantic accuracy sangat tinggi
- Better multilingual understanding
- Strong performance for metadata search

---


# Vector DB

## Saya pilih: weaviate


Karena Kadang user query exact keyword seperti nama table, nama schema, nama kolom yang mana Weaviate mendukung vector similarity dan hybrid weighted retrieval yang manani sangat cocok untuk metadata assistant.





---


# Dify Flow Recommendation

## Flow

1. User Question
2. LLM Query Refiner Node
3. Knowledge Retrieval Node with Hybrid Retrieval (Menggunakan refined query dan category filtering akan meretrieve top 5 chunks metadata knowledge)
4. HTTP Request Node (python service yang akan emngambil DD meliputi jumlah kolom, nama kolom, dan schema detail)
5. LLM Answer with context retrieval and metadata information (LLM memberikan: penjelasan table, fungsi bisnis, jumlah kolom, daftar kolom penting, related table recommendation)


---

# Chunking Strategy

Saya tidak akan chunk terlalu kecil.

1 metadata entity = 1 chunk

Isi:
- table description
- owner
- columns
- glossary
- lineage
- tags

---