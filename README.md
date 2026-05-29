# Multi-Agent Fraud Detection

A modular fraud detection system using FastAPI, Docker, vector database retrieval, and multi-agent architecture.
```text
                    ┌────────────────┐
                    │ Transaction    │
                    └──────┬─────────┘
                           ↓
         ┌────────────────────────────────┐
         │ Customer Context Agent        │
         └────────────────────────────────┘
                           ↓
         ┌────────────────────────────────┐
         │ Fraud Scoring Agent           │
         └────────────────────────────────┘
                           ↓
         ┌────────────────────────────────┐
         │ RAG Retrieval Agent           │
         └────────────────────────────────┘
                           ↓
         ┌────────────────────────────────┐
         │ Reasoning Agent               │
         │ - analyze all signals         │
         │ - determine fraud pattern     │
         │ - determine escalation        │
         │ - generate investigation      │
         └────────────────────────────────┘
```

---
## Customer Context Agent

The Customer Context Agent collects and analyzes customer-related information before fraud analysis.

Responsibilities:
- retrieve customer transaction history
- identify customer behavior patterns
- analyze device usage and login activity
- check customer risk profile
- provide contextual information for downstream agents


---

## Fraud Scoring Agent

The Fraud Scoring Agent predicts the probability of fraud using machine learning models.

Responsibilities:
- generate fraud probability score
- evaluate suspicious transaction patterns
- detect anomalies
- classify transaction as low, medium, or high risk

---

## RAG Retrieval Agent

The RAG (Retrieval-Augmented Generation) Agent retrieves relevant historical fraud cases and investigation knowledge from the vector database.

Responsibilities:
- perform semantic search
- retrieve similar fraud cases
- retrieve investigation documents
- provide supporting evidence for reasoning agent


Data Source:
- vector database
- ---

# Tech Stack

- Python 3.10
- FastAPI
- Docker
- ChromaDB / Vector Database
- OpenAI API
- Machine Learning (Scikit-learn)

---

# Project Structure

```text
project-root/
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── agents/
│   │   ├── customer_context_agent.py
│   │   ├── fraud_scoring_agent.py
│   │   ├── ingests.py
│   │   ├── orchestrator.py
│   │   ├── rag_agent.py
│   │   └── reasoning_agent.py
│   │
│   │
│   ├── vector_db/
│   │   └── collections/
│   │
│   ├── ml/
│   │   ├── logreg_model.py
│   │ 
├── notebook/
│
├── models/
│   └── fraud_logistic_regression.pkl
│
├── data/
│   ├── 01_SOP_fraud_thresholds.txt
│   ├── 02_escalation_matrix.txt
│   ├── 03_fraud_patterns_playbook.txt
│   ├── customer_history.json
│   └── transactions.csv
│
├── docker-compose.yml
├── dockerfile
├── requirements.txt
├── .env
├── .gitignore
├── TEORI.md
└── README.md
```

---

# FastAPI Setup

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run FastAPI Locally

```bash
uvicorn src.api.main:app --reload
```

---

## API URL

```text
http://127.0.0.1:8000
```

---

## Swagger Documentation

```text
http://127.0.0.1:8000/docs
```

---

# Example FastAPI Main File

`src/api/main.py`

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}
```

---

# Docker Setup

## Build Docker Image

```bash
docker build -t fraud-api .
```

---

## Run Docker Container

```bash
docker run -p 8000:8000 fraud-api
```

---

# Docker Compose

## Run All Services

```bash
docker-compose up --build
```

---

# Collection Workflow

## Step 1 — Ingest Data

Agent reads txt file investigation reports.

```text
ingest.py
```
collection will save in vector_db folder

---

# Retrieval Flow

```text
refine query from customer context
    ↓
Embedding Query
    ↓
Vector Search
    ↓
Relevant Chunks Retrieved
    ↓
LLM Analysis
    ↓
Fraud Insight Response
```


---

# Environment Variables

`.env`

```env
OPENAI_API_KEY=your_api_key
```
- RAG pipeline
- LangGraph integration
- ML model serving
- PostgreSQL integration

---
# Example Output
```
{
    "transaction": {
        "transaction_id": "TXN-90001",
        "customer_id": "CUST-00001",
        "amount_idr": 7000000,
        "merchant_category": "CRYPTO",
        "city": "Bangkok",
        "is_foreign": true,
        "device_id": "DEV-NEW-999",
        "hour": 3,
        "day_of_week": 6
    },
    "customer_context": {
        "customer_found": true,
        "customer_id": "CUST-00001",
        "name_masked": "B**i S*****o",
        "risk_profile": "LOW",
        "kyc_status": "VERIFIED",
        "account_age_months": 84,
        "home_city": "Jakarta",
        "typical_transaction_amount_idr": {
            "min": 50000,
            "median": 350000,
            "max": 2500000,
            "p95": 1800000
        },
        "typical_transaction_locations": [
            "Jakarta",
            "Bekasi",
            "Tangerang"
        ],
        "typical_merchant_categories": [
            "GROCERY",
            "RESTAURANT",
            "FUEL",
            "ECOMMERCE_LOCAL"
        ],
        "active_devices": 2,
        "previous_fraud_reports": 0,
        "device_fingerprints": [
            "DEV-A7F3-Jakarta-iOS",
            "DEV-B921-Jakarta-Web"
        ],
        "recent_transactions": [
            {
                "date": "2026-05-18",
                "amount_idr": 450000,
                "merchant": "INDOMARET TEBET",
                "category": "GROCERY",
                "city": "Jakarta",
                "status": "APPROVED"
            },
            {
                "date": "2026-05-17",
                "amount_idr": 125000,
                "merchant": "GOFOOD MERCHANT",
                "category": "RESTAURANT",
                "city": "Jakarta",
                "status": "APPROVED"
            },
            {
                "date": "2026-05-16",
                "amount_idr": 875000,
                "merchant": "TOKOPEDIA",
                "category": "ECOMMERCE_LOCAL",
                "city": "Jakarta",
                "status": "APPROVED"
            },
            {
                "date": "2026-05-15",
                "amount_idr": 280000,
                "merchant": "SHELL KUNINGAN",
                "category": "FUEL",
                "city": "Jakarta",
                "status": "APPROVED"
            },
            {
                "date": "2026-05-14",
                "amount_idr": 1250000,
                "merchant": "ELECTRONIC CITY",
                "category": "ELECTRONICS",
                "city": "Jakarta",
                "status": "APPROVED"
            }
        ]

    },
    "retrieved_context": "# SOP Fraud Detection — Threshold Definitions\n\n**Document ID:** SOP-FD-001\n**Version:** 2.4\n**Last Updated:** 2026-04-15\n**Owner:** Fraud Risk Management Division — Allo Bank\n**Classification:** Internal Use Only\n\n## 4. Threshold Geografis\n\n### 4.1 Lokasi vs Home City\n\n- Transaksi di **home city atau typical_transaction_locations** → tidak ada flag geografis\n- Transaksi di **kota lain dalam negeri** → +10 poin risiko\n- Transaksi di **luar negeri** → +30 poin risiko\n- Transaksi di **luar negeri dengan akun yang belum pernah transaksi luar negeri** → +50 poin risiko\n\n### 4.2 Velocity (Kecepatan Perpindahan Lokasi)\n\n- 2+ transaksi di kota berbeda dalam **< 1 jam** → +60 poin risiko (kecuali kedua kota berdekatan dan masuk typical_transaction_locations)\n- Transaksi di 2 negara berbeda dalam **< 4 jam** → +80 poin risiko + auto FLAGGED\n\n## 5. Threshold Device & Behavioral\n\n- Transaksi dari **device baru** (tidak ada di device_fingerprints) → +20 poin risiko\n- Transaksi dari **device baru DAN lokasi baru DAN nominal besar** (kombinasi 3 anomali) → +60 poin risiko + auto FLAGGED\n- Transaksi di luar **jam wajar nasabah** (03:00-05:00 lokal, jika bukan pola normal) → +15 poin risiko\n\n# Fraud Investigation Escalation Matrix\n**Document ID:** SOP-FD-002\n**Version:** 1.8\n**Last Updated:** 2026-03-22\n**Owner:** Fraud Risk Management Division — Allo Bank\n**Classification:** Internal Use Only\n\n## 3. Routing Rules Berdasarkan Kasus\n### 3.1 Transaksi Luar Negeri\n\n- Skor 40-69 + lokasi LN → **Tier 2** (Junior tidak handle LN)\n- Skor 70+ + lokasi LN → **Tier 3** langsung\n- Multiple country dalam < 24 jam → **Tier 3** + cross-border fraud check\n\n### 3.2 Crypto & High-Risk Categories\n\n- Single crypto transaction < Rp 5jt → **Tier 1**\n- Crypto/gaming > Rp 5jt → **Tier 2**\n- Repeated crypto pattern (3+ dalam minggu) → **Tier 3** + AML team notification\n\n### 3.3 Customer dengan Riwayat Fraud\n\n- previous_fraud_reports = 1 → automatis +1 tier (mis. Tier 1 → Tier 2)\n- previous_fraud_reports ≥ 2 → langsung **Tier 3** untuk transaksi apapun yang FLAGGED\n- Customer yang sudah ada di internal blacklist → **Tier 3** + freeze account\n\n### 3.4 Pattern Anomalies\n\n- Device baru + lokasi baru + nominal besar → minimum **Tier 2**\n- Velocity attack (5+ transaksi flagged dalam 1 jam) → **Tier 3** + auto-freeze card\n- Identity theft suspicion (KYC mismatch) → **Tier 3** + Compliance team\n\n# Fraud Pattern Recognition Playbook\n**Document ID:** SOP-FD-003\n**Version:** 3.1\n**Last Updated:** 2026-05-02\n**Owner:** Fraud Intelligence Team — Allo Bank\n**Classification:** Internal Use Only\n\n## Tujuan\nberisi pattern fraud yang paling sering terjadi di Allo Bank dan industri perbankan Indonesia secara umum. AI agents dan fraud analyst menggunakan playbook ini untuk identifikasi cepat tipe fraud dan menentukan tindakan.\nSetiap pattern punya: ciri-ciri, contoh skenario, dan recommended action.\n\n## 5. Pattern D — Mule Account / Money Laundering\n\n### Ciri-ciri\n- Akun (sering akun baru atau dormant) digunakan untuk receive dana fraud lalu dilanjut transfer keluar\n- Pattern in-and-out cepat, balance tidak pernah idle lama\n- Sering melibatkan crypto exchange atau merchant kategori suspicious\n\n### Red Flags\n- Akun baru (account_age_months < 3) tiba-tiba ada incoming besar\n- Diikuti outgoing dalam < 1 jam ke rekening lain / crypto exchange\n- Multiple incoming dari berbagai source ke akun yang sama\n- Customer profile tidak match dengan volume transaksi (mis. mahasiswa terima Rp 200jt)\n\n### Skenario Tipikal\nAkun mahasiswa berumur 2 bulan tiba-tiba terima Rp 50jt dari 5 source berbeda dalam 2 jam, lalu langsung transfer semua ke crypto exchange.\n\n### Recommended Action\n- Auto-freeze account\n- Tier 3 investigation + AML team notification\n- Wajib lapor ke PPATK (Suspicious Transaction Report)\n- Coordinate dengan bank source untuk recall jika fraud terkonfirmasi\n- Close account dan blacklist customer\n\n# Fraud Pattern Recognition Playbook\n**Document ID:** SOP-FD-003\n**Version:** 3.1\n**Last Updated:** 2026-05-02\n**Owner:** Fraud Intelligence Team — Allo Bank\n**Classification:** Internal Use Only\n\n## Tujuan\nberisi pattern fraud yang paling sering terjadi di Allo Bank dan industri perbankan Indonesia secara umum. AI agents dan fraud analyst menggunakan playbook ini untuk identifikasi cepat tipe fraud dan menentukan tindakan.\nSetiap pattern punya: ciri-ciri, contoh skenario, dan recommended action.\n\n## 2. Pattern A — Card Not Present (CNP) Fraud\n\n### Ciri-ciri\n- Transaksi online (e-commerce, subscription) di mana kartu fisik tidak dipakai\n- Sering terjadi dalam waktu singkat setelah data kartu bocor\n- Biasanya nominal kecil-menengah untuk testing, lalu eskalasi ke nominal besar\n\n### Red Flags\n- Multiple transaksi kecil dalam 10-30 menit (testing card validity)\n- Diikuti transaksi besar dalam jam yang sama\n- Merchant: ecommerce internasional, digital goods, gift cards\n- Billing address tidak match dengan home_city nasabah\n\n### Skenario Tipikal\nNasabah di Jakarta tiba-tiba ada 3 transaksi $5-$10 di merchant US dalam 15 menit, diikuti 1 transaksi $500. Card data kemungkinan bocor dari merchant breach.\n\n### Recommended Action\n- Block kartu sementara\n- Issue kartu baru\n- Investigate merchant breach (cek apakah customer lain dengan kartu di merchant yang sama juga kena)\n- Refund nasabah dalam 5 hari kerja\n\n# Fraud Pattern Recognition Playbook\n**Document ID:** SOP-FD-003\n**Version:** 3.1\n**Last Updated:** 2026-05-02\n**Owner:** Fraud Intelligence Team — Allo Bank\n**Classification:** Internal Use Only\n\n## Tujuan\nberisi pattern fraud yang paling sering terjadi di Allo Bank dan industri perbankan Indonesia secara umum. AI agents dan fraud analyst menggunakan playbook ini untuk identifikasi cepat tipe fraud dan menentukan tindakan.\nSetiap pattern punya: ciri-ciri, contoh skenario, dan recommended action.\n\n## 4. Pattern C — Authorized Push Payment (APP) Fraud / Social Engineering\n\n### Ciri-ciri\n- Nasabah ditipu untuk mentransfer uang sendiri ke rekening fraudster\n- Modus: penipuan investasi, romance scam, fake customer service, impersonation\n- Tantangan: secara teknis nasabah authorize sendiri, jadi sistem fraud sulit detect\n\n### Red Flags\n- Transfer ke beneficiary baru dengan nominal besar\n- Pattern transfer berulang ke rekening yang sama dalam waktu pendek\n- Nasabah tampak bingung saat dihubungi (pancingan dari fraudster yang sedang on-call)\n- Beneficiary account baru dibuka < 30 hari\n- Memo transfer ambiguous (\"untuk investasi\", \"membayar pajak\", dll)\n\n### Skenario Tipikal\nIbu rumah tangga di-call orang yang ngaku petugas pajak, ditakut-takuti tagihan pajak Rp 15jt yang harus dibayar segera ke rekening \"kantor pajak\" yang ternyata fraudster.\n\n### Recommended Action\n- Hold transaksi jika real-time detection sempat catch\n- Cooling period 24 jam untuk transfer pertama ke beneficiary baru > Rp 10jt\n- Edukasi via SMS warning sebelum confirm\n- Bila sudah ter-transfer: bantu nasabah lapor polisi, recall request ke bank tujuan\n- Note: recovery rate untuk APP fraud rendah (< 30%), prioritas adalah prevention",
    "investigation_result": "```json\n{\n    \"risk_level\": \"HIGH\",\n    \"fraud_probability\": 1.0,\n    \"likely_fraud_pattern\": \"Mule Account / Money Laundering\",\n    \"recommended_action\": \"Auto-freeze account and initiate Tier 3 investigation with AML team notification.\",\n    \"recommended_escalation_tier\": \"Tier 3\",\n    \"supporting_signals\": [\n        \"Transaction amount of IDR 7,000,000 exceeds typical transaction limits.\",\n        \"Transaction made from a new device not previously associated with the customer.\",\n        \"Transaction occurred in a foreign country (Bangkok) while the customer is based in Jakarta.\",\n        \"Transaction made at an unusual hour (3 AM) which is outside the customer's typical transaction times.\"\n    ],\n    \"reasoning\": \"The transaction shows multiple red flags: it is a large amount compared to the customer's typical transactions, made from a new device, in a foreign location, and at an unusual hour. These factors combined indicate a high risk of fraud, particularly consistent with patterns of money laundering where large sums are transferred to crypto exchanges from accounts that have not previously engaged in such activities.\"\n}\n```"
}```
