# 🚀 Deploying AI-Powered Demand Forecasting with API, Stock Integration & Kafka

## 📌 Project Overview
This project implements **Transformer-based AI forecasting**, integrates **real-time stock inventory updates**, and deploys as a **cloud-based API using FastAPI**.

## 🛠️ Tech Stack
- **Deep Learning:** PyTorch (LSTM & Transformer-based forecasting)
- **Cloud Deployment:** FastAPI (REST API for Predictions)
- **Streaming Pipeline:** Apache Kafka (Real-time Sales Data)
- **Stock Inventory Integration:** API-based stock updates
- **Data Processing:** Pandas, NumPy

## 📌 Features Implemented
✅ Transformer-based Time Series Forecasting  
✅ Cloud API Deployment with FastAPI  
✅ Real-time Sales Streaming via Kafka  
✅ Stock Inventory Management Integration  

## 🚀 How to Run the Project
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/bruteai/deploying-demand-forecasting-api.git
cd deploying-demand-forecasting-api
```
### 2️⃣ Install Dependencies
```sh
pip install torch fastapi uvicorn requests kafka-python pandas numpy
```
### 3️⃣ Start the API Server
```sh
uvicorn api.cloud_api:app --host 0.0.0.0 --port 8000 --reload
```
### 4️⃣ Start Kafka Consumer
```sh
python kafka_streaming/kafka_consumer.py
```
### 5️⃣ Start Stock Inventory Updates
```sh
python stock_inventory/stock_inventory_manager.py
```

## 📂 Project Structure
```
📁 deploying-demand-forecasting-api
│── data_processing/         
│   ├── data_preprocessing.py  
│
│── models/                  
│   ├── lstm_model.py        
│   ├── transformer_model.py 
│
│── api/                    
│   ├── cloud_api.py         
│
│── stock_inventory/         
│   ├── stock_inventory_manager.py 
│
│── kafka_streaming/         
│   ├── kafka_consumer.py     
│
│── README.md                
```

## 🎯 Future Enhancements
- Implement **Multi-modal Forecasting (Time Series + External Factors)**  
- Deploy on **AWS/GCP as a scalable microservice**  
- Improve **Stock Management with AI-driven predictions**  

---

💡 **Contributions Welcome!** If you'd like to improve this project, feel free to open a pull request.

