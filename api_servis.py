from fastapi import FastAPI
import joblib
import pandas as pd
import uvicorn

app = FastAPI(title="Fintech Tahsilat AI Agent API")

# 1. Paketlediğin dosyaları yükle
model = joblib.load("tahmin_modeli.pkl")
reg_model = joblib.load("gecikme_modeli.pkl")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

@app.post("/tahmin")
async def predict(data: dict):
    # Buğra'dan (Backend) gelen JSON verisini DataFrame'e çeviriyoruz
    input_df = pd.DataFrame([data])
    
    # Sayısal sütunları senin eğittiğin scaler ile ölçeklendiriyoruz
    numeric_cols = ['income', 'invoice_amount', 'dpd']
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    
    # Tahminleri yapıyoruz
    risk_skoru = model.predict_proba(input_df.drop(columns=['dpd'], errors='ignore'))[:, 0][0] * 100
    gecikme_tahmini = reg_model.predict(input_df.drop(columns=['dpd'], errors='ignore'))[0]
    
    # Karar Mekanizması (V4'teki mantığın aynısı)
    def get_action(score):
        if score >= 85: return "CRITICAL: Acil Kıdemli Tahsilatçı Araması"
        elif score >= 60: return "WARNING: AI Sesli Ajan Araması"
        else: return "NORMAL: Rutin İzleme"

    return {
        "risk_puanı": round(float(risk_skoru), 2),
        "tahmini_gecikme": round(float(gecikme_tahmini), 1),
        "onerilen_aksiyon": get_action(risk_skoru)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)