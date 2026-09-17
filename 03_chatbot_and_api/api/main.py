from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Car Valuation API",
    version="1.0.0"
)

class CarValuationRequest(BaseModel):
    brand: str
    model_name: str
    manufacture_year: int
    odo_km: int
    gearbox: str
    province: str
    is_one_owner: int = 0

class CarValuationResponse(BaseModel):
    predicted_price_vnd: float
    price_range_low: float
    price_range_high: float
    deal_status: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/webhook/watson")
def watson_webhook(payload: dict):
    raise NotImplementedError

@app.post("/api/v1/valuation", response_model=CarValuationResponse)
def predict_price(request: CarValuationRequest):
    raise NotImplementedError
