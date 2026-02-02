from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = joblib.load("model.pkl")


@app.get("/", response_class=HTMLResponse)
def form(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None}
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    Age: int = Form(...),
    Income: float = Form(...),
    LoanAmount: float = Form(...),
    NumCreditLines: int = Form(...),
    InterestRate: float = Form(...)
):
    df = pd.DataFrame({
    "Age": [Age],
    "Income": [Income],
    "LoanAmount": [LoanAmount],
    "NumCreditLines": [NumCreditLines],
    "InterestRate": [InterestRate],
    })


    prediction = model.predict(df)[0]

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": int(prediction)
        }
    )
