from fastapi import FastAPI
from graph import SupportRequest, SupportResponse, ask

app = FastAPI(title="Zepto Support Assistant")


@app.get("/")
def root():
    return {"service": "Zepto Support Assistant", "status": "ok"}


@app.post("/ask", response_model=SupportResponse)
def ask_endpoint(request: SupportRequest):
    return ask(request.query)
