import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from handler import EndpointHandler
import time
import asyncio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Eventsnap Inference API", description="Local testing server for HF Endpoint Handler")

# Initialize the Hugging Face custom handler
handler = EndpointHandler(path=".")

@app.post("/")
async def predict(request: Request):
    """
    Acts exactly like the Hugging Face Inference Endpoint.
    Expects a JSON payload with `inputs` and `parameters` keys.
    """
    try:
        tt = time.perf_counter()
        data = await request.json()
        logger.info(f"recieved request {len(data.get('inputs', data))} images")
        if not data:
            return JSONResponse(content={"error": "Empty JSON payload"}, status_code=400)
        
        result = await asyncio.to_thread(handler, data)
        
        # If the handler returned an error key, it failed gracefully
        if isinstance(result, dict) and "error" in result:
            return JSONResponse(content=result, status_code=400)
        tt = time.perf_counter() - tt
        logger.info(f"Time taken: {tt}")
        return result

    except Exception as e:
        logger.error(f"Server error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    port = 5000
    print("\n" + "="*50)
    print("🚀 Local Eventsnap Inference Server (FastAPI) is Running!")
    print(f"Send HTTP POST requests to: http://0.0.0.0:{port}/")
    print(f"Auto-generated interactive docs available at: http://0.0.0.0:{port}/docs")
    print("="*50 + "\n")
    uvicorn.run(app='app:app', host="0.0.0.0", port=port)
