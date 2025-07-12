import os
import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import datetime





# --- Database Setup ---
DATABASE_URL = "sqlite:///./llm_costs.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Log(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    model = Column(String)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)

Base.metadata.create_all(bind=engine) # creates logs table in the database


# --- Configuration ---
OPENAI_API_BASE_URL = "https://api.openai.com"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")





# --- FastAPI Application ---
app = FastAPI()

@app.on_event("startup")
def startup_event():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    print("Server started. Proxying requests to OpenAI API.")

@app.post("/{full_path:path}")
async def proxy_request(request: Request, full_path: str):
    print(f"[LOG] Intercepted request for path: /{full_path}")
    
    try:
        body = await request.json()
    except Exception:
        # This handles cases where the curl command sends invalid JSON
        print("[ERROR] Could not parse incoming JSON body from client.")
        return JSONResponse(content={"detail": "Invalid JSON in request body."}, status_code=400)

    target_url = f"{OPENAI_API_BASE_URL}/{full_path}"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    
    # Make the request to OpenAI
    response = requests.post(target_url, headers=headers, json=body)

    # Check if the response from OpenAI is a success
    if response.ok:
        response_data = response.json()

        db = SessionLocal()
        try:
            usage = response_data.get("usage", {})
            new_log = Log(
                model=response_data.get("model"),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens")
            )
            db.add(new_log)
            db.commit()
            print("[LOG] Successfully saved request details to the database.")
        except Exception as e:
            print(f"[ERROR] Failed to log to database: {e}")
            db.rollback()
        finally:
            db.close()
        return JSONResponse(content=response.json(), status_code=response.status_code)
    else:
        # If it's an error, print the detailed reason for debugging
        print(f"[ERROR] OpenAI returned a non-successful status: {response.status_code}")
        print(f"[ERROR] OpenAI's Response Text: {response.text}")
        
        # Return the exact error from OpenAI back to the curl command
        return JSONResponse(content={"detail_from_openai": response.text}, status_code=response.status_code)
