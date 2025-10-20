from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import os
from dotenv import load_dotenv

from agents.web_researcher import WebResearcherAgent

# Load environment variables
load_dotenv()

app = FastAPI(title="Multi-Agent Web Research System", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchQuery(BaseModel):
    query: str

class ResearchResponse(BaseModel):
    query: str
    result: Dict[str, Any]
    status: str

# Initialize the Web Researcher Agent
web_researcher = WebResearcherAgent()

@app.get("/")
async def root():
    return {"message": "Multi-Agent Web Research System API"}

@app.post("/api/research", response_model=ResearchResponse)
async def research_query(request: ResearchQuery):
    """
    Main research endpoint that processes user queries through the multi-agent system
    """
    try:
        # Process the query through the Web Researcher Agent
        result = await web_researcher.research(request.query)
        
        return ResearchResponse(
            query=request.query,
            result=result,
            status="success"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "multi-agent-research-api"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)