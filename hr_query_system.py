"""
HR Resource Query System - Minimal but Powerful
Single-file FastAPI application with embedded vector search
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
from datetime import datetime
import time

# Initialize FastAPI
app = FastAPI(
    title="HR Resource Query System",
    description="Lightweight but powerful HR assistant",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model (lightweight, runs on CPU)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class Employee(BaseModel):
    id: str
    name: str
    title: str
    experience_years: int
    skills: List[str]
    domains: List[str]
    availability: str
    summary: str

# In-memory data store (no database needed)
EMPLOYEES_DATA = [
    {
        "id": "1",
        "name": "Sarah Chen",
        "title": "Senior ML Engineer",
        "experience_years": 8,
        "skills": ["Python", "TensorFlow", "PyTorch", "Docker", "AWS"],
        "domains": ["Healthcare", "AI", "Computer Vision"],
        "availability": "immediate",
        "summary": "8 years ML experience. Led medical imaging AI reducing diagnosis time 60%. Expert in TensorFlow, PyTorch, healthcare AI."
    },
    {
        "id": "2",
        "name": "Michael Rodriguez",
        "title": "Full Stack Developer",
        "experience_years": 6,
        "skills": ["React", "Node.js", "Python", "PostgreSQL", "Docker"],
        "domains": ["E-commerce", "Finance", "Web"],
        "availability": "2 weeks",
        "summary": "6 years full-stack. Built e-commerce platform handling 1M users daily. Expert in React, Node.js, scalable systems."
    },
    {
        "id": "3",
        "name": "Emily Watson",
        "title": "DevOps Engineer",
        "experience_years": 7,
        "skills": ["Kubernetes", "Docker", "Terraform", "AWS", "Python"],
        "domains": ["Cloud", "Infrastructure", "Enterprise"],
        "availability": "1 month",
        "summary": "7 years DevOps. Managed multi-cloud migration saving 40% costs. Expert in Kubernetes, Terraform, AWS."
    },
    {
        "id": "4",
        "name": "James Liu",
        "title": "Data Scientist",
        "experience_years": 5,
        "skills": ["Python", "R", "SQL", "Spark", "Machine Learning"],
        "domains": ["Finance", "Analytics", "Fraud Detection"],
        "availability": "immediate",
        "summary": "5 years data science. Built fraud detection system with 95% accuracy. Expert in Python, ML, financial analytics."
    },
    {
        "id": "5",
        "name": "Priya Sharma",
        "title": "Backend Engineer",
        "experience_years": 9,
        "skills": ["Java", "Spring Boot", "Microservices", "Kafka", "PostgreSQL"],
        "domains": ["Fintech", "Healthcare", "Distributed Systems"],
        "availability": "3 weeks",
        "summary": "9 years backend. Built payment platform processing $500M. Expert in Java, microservices, distributed systems."
    },
    {
        "id": "6",
        "name": "Alex Thompson",
        "title": "Frontend Developer",
        "experience_years": 4,
        "skills": ["React", "TypeScript", "Next.js", "CSS", "GraphQL"],
        "domains": ["EdTech", "Healthcare", "UI/UX"],
        "availability": "immediate",
        "summary": "4 years frontend. Built educational platform for 100K students. Expert in React, TypeScript, responsive design."
    },
    {
        "id": "7",
        "name": "Diana Martinez",
        "title": "Cloud Architect",
        "experience_years": 10,
        "skills": ["AWS", "Azure", "Terraform", "Security", "Python"],
        "domains": ["Enterprise", "Healthcare", "Security"],
        "availability": "1 month",
        "summary": "10 years cloud architecture. Led enterprise migration saving $5M. Expert in AWS, security, HIPAA compliance."
    },
    {
        "id": "8",
        "name": "Robert Kim",
        "title": "Mobile Developer",
        "experience_years": 6,
        "skills": ["React Native", "iOS", "Android", "TypeScript", "Firebase"],
        "domains": ["Healthcare", "Mobile", "Consumer Apps"],
        "availability": "2 weeks",
        "summary": "6 years mobile. Built healthcare app with 500K downloads. Expert in React Native, cross-platform development."
    },
    {
        "id": "9",
        "name": "Nina Patel",
        "title": "Data Engineer",
        "experience_years": 7,
        "skills": ["Python", "Spark", "Airflow", "SQL", "Kafka"],
        "domains": ["Healthcare", "Finance", "Big Data"],
        "availability": "2 weeks",
        "summary": "7 years data engineering. Built real-time pipeline processing 10M events/day. Expert in Spark, Kafka, Airflow."
    },
    {
        "id": "10",
        "name": "Kevin Zhang",
        "title": "Security Engineer",
        "experience_years": 7,
        "skills": ["Security", "Python", "AWS", "Penetration Testing", "Compliance"],
        "domains": ["Cybersecurity", "Healthcare", "Finance"],
        "availability": "3 weeks",
        "summary": "7 years security. Implemented zero-trust architecture. Expert in penetration testing, HIPAA compliance."
    },
    {
        "id": "11",
        "name": "Rachel Green",
        "title": "AI Research Engineer",
        "experience_years": 6,
        "skills": ["Python", "PyTorch", "NLP", "Transformers", "LLMs"],
        "domains": ["Healthcare", "AI", "NLP"],
        "availability": "immediate",
        "summary": "6 years AI research. Built medical text analysis with 92% accuracy. Expert in NLP, transformers, medical AI."
    },
    {
        "id": "12",
        "name": "David Johnson",
        "title": "Solutions Architect",
        "experience_years": 12,
        "skills": ["AWS", "Architecture", "Microservices", "Java", "Python"],
        "domains": ["Healthcare", "Enterprise", "Integration"],
        "availability": "1 month",
        "summary": "12 years architecture. Integrated 100+ healthcare systems. Expert in AWS, enterprise architecture, HIPAA."
    },
    {
        "id": "13",
        "name": "Sophie Turner",
        "title": "Junior Developer",
        "experience_years": 2,
        "skills": ["JavaScript", "React", "Node.js", "MongoDB", "Git"],
        "domains": ["Web Development", "Healthcare", "Startups"],
        "availability": "immediate",
        "summary": "2 years development. Built appointment system for clinics. Eager learner in React, Node.js, full-stack."
    },
    {
        "id": "14",
        "name": "Marcus Williams",
        "title": "Platform Engineer",
        "experience_years": 9,
        "skills": ["Kubernetes", "Go", "Docker", "Helm", "GitOps"],
        "domains": ["Platform", "Healthcare", "DevOps"],
        "availability": "2 weeks",
        "summary": "9 years platform engineering. Built developer platform for 200+ engineers. Expert in Kubernetes, Go, GitOps."
    },
    {
        "id": "15",
        "name": "Lisa Anderson",
        "title": "QA Engineer",
        "experience_years": 5,
        "skills": ["Selenium", "Python", "JavaScript", "Cypress", "Testing"],
        "domains": ["Quality", "Healthcare", "Automation"],
        "availability": "immediate",
        "summary": "5 years QA. Achieved 95% test coverage, zero critical bugs. Expert in test automation, Selenium, Cypress."
    }
]

class EmployeeSearchEngine:
    """Simple but effective vector search engine"""
    
    def __init__(self):
        self.employees = EMPLOYEES_DATA
        self.embeddings = None
        self.model = model
        self._build_index()
    
    def _build_index(self):
        """Build embeddings for all employees"""
        # Create searchable text for each employee
        documents = []
        for emp in self.employees:
            text = f"{emp['name']} {emp['title']} {emp['summary']} "
            text += f"Skills: {', '.join(emp['skills'])} "
            text += f"Experience: {emp['experience_years']} years "
            text += f"Domains: {', '.join(emp['domains'])}"
            documents.append(text)
        
        # Generate embeddings
        self.embeddings = self.model.encode(documents, convert_to_numpy=True)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for employees using semantic similarity"""
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            employee = self.employees[idx].copy()
            employee['match_score'] = float(similarities[idx])
            employee['match_reason'] = self._generate_match_reason(query, employee)
            results.append(employee)
        
        return results
    
    def _generate_match_reason(self, query: str, employee: Dict) -> str:
        """Generate explanation for why employee matches"""
        query_lower = query.lower()
        reasons = []
        
        # Check skill matches
        matched_skills = [s for s in employee['skills'] if s.lower() in query_lower]
        if matched_skills:
            reasons.append(f"Has {', '.join(matched_skills)} skills")
        
        # Check domain matches
        for domain in employee['domains']:
            if domain.lower() in query_lower:
                reasons.append(f"Experience in {domain}")
        
        # Check experience level
        if "senior" in query_lower and employee['experience_years'] >= 7:
            reasons.append(f"{employee['experience_years']} years of experience")
        elif "junior" in query_lower and employee['experience_years'] <= 3:
            reasons.append("Junior level matches requirement")
        
        # Check availability
        if "immediate" in query_lower and employee['availability'] == "immediate":
            reasons.append("Available immediately")
        
        if not reasons:
            reasons.append(f"Strong match based on {employee['title']} background")
        
        return " • ".join(reasons)

# Initialize search engine
search_engine = EmployeeSearchEngine()

# API Endpoints
@app.get("/")
async def root():
    """Serve simple HTML interface"""
    return HTMLResponse(content=HTML_INTERFACE)

@app.post("/api/chat")
async def chat(request: QueryRequest):
    """Main chat endpoint"""
    start_time = time.time()
    
    # Search for matching employees
    results = search_engine.search(request.query, request.top_k)
    
    # Generate response
    if results:
        top_match = results[0]
        response = f"Based on your search for '{request.query}', I found {len(results)} qualified candidates.\n\n"
        response += f"**Top Recommendation:** {top_match['name']} ({top_match['title']})\n"
        response += f"• {top_match['summary']}\n"
        response += f"• Match Score: {top_match['match_score']:.2%}\n"
        response += f"• Why: {top_match['match_reason']}\n"
        response += f"• Availability: {top_match['availability']}\n\n"
        
        if len(results) > 1:
            response += "**Other Strong Candidates:**\n"
            for emp in results[1:3]:  # Show next 2
                response += f"• {emp['name']} - {emp['title']} ({emp['match_score']:.2%})\n"
    else:
        response = "No matching candidates found. Try broadening your search criteria."
    
    return {
        "response": response,
        "candidates": results,
        "processing_time": time.time() - start_time,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/employees")
async def get_employees():
    """Get all employees"""
    return {"employees": EMPLOYEES_DATA}

@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "employees_count": len(EMPLOYEES_DATA),
        "model": "all-MiniLM-L6-v2",
        "timestamp": datetime.utcnow().isoformat()
    }

# Simple HTML Interface (no separate frontend needed!)
HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HR Resource Query System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 900px;
            padding: 40px;
        }
        
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        
        .subtitle {
            color: #666;
            margin-bottom: 30px;
        }
        
        .search-box {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
        }
        
        input {
            flex: 1;
            padding: 15px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
        }
        
        button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .results {
            margin-top: 20px;
        }
        
        .response {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            white-space: pre-wrap;
            line-height: 1.6;
        }
        
        .candidate {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            transition: box-shadow 0.3s;
        }
        
        .candidate:hover {
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .candidate-header {
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 10px;
        }
        
        .candidate-name {
            font-size: 18px;
            font-weight: 600;
            color: #333;
        }
        
        .candidate-title {
            color: #666;
            margin-top: 5px;
        }
        
        .match-score {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        
        .candidate-details {
            margin-top: 15px;
            color: #555;
        }
        
        .skills {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        
        .skill-tag {
            background: #f0f0f0;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 14px;
        }
        
        .availability {
            margin-top: 10px;
            color: #667eea;
            font-weight: 500;
        }
        
        .loading {
            text-align: center;
            color: #666;
        }
        
        .examples {
            margin-bottom: 20px;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 10px;
        }
        
        .example-query {
            display: inline-block;
            margin: 5px;
            padding: 8px 15px;
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .example-query:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>HR Resource Query System</h1>
        <p class="subtitle">Find the perfect candidate using natural language</p>
        
        <div class="examples">
            <strong>Try these examples:</strong><br>
            <span class="example-query" onclick="setQuery('Find Python developers with healthcare experience')">Python + Healthcare</span>
            <span class="example-query" onclick="setQuery('Senior backend engineer with Java expertise')">Senior Java Backend</span>
            <span class="example-query" onclick="setQuery('DevOps engineer with Kubernetes experience')">DevOps + Kubernetes</span>
            <span class="example-query" onclick="setQuery('Junior developer available immediately')">Junior Immediate</span>
            <span class="example-query" onclick="setQuery('Mobile developer for healthcare app')">Mobile Healthcare</span>
        </div>
        
        <div class="search-box">
            <input type="text" id="query" placeholder="e.g., 'Find senior Python developers with healthcare experience'" />
            <button onclick="search()" id="searchBtn">Search</button>
        </div>
        
        <div id="results"></div>
    </div>
    
    <script>
        function setQuery(text) {
            document.getElementById('query').value = text;
            search();
        }
        
        async function search() {
            const query = document.getElementById('query').value;
            if (!query) return;
            
            const resultsDiv = document.getElementById('results');
            const searchBtn = document.getElementById('searchBtn');
            
            searchBtn.disabled = true;
            resultsDiv.innerHTML = '<div class="loading">Searching...</div>';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query, top_k: 5 })
                });
                
                const data = await response.json();
                
                let html = '<div class="response">' + data.response + '</div>';
                html += '<div class="candidates-list">';
                
                data.candidates.forEach(candidate => {
                    const score = (candidate.match_score * 100).toFixed(0);
                    html += `
                        <div class="candidate">
                            <div class="candidate-header">
                                <div>
                                    <div class="candidate-name">${candidate.name}</div>
                                    <div class="candidate-title">${candidate.title}</div>
                                </div>
                                <div class="match-score">${score}% Match</div>
                            </div>
                            <div class="candidate-details">
                                <p>${candidate.summary}</p>
                                <div class="skills">
                                    ${candidate.skills.map(skill => `<span class="skill-tag">${skill}</span>`).join('')}
                                </div>
                                <div class="availability">Available: ${candidate.availability}</div>
                            </div>
                        </div>
                    `;
                });
                
                html += '</div>';
                resultsDiv.innerHTML = html;
                
            } catch (error) {
                resultsDiv.innerHTML = '<div class="response">Error: ' + error.message + '</div>';
            } finally {
                searchBtn.disabled = false;
            }
        }
        
        // Allow Enter key to search
        document.getElementById('query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                search();
            }
        });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("Starting HR Resource Query System...")
    print("Access the application at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)