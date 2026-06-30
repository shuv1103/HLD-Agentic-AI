import json
import fitz  # PyMuPDF
import google.generativeai as genai
import streamlit as st

selected_path = st.session_state.get("selected_path")

def extract_tech_stack_from_pdf(pdf_path: str, gemini_api_key: str):
    """
    Extracts high-level tech stack requirements from a given PDF.
    """
    try:
        # Read the PDF text
        doc_text = ""
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                doc_text += page.get_text("text")

        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Define prompt
        prompt = f"""
                
        You are a senior software architect analyzing a Product Requirements Document (PRD).

        YOUR TASK:
        Read and understand the product requirements, then recommend the SPECIFIC technology stack needed to build this product successfully.

        ANALYSIS FRAMEWORK:

        1. UNDERSTAND THE PRODUCT CONTEXT:
        - What type of product is this? (e-commerce, fintech, healthcare, social media, etc.)
        - What are the core features and functionalities?
        - Who are the end users and what are their needs?

        2. ANALYZE KEY REQUIREMENTS:
        
        Functional Requirements:
        - User authentication and authorization needs
        - Data storage and retrieval patterns
        - Real-time vs batch processing requirements
        - Integration with external services/APIs
        - File/document handling requirements
        - Communication features (messaging, notifications)
        
        Non-Functional Requirements:
        - Expected user load (concurrent users, daily active users)
        - Performance targets (response time, throughput)
        - Availability requirements (uptime SLA %)
        - Scalability needs (horizontal/vertical scaling)
        - Security and compliance requirements (GDPR, HIPAA, PCI-DSS, etc.)
        
        Technical Constraints:
        - Team size and composition
        - Budget constraints (if mentioned)
        - Timeline and delivery schedule
        - Existing systems to integrate with
        - Deployment preferences (cloud/on-premise)

        3. RECOMMEND SPECIFIC TECHNOLOGIES:

        FRONTEND:
        - Analyze: Does it need web app? Mobile app? Both?
        - Consider: UI complexity, real-time updates, offline support
        - Recommend: Specific framework (React, Angular, Vue, React Native, Flutter, etc.)
        - Include: UI libraries, state management, build tools if needed
        
        BACKEND:
        - Analyze: API complexity, business logic, performance needs
        - Consider: Microservices vs monolith, async processing needs
        - Recommend: Specific language and framework (Node.js/Express, Python/FastAPI, Java/Spring Boot, Go, etc.)
        - Include: API style (REST/GraphQL/gRPC), authentication mechanism
        
        DATABASE:
        - Analyze: Data structure, relationships, query patterns, scale
        - Consider: ACID requirements, read/write patterns, data volume
        - Recommend: Specific database (PostgreSQL, MongoDB, MySQL, Redis, Cassandra, etc.)
        - Include: Caching layer if high performance is needed
        
        INTEGRATIONS:
        - Identify: Which external services are needed?
        - Examples: Payment gateways (Stripe, PayPal), SMS (Twilio), Email (SendGrid), 
            Credit bureaus (Experian), Cloud storage (AWS S3), OCR services, etc.
        - Recommend: Specific service providers based on requirements
        
        AUTHENTICATION & SECURITY:
        - Analyze: Security level needed, compliance requirements
        - Recommend: Auth solutions (OAuth2.0, JWT, Auth0, Keycloak)
        - Include: Encryption tools, security scanners, compliance tools
        
        INFRASTRUCTURE:
        - Analyze: Scale, availability, geographic distribution
        - Recommend: Cloud platform (AWS, GCP, Azure) or on-premise
        - Include: Container orchestration (Docker, Kubernetes), load balancers, CDN
        
        DEVOPS:
        - Analyze: Team size, deployment frequency, monitoring needs
        - Recommend: CI/CD tools (GitHub Actions, Jenkins, GitLab CI)
        - Include: Monitoring (Prometheus, Datadog), logging (ELK stack)
        
        COMPLIANCE:
        - Analyze: Industry regulations mentioned (banking, healthcare, etc.)
        - Recommend: Compliance-specific tools (KYC/AML providers, audit logging)

        4. MATCHING LOGIC:

        For Small Projects (< 10k users):
        - Lightweight frameworks, managed services, minimal infrastructure
        
        For Medium Projects (10k - 100k users):
        - Proven frameworks, moderate scalability, some redundancy
        
        For Large Projects (> 100k users):
        - Enterprise-grade solutions, high availability, horizontal scaling
        
        For Real-time Features:
        - WebSocket support, message queues, caching layers
        
        For High Security:
        - End-to-end encryption, security scanning, compliance tools
        
        For Financial/Banking:
        - High availability, strong consistency, audit trails, regulatory compliance
        
        For Healthcare:
        - HIPAA compliance, data encryption, secure communication
        
        For E-commerce:
        - Payment gateways, inventory management, recommendation engines

        5. OUTPUT FORMAT:

        Provide ONLY valid JSON with specific technology recommendations:

        {{
        "frontend": ["specific frameworks and tools with brief reason"],
        "backend": ["specific language, framework, and architecture with brief reason"],
        "database": ["specific database(s) with brief reason"],
        "integrations": ["specific third-party services needed based on requirements"],
        "authentication_security": ["specific auth and security tools based on requirements"],
        "infrastructure": ["specific cloud platform and orchestration tools based on scale"],
        "devops_tools": ["specific CI/CD, monitoring, and logging tools"],
        "compliance": ["specific compliance tools if regulatory requirements exist"],
        "other": ["message queues, caching, CDN, or other essential tools based on requirements"]
        }}

        CRITICAL RULES:
        - Recommend technologies that MATCH the actual scale and complexity described
        - Don't over-engineer for simple projects
        - Don't under-engineer for enterprise projects
        - Every recommendation should be justified by something in the requirements
        - Be specific (e.g., "PostgreSQL with Redis caching" not just "a database")
        - Consider the full stack needed to deliver the described product

        PRD Document:
        {doc_text}
        
        """

        response = model.generate_content(prompt)
        json_text = response.text.strip()

        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", json_text, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}

        return parsed
    except Exception as e:
        print(f"❌ BaseAgent Error: {e}")
        return {"error": str(e)}



                  