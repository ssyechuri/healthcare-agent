"""
Symptom Triage Agent - A2A Protocol Compatible
Medical triage system with GPT-4o intelligence
"""

import asyncio
import json
import logging
import os
import csv
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import uuid

# Core dependencies
import requests
import pandas as pd
from dotenv import load_dotenv

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TriageSession:
    """Medical triage session with comprehensive clinical data"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Patient data
    patient_name: str = ""
    patient_phone: str = ""
    chief_complaint: str = ""
    
    # Triage data
    symptoms: List[str] = field(default_factory=list)
    symptom_duration: str = ""
    severity_score: int = 0
    
    # Clinical assessment
    answers: Dict[str, str] = field(default_factory=dict)
    urgency_level: str = ""  # low, medium, high
    recommendation: str = ""
    doctor_type: str = ""
    
    # Session tracking
    conversation: List[Dict] = field(default_factory=list)
    current_stage: str = "initial"  # initial, generic, specific, assessment, complete
    
    def add_message(self, role: str, message: str):
        """Add message with clinical context"""
        self.conversation.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "message": message,
            "stage": self.current_stage,
            "turn_number": len(self.conversation) + 1
        })
    
    def to_a2a_handoff(self) -> Dict:
        """Generate A2A handoff data for supervisor agent"""
        return {
            "triage_complete": True,
            "patient_name": self.patient_name,
            "patient_phone": self.patient_phone,
            "chief_complaint": self.chief_complaint,
            "symptoms": self.symptoms,
            "symptom_duration": self.symptom_duration,
            "severity_score": self.severity_score,
            "urgency_level": self.urgency_level,
            "recommendation": self.recommendation,
            "doctor_type": self.doctor_type,
            "clinical_notes": self._generate_clinical_notes(),
            "session_id": self.session_id,
            "triage_timestamp": datetime.now().isoformat()
        }
    
    def _generate_clinical_notes(self) -> str:
        """Generate structured clinical notes"""
        notes = f"TRIAGE ASSESSMENT:\n"
        notes += f"Chief Complaint: {self.chief_complaint}\n"
        notes += f"Symptoms: {', '.join(self.symptoms)}\n"
        notes += f"Duration: {self.symptom_duration}\n"
        notes += f"Severity: {self.severity_score}/10\n"
        notes += f"Urgency: {self.urgency_level.upper()}\n"
        notes += f"Recommendation: {self.recommendation}\n"
        
        if self.answers:
            notes += f"\nCLINICAL RESPONSES:\n"
            for question, answer in self.answers.items():
                notes += f"Q: {question}\nA: {answer}\n"
        
        return notes

class SymptomDatabase:
    """Intelligent symptom database with CSV integration"""
    
    def __init__(self, csv_file_path: str = "symptoms.csv"):
        self.csv_file_path = csv_file_path
        self.symptoms_data = {}
        self.load_symptoms_database()
    
    def load_symptoms_database(self):
        """Load and validate symptoms database"""
        try:
            print(f"📋 LOADING SYMPTOMS DATABASE: {self.csv_file_path}")
            
            if not os.path.exists(self.csv_file_path):
                print(f"❌ SYMPTOMS FILE NOT FOUND: {self.csv_file_path}")
                self._create_sample_database()
                return
            
            df = pd.read_csv(self.csv_file_path)
            
            # Validate required columns
            required_columns = ['symptoms', 'question1', 'question2', 'question3']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ MISSING COLUMNS: {missing_columns}")
                self._create_sample_database()
                return
            
            # Load symptoms data
            for _, row in df.iterrows():
                symptom = row['symptoms'].lower().strip()
                self.symptoms_data[symptom] = {
                    'question1': row['question1'],
                    'question2': row['question2'],
                    'question3': row['question3']
                }
            
            print(f"✅ SYMPTOMS DATABASE LOADED: {len(self.symptoms_data)} symptoms")
            print(f"📝 Available symptoms: {list(self.symptoms_data.keys())[:5]}...")
            
        except Exception as e:
            logger.error(f"Failed to load symptoms database: {e}")
            self._create_sample_database()
    
    def _create_sample_database(self):
        """Create sample symptoms database if file doesn't exist"""
        print("🔧 CREATING SAMPLE SYMPTOMS DATABASE")
        
        sample_data = [
            {
                'symptoms': 'headache',
                'question1': 'Is the headache throbbing or constant?',
                'question2': 'Do you have any visual changes or sensitivity to light?',
                'question3': 'Have you had any recent head injuries or changes in medication?'
            },
            {
                'symptoms': 'chest pain',
                'question1': 'Is the pain sharp, crushing, or burning?',
                'question2': 'Does the pain radiate to your arm, jaw, or back?',
                'question3': 'Do you have shortness of breath or sweating?'
            },
            {
                'symptoms': 'abdominal pain',
                'question1': 'Where exactly is the pain located in your abdomen?',
                'question2': 'Is the pain crampy, sharp, or dull?',
                'question3': 'Do you have nausea, vomiting, or changes in bowel movements?'
            },
            {
                'symptoms': 'fever',
                'question1': 'What is your current temperature if you\'ve measured it?',
                'question2': 'Do you have chills, body aches, or sweating?',
                'question3': 'Do you have any other symptoms like cough, sore throat, or rash?'
            },
            {
                'symptoms': 'cough',
                'question1': 'Is it a dry cough or are you bringing up phlegm?',
                'question2': 'Do you have shortness of breath or wheezing?',
                'question3': 'Do you have fever, chest pain, or blood in your sputum?'
            },
            {
                'symptoms': 'back pain',
                'question1': 'Is the pain in your upper, middle, or lower back?',
                'question2': 'Does the pain shoot down your legs or cause numbness?',
                'question3': 'Did the pain start after an injury or gradually over time?'
            },
            {
                'symptoms': 'nausea',
                'question1': 'Are you actually vomiting or just feeling nauseous?',
                'question2': 'Do you have abdominal pain or diarrhea?',
                'question3': 'Have you eaten anything unusual or started new medications?'
            },
            {
                'symptoms': 'dizziness',
                'question1': 'Do you feel like the room is spinning or like you might faint?',
                'question2': 'Does it happen when you stand up or all the time?',
                'question3': 'Do you have hearing changes, headache, or chest pain?'
            }
        ]
        
        try:
            df = pd.DataFrame(sample_data)
            df.to_csv(self.csv_file_path, index=False)
            
            # Load the created data
            for item in sample_data:
                symptom = item['symptoms'].lower().strip()
                self.symptoms_data[symptom] = {
                    'question1': item['question1'],
                    'question2': item['question2'],
                    'question3': item['question3']
                }
            
            print(f"✅ SAMPLE DATABASE CREATED: {self.csv_file_path}")
            
        except Exception as e:
            logger.error(f"Failed to create sample database: {e}")
    
    def find_matching_symptoms(self, user_input: str) -> List[str]:
        """Intelligently match user input to known symptoms"""
        user_input_lower = user_input.lower()
        matched_symptoms = []
        
        # Direct matches
        for symptom in self.symptoms_data.keys():
            if symptom in user_input_lower:
                matched_symptoms.append(symptom)
        
        # Synonym matching
        symptom_synonyms = {
            'headache': ['head pain', 'migraine', 'head ache'],
            'chest pain': ['chest hurt', 'heart pain', 'chest discomfort'],
            'abdominal pain': ['stomach pain', 'belly pain', 'stomach ache', 'tummy pain'],
            'fever': ['temperature', 'hot', 'chills'],
            'cough': ['coughing', 'hacking'],
            'back pain': ['back hurt', 'spine pain'],
            'nausea': ['sick to stomach', 'queasy', 'nauseated'],
            'dizziness': ['dizzy', 'lightheaded', 'vertigo']
        }
        
        for symptom, synonyms in symptom_synonyms.items():
            if symptom not in matched_symptoms:
                for synonym in synonyms:
                    if synonym in user_input_lower:
                        matched_symptoms.append(symptom)
                        break
        
        return matched_symptoms
    
    def get_questions_for_symptom(self, symptom: str) -> List[str]:
        """Get follow-up questions for a specific symptom"""
        symptom_lower = symptom.lower()
        if symptom_lower in self.symptoms_data:
            data = self.symptoms_data[symptom_lower]
            return [data['question1'], data['question2'], data['question3']]
        return []

class MedicalIntelligence:
    """Advanced medical AI for clinical decision making with Azure OpenAI support"""
    
    def __init__(self, openai_url: str, openai_api_key: str, openai_project_id: str = "", 
                 openai_connection_id: str = "", openai_connection_name: str = "", 
                 openai_provider: str = "Azure-OpenAI", openai_model: str = "gpt-4o"):
        self.openai_url = openai_url
        self.openai_api_key = openai_api_key
        self.openai_project_id = openai_project_id
        self.openai_connection_id = openai_connection_id
        self.openai_connection_name = openai_connection_name
        self.openai_provider = openai_provider
        self.openai_model = openai_model
        
        # Working format confirmed: Standard Authorization Bearer
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {openai_api_key}'
        }
        
        print(f"🧠 MEDICAL AI INITIALIZED")
        print(f"🔗 Provider: {openai_provider}")
        print(f"🤖 Model: {openai_model}")
        print(f"🌐 Endpoint: {openai_url}")
        print(f"🔑 Auth Header: Authorization: Bearer {openai_api_key[:8]}...{openai_api_key[-4:]}")
        if openai_project_id:
            print(f"📁 Project: {openai_project_id}")
        if openai_connection_id:
            print(f"🔌 Connection: {openai_connection_id}")
    
    async def process_triage_response(self, user_input: str, session: TriageSession, 
                                   context: str = "") -> Dict:
        """Process user response with medical intelligence"""
        
        print(f"\n🧠 MEDICAL AI PROCESSING")
        print(f"👤 User Input: '{user_input}'")
        print(f"📋 Stage: {session.current_stage}")
        
        # Construct medical prompt based on current stage
        prompt = self._construct_medical_prompt(user_input, session, context)
        
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]
            
            # Working payload format from successful test
            payload = {
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 800
            }
            
            print(f"📤 MEDICAL AI REQUEST:")
            print(f"   URL: {self.openai_url}")
            print(f"   Headers: {json.dumps({k: v if k != 'Authorization' else f'Bearer {v[7:15]}...' for k, v in self.headers.items()}, indent=6)}")
            print(f"   Payload: {json.dumps({k: v if k != 'messages' else f'[{len(v)} messages]' for k, v in payload.items()}, indent=6)}")
            
            def _execute_medical_request():
                try:
                    print(f"🌐 Making POST request to: {self.openai_url}")
                    print(f"🔑 Headers being sent: {dict(self.headers)}")  # Show actual headers
                    
                    response = requests.post(
                        self.openai_url,
                        headers=self.headers,
                        json=payload,
                        timeout=30
                    )
                    
                    print(f"📥 HTTP Response Status: {response.status_code}")
                    print(f"📥 Response Headers: {dict(response.headers)}")
                    
                    if response.status_code == 401:
                        print(f"❌ AUTHENTICATION ERROR Details:")
                        print(f"   - Status: {response.status_code}")
                        print(f"   - Response: {response.text}")
                        print(f"   - Headers sent: {self.headers}")
                        return {"error": f"Authentication failed (401). Response: {response.text}"}
                    elif response.status_code == 403:
                        print(f"❌ AUTHORIZATION ERROR Details:")
                        print(f"   - Status: {response.status_code}")
                        print(f"   - Response: {response.text}")
                        print(f"   - Response Headers: {dict(response.headers)}")
                        print(f"   - Headers we sent: {dict(self.headers)}")
                        print(f"   - JWT Token length: {len(self.openai_api_key)} characters")
                        print(f"   - JWT Token starts with: {self.openai_api_key[:20]}...")
                        return {"error": f"Authorization failed (403): {response.text}"}
                    elif response.status_code == 404:
                        print(f"❌ ENDPOINT ERROR - Check URL")
                        print(f"   - URL: {self.openai_url}")
                        print(f"   - Response: {response.text}")
                        return {"error": f"Endpoint not found (404): {response.text}"}
                    elif response.status_code >= 400:
                        print(f"❌ HTTP ERROR {response.status_code}")
                        print(f"   - Response: {response.text}")
                        return {"error": f"HTTP error {response.status_code}: {response.text}"}
                    
                    response.raise_for_status()
                    response_json = response.json()
                    print(f"✅ SUCCESS - Response received")
                    return response_json
                    
                except requests.exceptions.Timeout:
                    return {"error": "Request timeout - AI service took too long to respond"}
                except requests.exceptions.ConnectionError as e:
                    print(f"❌ CONNECTION ERROR: {e}")
                    return {"error": f"Connection error - Cannot reach {self.openai_url}. Check if the endpoint is correct and accessible."}
                except requests.exceptions.HTTPError as e:
                    print(f"❌ HTTP ERROR: {e}")
                    return {"error": f"HTTP error {e.response.status_code}: {e.response.text}"}
                except json.JSONDecodeError as e:
                    print(f"❌ JSON DECODE ERROR: {e}")
                    return {"error": f"Invalid JSON response: {e}"}
                except Exception as e:
                    print(f"❌ UNEXPECTED ERROR: {e}")
                    return {"error": f"Request failed: {str(e)}"}
            
            loop = asyncio.get_event_loop()
            raw_response = await loop.run_in_executor(None, _execute_medical_request)
            
            if "error" in raw_response:
                print(f"❌ MEDICAL AI ERROR: {raw_response['error']}")
                return self._create_medical_fallback(session.current_stage)
            
            print(f"📥 MEDICAL AI RAW RESPONSE:")
            print(json.dumps(raw_response, indent=2))
            
            # Extract response - handle different response formats
            ai_content = ""
            if 'choices' in raw_response and raw_response['choices']:
                if 'message' in raw_response['choices'][0]:
                    ai_content = raw_response['choices'][0]['message']['content']
                elif 'text' in raw_response['choices'][0]:
                    ai_content = raw_response['choices'][0]['text']
            elif 'response' in raw_response:
                ai_content = raw_response['response']
            elif 'content' in raw_response:
                ai_content = raw_response['content']
            elif 'completion' in raw_response:
                ai_content = raw_response['completion']
            
            print(f"🧠 EXTRACTED AI CONTENT: {ai_content}")
            
            if ai_content:
                parsed_response = self._parse_medical_response(ai_content)
                return parsed_response
            else:
                print(f"⚠️  MEDICAL AI WARNING: Empty response content")
                print(f"   Full response keys: {list(raw_response.keys())}")
                return self._create_medical_fallback(session.current_stage)
                
        except Exception as e:
            logger.error(f"Medical AI processing error: {e}")
            print(f"💥 MEDICAL AI EXCEPTION: {str(e)}")
            return self._create_medical_fallback(session.current_stage)
    
    def _construct_medical_prompt(self, user_input: str, session: TriageSession, context: str) -> str:
        """Construct stage-specific medical prompts"""
        
        base_context = f"""
CURRENT SESSION DATA: {json.dumps(session.__dict__, default=str, indent=2)}
CONVERSATION HISTORY: {json.dumps(session.conversation[-3:], indent=2)}
ADDITIONAL CONTEXT: {context}
"""
        
        if session.current_stage == "initial":
            return f"""You are an expert medical triage nurse with 20+ years of experience. You are conducting initial symptom assessment.

{base_context}

TASK: Analyze the user's chief complaint and extract symptoms.

GUIDELINES:
- Identify medical symptoms mentioned
- Determine if this is a medical concern requiring triage
- Extract patient demographics if provided
- Be professional and empathetic

REQUIRED JSON RESPONSE:
{{
    "response": "your professional response to the patient",
    "is_medical": true/false,
    "symptoms_identified": ["symptom1", "symptom2"],
    "extract": {{"field_name": "value"}},
    "next_stage": "generic|complete",
    "medical_concern": true/false
}}"""
        
        elif session.current_stage == "generic":
            return f"""You are conducting generic symptom assessment. Ask about duration and severity.

{base_context}

CURRENT TASK: Process duration and severity responses.

GUIDELINES:
- Extract symptom duration (convert to standardized format)
- Extract severity score (1-10 scale)
- Validate responses are reasonable
- Move to specific questions next

REQUIRED JSON RESPONSE:
{{
    "response": "your response acknowledging the information",
    "extract": {{"symptom_duration": "standardized_duration", "severity_score": number}},
    "next_stage": "specific",
    "duration_valid": true/false,
    "severity_valid": true/false
}}"""
        
        elif session.current_stage == "specific":
            return f"""You are processing specific symptom follow-up questions.

{base_context}

TASK: Process answer to specific clinical question and determine next action.

GUIDELINES:
- Record the clinical answer accurately
- Determine if more questions needed for current symptom
- Decide if ready for final assessment
- Maintain clinical accuracy

REQUIRED JSON RESPONSE:
{{
    "response": "acknowledgment and next question or transition",
    "extract": {{"answer_key": "user_response"}},
    "next_stage": "specific|assessment",
    "question_complete": true/false,
    "ready_for_assessment": true/false
}}"""
        
        elif session.current_stage == "assessment":
            return f"""You are an expert medical triage specialist conducting FINAL URGENCY ASSESSMENT.

{base_context}

CRITICAL TASK: Determine urgency level and provide medical recommendation.

URGENCY LEVELS:
- HIGH: Life-threatening, immediate 911 required (chest pain with radiation, severe breathing issues, severe trauma, altered mental status, severe allergic reactions)
- MEDIUM: Urgent care needed, specialty referral (moderate to severe symptoms requiring prompt attention, potential complications)
- LOW: Can see general practitioner (mild symptoms, routine care, preventive care)

DOCTOR RECOMMENDATIONS:
- High urgency: "EMERGENCY - Call 911 immediately"
- Medium urgency: Specific specialist (cardiologist, neurologist, gastroenterologist, etc.)
- Low urgency: "General practitioner"

REQUIRED JSON RESPONSE:
{{
    "response": "your professional assessment and recommendation",
    "urgency_level": "low|medium|high",
    "doctor_type": "specific doctor type or 911",
    "recommendation": "detailed medical recommendation",
    "reasoning": "clinical reasoning for urgency level",
    "next_stage": "complete",
    "emergency_alert": true/false
}}"""
        
        return f"""You are a medical triage expert. Process this medical interaction professionally.

{base_context}

Respond with appropriate medical guidance in JSON format."""
    
    def _parse_medical_response(self, response_text: str) -> Dict:
        """Parse medical AI response with error recovery"""
        try:
            # Clean response
            content = response_text.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            parsed = json.loads(content)
            
            # Validate required fields exist
            if "response" not in parsed:
                parsed["response"] = "I understand. Please continue."
            
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"❌ MEDICAL JSON PARSE ERROR: {e}")
            # Try to extract response with regex
            response_match = re.search(r'"response":\s*"([^"]*)"', response_text)
            if response_match:
                return {
                    "response": response_match.group(1),
                    "extract": {},
                    "next_stage": "generic"
                }
            return self._create_medical_fallback("generic")
        
        except Exception as e:
            print(f"❌ MEDICAL PARSE EXCEPTION: {e}")
            return self._create_medical_fallback("generic")
    
    def _create_medical_fallback(self, current_stage: str) -> Dict:
        """Create appropriate fallback response based on stage"""
        fallbacks = {
            "initial": {
                "response": "I understand you have a medical concern. Could you tell me more about your symptoms?",
                "is_medical": True,
                "symptoms_identified": [],
                "extract": {},
                "next_stage": "generic"
            },
            "generic": {
                "response": "Thank you for that information. Let me ask you some specific questions.",
                "extract": {},
                "next_stage": "specific"
            },
            "specific": {
                "response": "I understand. Let me ask you another question.",
                "extract": {},
                "next_stage": "specific"
            },
            "assessment": {
                "response": "Based on your symptoms, I recommend seeing a general practitioner.",
                "urgency_level": "low",
                "doctor_type": "general practitioner",
                "recommendation": "Schedule an appointment with your primary care physician",
                "next_stage": "complete"
            }
        }
        
        return fallbacks.get(current_stage, fallbacks["generic"])

class SymptomTriageAgent:
    """Production Symptom Triage Agent with A2A Protocol"""
    
    def __init__(self, openai_url: str, openai_api_key: str, openai_project_id: str = "",
                 openai_connection_id: str = "", openai_connection_name: str = "",
                 openai_provider: str = "Azure-OpenAI", openai_model: str = "gpt-4o",
                 symptoms_csv: str = "symptoms.csv"):
        print("🏥 SYMPTOM TRIAGE AGENT INITIALIZATION")
        print("=" * 50)
        
        self.medical_ai = MedicalIntelligence(
            openai_url=openai_url,
            openai_api_key=openai_api_key,
            openai_project_id=openai_project_id,
            openai_connection_id=openai_connection_id,
            openai_connection_name=openai_connection_name,
            openai_provider=openai_provider,
            openai_model=openai_model
        )
        self.symptom_db = SymptomDatabase(symptoms_csv)
        self.current_session = None
        
        print("✅ TRIAGE AGENT READY")
        print("=" * 50)
    
    async def start_triage_session(self, patient_name: str = "", patient_phone: str = "", 
                                 chief_complaint: str = "") -> TriageSession:
        """Start new triage session with A2A handoff capability"""
        
        print(f"🆕 STARTING TRIAGE SESSION")
        print(f"👤 Patient: {patient_name}")
        print(f"📞 Phone: {patient_phone}")
        print(f"🩺 Complaint: {chief_complaint}")
        
        self.current_session = TriageSession(
            patient_name=patient_name,
            patient_phone=patient_phone,
            chief_complaint=chief_complaint,
            current_stage="initial"
        )
        
        return self.current_session
    
    async def process_user_input(self, user_input: str) -> Dict:
        """Process user input through triage workflow"""
        
        if not self.current_session:
            raise ValueError("No active triage session. Call start_triage_session first.")
        
        print(f"\n🔄 PROCESSING TRIAGE INPUT")
        print(f"📋 Stage: {self.current_session.current_stage}")
        print(f"👤 Input: '{user_input}'")
        
        # Add user message to session
        self.current_session.add_message("user", user_input)
        
        # Process based on current stage
        if self.current_session.current_stage == "initial":
            result = await self._process_initial_assessment(user_input)
        elif self.current_session.current_stage == "generic":
            result = await self._process_generic_questions(user_input)
        elif self.current_session.current_stage == "specific":
            result = await self._process_specific_questions(user_input)
        elif self.current_session.current_stage == "assessment":
            result = await self._process_final_assessment(user_input)
        else:
            result = {"response": "Triage session complete.", "stage_complete": True}
        
        # Add agent response to session
        if result.get("response"):
            self.current_session.add_message("assistant", result["response"])
        
        return result
    
    async def _process_initial_assessment(self, user_input: str) -> Dict:
        """Process initial symptom identification"""
        
        print("🔍 INITIAL ASSESSMENT")
        
        # Use AI to analyze chief complaint
        ai_result = await self.medical_ai.process_triage_response(
            user_input, self.current_session, 
            "Analyze chief complaint and identify symptoms"
        )
        
        # Update session with extracted data
        if ai_result.get("extract"):
            for key, value in ai_result["extract"].items():
                setattr(self.current_session, key, value)
        
        # Identify symptoms
        symptoms_mentioned = ai_result.get("symptoms_identified", [])
        if not symptoms_mentioned:
            # Fallback symptom detection
            symptoms_mentioned = self.symptom_db.find_matching_symptoms(user_input)
        
        self.current_session.symptoms = symptoms_mentioned
        
        # Determine if medical triage needed
        is_medical = ai_result.get("is_medical", True)
        
        if not is_medical:
            # Non-medical complaint
            self.current_session.current_stage = "complete"
            self.current_session.urgency_level = "low"
            self.current_session.recommendation = "Non-medical appointment scheduling"
            
            return {
                "response": "I understand this is not a medical emergency. Let me help you schedule your appointment.",
                "triage_complete": True,
                "is_medical": False,
                "stage_complete": True,
                "a2a_handoff": self.current_session.to_a2a_handoff()
            }
        
        # Move to generic questions
        self.current_session.current_stage = "generic"
        
        response = ai_result.get("response", "I understand your symptoms. Let me ask you a couple of questions.")
        response += "\n\nFirst, how long have you been experiencing these symptoms?"
        
        return {
            "response": response,
            "symptoms_identified": symptoms_mentioned,
            "next_question": "duration",
            "stage_complete": False
        }
    
    async def _process_generic_questions(self, user_input: str) -> Dict:
        """Process generic triage questions (duration and severity)"""
        
        print("🔄 GENERIC QUESTIONS")
        
        # Use AI to extract duration and severity
        ai_result = await self.medical_ai.process_triage_response(
            user_input, self.current_session,
            "Extract symptom duration and severity information"
        )
        
        # Update session data
        if ai_result.get("extract"):
            for key, value in ai_result["extract"].items():
                setattr(self.current_session, key, value)
        
        # Check what we still need
        need_duration = not self.current_session.symptom_duration
        need_severity = not self.current_session.severity_score
        
        if need_duration:
            return {
                "response": "Thank you. How long have you been experiencing these symptoms?",
                "next_question": "duration",
                "stage_complete": False
            }
        elif need_severity:
            return {
                "response": "Thank you. On a scale of 1 to 10, with 10 being the worst pain imaginable, how severe are your symptoms?",
                "next_question": "severity",
                "stage_complete": False
            }
        else:
            # Both collected, move to specific questions
            self.current_session.current_stage = "specific"
            return await self._start_specific_questions()
    
    async def _start_specific_questions(self) -> Dict:
        """Start asking specific symptom questions"""
        
        print("🎯 STARTING SPECIFIC QUESTIONS")
        
        if not self.current_session.symptoms:
            # No specific symptoms identified, move to assessment
            self.current_session.current_stage = "assessment"
            return await self._process_final_assessment("")
        
        # Get questions for first symptom
        first_symptom = self.current_session.symptoms[0]
        questions = self.symptom_db.get_questions_for_symptom(first_symptom)
        
        if not questions:
            # No questions available, move to assessment
            self.current_session.current_stage = "assessment"
            return await self._process_final_assessment("")
        
        # Store current symptom and question info
        self.current_session.current_symptom = first_symptom
        self.current_session.current_questions = questions
        self.current_session.current_question_index = 0
        
        first_question = questions[0]
        
        return {
            "response": f"Now I need to ask you some specific questions about your {first_symptom}. {first_question}",
            "current_symptom": first_symptom,
            "question_number": 1,
            "total_questions": len(questions),
            "stage_complete": False
        }
    
    async def _process_specific_questions(self, user_input: str) -> Dict:
        """Process specific symptom follow-up questions"""
        
        print("🎯 SPECIFIC QUESTIONS")
        
        # Store the answer
        current_symptom = getattr(self.current_session, 'current_symptom', '')
        current_q_index = getattr(self.current_session, 'current_question_index', 0)
        questions = getattr(self.current_session, 'current_questions', [])
        
        if questions and current_q_index < len(questions):
            question_key = f"{current_symptom}_q{current_q_index + 1}"
            self.current_session.answers[question_key] = user_input
        
        # Move to next question
        self.current_session.current_question_index = current_q_index + 1
        
        # Check if more questions for current symptom
        if self.current_session.current_question_index < len(questions):
            next_question = questions[self.current_session.current_question_index]
            question_num = self.current_session.current_question_index + 1
            
            return {
                "response": f"Thank you. {next_question}",
                "question_number": question_num,
                "total_questions": len(questions),
                "stage_complete": False
            }
        
        # Finished questions for current symptom
        # Check if more symptoms to process
        current_symptom_index = self.current_session.symptoms.index(current_symptom)
        if current_symptom_index + 1 < len(self.current_session.symptoms):
            # Move to next symptom
            next_symptom = self.current_session.symptoms[current_symptom_index + 1]
            next_questions = self.symptom_db.get_questions_for_symptom(next_symptom)
            
            if next_questions:
                self.current_session.current_symptom = next_symptom
                self.current_session.current_questions = next_questions
                self.current_session.current_question_index = 0
                
                return {
                    "response": f"Thank you. Now let me ask about your {next_symptom}. {next_questions[0]}",
                    "current_symptom": next_symptom,
                    "question_number": 1,
                    "total_questions": len(next_questions),
                    "stage_complete": False
                }
        
        # All questions complete, move to assessment
        self.current_session.current_stage = "assessment"
        return {
            "response": "Thank you for answering all the questions. Let me assess your symptoms now.",
            "questions_complete": True,
            "moving_to_assessment": True,
            "stage_complete": False
        }
    
    async def _process_final_assessment(self, user_input: str) -> Dict:
        """Process final medical assessment and urgency determination"""
        
        print("⚕️ FINAL MEDICAL ASSESSMENT")
        
        # Use AI for final clinical assessment
        assessment_context = f"""
CLINICAL DATA FOR ASSESSMENT:
- Symptoms: {', '.join(self.current_session.symptoms)}
- Duration: {self.current_session.symptom_duration}
- Severity: {self.current_session.severity_score}/10
- Clinical Answers: {json.dumps(self.current_session.answers, indent=2)}

Determine urgency level and appropriate medical recommendation.
"""
        
        ai_result = await self.medical_ai.process_triage_response(
            "CONDUCT_FINAL_ASSESSMENT", self.current_session, assessment_context
        )
        
        # Extract assessment results
        urgency_level = ai_result.get("urgency_level", "low")
        doctor_type = ai_result.get("doctor_type", "general practitioner")
        recommendation = ai_result.get("recommendation", "Schedule an appointment with your primary care physician")
        
        # Update session with final assessment
        self.current_session.urgency_level = urgency_level
        self.current_session.doctor_type = doctor_type
        self.current_session.recommendation = recommendation
        self.current_session.current_stage = "complete"
        self.current_session.end_time = datetime.now()
        
        # Handle emergency situations
        if urgency_level.lower() == "high" or ai_result.get("emergency_alert", False):
            emergency_response = "⚠️ EMERGENCY: Based on your symptoms, this could be a medical emergency. Please hang up immediately and call 911 or go to the nearest emergency room."
            
            return {
                "response": emergency_response,
                "urgency_level": "high",
                "emergency_alert": True,
                "recommendation": "CALL 911 IMMEDIATELY",
                "triage_complete": True,
                "stage_complete": True,
                "end_call": True
            }
        
        # Generate final recommendation response
        response = ai_result.get("response", "")
        if not response:
            if urgency_level.lower() == "medium":
                response = f"Based on your symptoms, I recommend seeing a {doctor_type} soon. {recommendation}"
            else:
                response = f"Based on your symptoms, you can schedule an appointment with a {doctor_type}. {recommendation}"
        
        return {
            "response": response,
            "urgency_level": urgency_level,
            "doctor_type": doctor_type,
            "recommendation": recommendation,
            "triage_complete": True,
            "stage_complete": True,
            "a2a_handoff": self.current_session.to_a2a_handoff()
        }
    
    def get_triage_summary(self) -> Dict:
        """Get comprehensive triage summary for A2A handoff"""
        if not self.current_session:
            return {"error": "No active triage session"}
        
        return self.current_session.to_a2a_handoff()
    
    def save_triage_session(self) -> str:
        """Save triage session for records"""
        if not self.current_session:
            return ""
        
        try:
            os.makedirs("triage_sessions", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"triage_sessions/triage_{timestamp}_{self.current_session.session_id[:8]}.json"
            
            session_data = {
                "session_metadata": {
                    "session_id": self.current_session.session_id,
                    "start_time": self.current_session.start_time.isoformat(),
                    "end_time": self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                    "duration_minutes": ((self.current_session.end_time or datetime.now()) - self.current_session.start_time).total_seconds() / 60
                },
                "patient_info": {
                    "name": self.current_session.patient_name,
                    "phone": self.current_session.patient_phone,
                    "chief_complaint": self.current_session.chief_complaint
                },
                "clinical_assessment": {
                    "symptoms": self.current_session.symptoms,
                    "duration": self.current_session.symptom_duration,
                    "severity_score": self.current_session.severity_score,
                    "urgency_level": self.current_session.urgency_level,
                    "recommendation": self.current_session.recommendation,
                    "doctor_type": self.current_session.doctor_type
                },
                "clinical_responses": self.current_session.answers,
                "conversation_log": self.current_session.conversation,
                "clinical_notes": self.current_session._generate_clinical_notes()
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 TRIAGE SESSION SAVED: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save triage session: {e}")
            return ""

# A2A Protocol Integration Functions
class A2AProtocol:
    """Agent-to-Agent Protocol for Supervisor Integration"""
    
    @staticmethod
    def create_triage_handoff(triage_agent: SymptomTriageAgent) -> Dict:
        """Create A2A handoff data for supervisor agent"""
        
        if not triage_agent.current_session:
            return {"error": "No active triage session"}
        
        handoff_data = triage_agent.get_triage_summary()
        
        # Add A2A protocol metadata
        handoff_data.update({
            "a2a_protocol_version": "1.0",
            "source_agent": "symptom_triage",
            "target_agent": "supervisor_voice",
            "handoff_timestamp": datetime.now().isoformat(),
            "handoff_complete": True
        })
        
        print(f"🔗 A2A HANDOFF CREATED")
        print(f"📋 Urgency: {handoff_data.get('urgency_level', 'unknown')}")
        print(f"🩺 Recommendation: {handoff_data.get('doctor_type', 'unknown')}")
        
        return handoff_data
    
    @staticmethod
    def process_supervisor_request(patient_name: str, patient_phone: str, 
                                 chief_complaint: str) -> Dict:
        """Process request from supervisor agent to start triage"""
        
        return {
            "action": "start_triage",
            "patient_name": patient_name,
            "patient_phone": patient_phone,
            "chief_complaint": chief_complaint,
            "triage_required": True,
            "timestamp": datetime.now().isoformat()
        }

def load_triage_config():
    """Load triage agent configuration"""
    try:
        load_dotenv()
        print("🔧 Loading triage configuration...")
    except ImportError:
        print("⚠️ python-dotenv not available, reading from environment...")
    
    config = {}
    required_configs = {
        'openai_url': 'OPENAI_URL',
        'openai_api_key': 'OPENAI_API_KEY'
    }
    
    # Optional configs for Azure OpenAI
    optional_configs = {
        'openai_project_id': 'OPENAI_PROJECT_ID',
        'openai_connection_id': 'OPENAI_CONNECTION_ID',
        'openai_connection_name': 'OPENAI_CONNECTION_NAME',
        'openai_provider': 'OPENAI_PROVIDER',
        'openai_model': 'OPENAI_MODEL',
        'symptoms_csv': 'SYMPTOMS_CSV'
    }
    
    print("🔍 VALIDATING TRIAGE CONFIGURATION:")
    missing_configs = []
    
    # Check required configs
    for key, env_var in required_configs.items():
        value = os.getenv(env_var)
        if value and value.strip():
            config[key] = value.strip()
            # Show partial value for security
            display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"   ✅ {env_var}: {display_value}")
        else:
            missing_configs.append(env_var)
            print(f"   ❌ {env_var}: MISSING")
    
    # Check optional configs
    for key, env_var in optional_configs.items():
        value = os.getenv(env_var)
        if value and value.strip():
            config[key] = value.strip()
            print(f"   ✅ {env_var}: {value}")
        else:
            # Set defaults for optional configs
            defaults = {
                'openai_project_id': '',
                'openai_connection_id': '',
                'openai_connection_name': '',
                'openai_provider': 'Azure-OpenAI',
                'openai_model': 'gpt-4o',
                'symptoms_csv': 'symptoms.csv'
            }
            config[key] = defaults.get(key, '')
            print(f"   📋 {env_var}: {config[key]} (default)")
    
    if missing_configs:
        print(f"\n❌ CONFIGURATION ERROR: Missing required environment variables:")
        for var in missing_configs:
            print(f"   - {var}")
        return None
    
    print("✅ TRIAGE CONFIGURATION VALIDATED")
    print(f"🤖 Provider: {config.get('openai_provider', 'Unknown')}")
    print(f"🎯 Model: {config.get('openai_model', 'Unknown')}")
    
    return config

# Example A2A Integration with Supervisor Agent
async def integrate_with_supervisor(patient_name: str, patient_phone: str, chief_complaint: str):
    """Example integration function for supervisor agent"""
    
    print(f"🔗 A2A INTEGRATION INITIATED")
    print(f"👤 Patient: {patient_name}")
    print(f"🩺 Complaint: {chief_complaint}")
    
    # Load configuration
    config = load_triage_config()
    if not config:
        return {"error": "Configuration missing"}
    
    try:
        # Initialize triage agent with full parameters
        triage_agent = SymptomTriageAgent(
            openai_url=config['openai_url'],
            openai_api_key=config['openai_api_key'],
            openai_project_id=config.get('openai_project_id', ''),
            openai_connection_id=config.get('openai_connection_id', ''),
            openai_connection_name=config.get('openai_connection_name', ''),
            openai_provider=config.get('openai_provider', 'Azure-OpenAI'),
            openai_model=config.get('openai_model', 'gpt-4o'),
            symptoms_csv=config['symptoms_csv']
        )
        
        # Start triage session
        session = await triage_agent.start_triage_session(
            patient_name=patient_name,
            patient_phone=patient_phone,
            chief_complaint=chief_complaint
        )
        
        print(f"✅ A2A TRIAGE SESSION STARTED: {session.session_id}")
        
        # Return triage agent for continued interaction
        return {
            "success": True,
            "triage_agent": triage_agent,
            "session_id": session.session_id,
            "ready_for_interaction": True
        }
        
    except Exception as e:
        logger.error(f"A2A integration failed: {e}")
        return {"error": str(e)}

# Standalone testing function
async def test_triage_agent():
    """Test the triage agent standalone"""
    
    print("🧪 TESTING SYMPTOM TRIAGE AGENT")
    print("=" * 40)
    
    config = load_triage_config()
    if not config:
        print("❌ Cannot test without configuration")
        return
    
    # Initialize agent with full parameters
    agent = SymptomTriageAgent(
        openai_url=config['openai_url'],
        openai_api_key=config['openai_api_key'],
        openai_project_id=config.get('openai_project_id', ''),
        openai_connection_id=config.get('openai_connection_id', ''),
        openai_connection_name=config.get('openai_connection_name', ''),
        openai_provider=config.get('openai_provider', 'Azure-OpenAI'),
        openai_model=config.get('openai_model', 'gpt-4o'),
        symptoms_csv=config['symptoms_csv']
    )
    
    # Test session
    session = await agent.start_triage_session(
        patient_name="John Test",
        patient_phone="555-0123",
        chief_complaint="I have a headache"
    )
    
    # Simulate conversation
    test_inputs = [
        "I have a severe headache",
        "About 3 hours",
        "8 out of 10",
        "It's throbbing",
        "Yes, bright lights hurt my eyes",
        "No recent injuries"
    ]
    
    for user_input in test_inputs:
        print(f"\n👤 USER: {user_input}")
        result = await agent.process_user_input(user_input)
        print(f"🏥 AGENT: {result.get('response', '')}")
        
        if result.get('triage_complete'):
            print(f"✅ TRIAGE COMPLETE")
            print(f"🎯 Urgency: {result.get('urgency_level')}")
            print(f"🩺 Recommendation: {result.get('doctor_type')}")
            break
    
    # Save session
    filename = agent.save_triage_session()
    print(f"💾 Test session saved: {filename}")
    
    # Get A2A handoff data
    handoff = A2AProtocol.create_triage_handoff(agent)
    print(f"🔗 A2A Handoff Data:")
    print(json.dumps(handoff, indent=2))

if __name__ == "__main__":
    # Run test
    asyncio.run(test_triage_agent())
