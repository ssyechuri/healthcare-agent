"""
Enhanced Healthcare Voice Agent - A2A Triage Integration + MCP Insurance Integration
"""

import asyncio
import json
import logging
import tempfile
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import uuid
import random
import string

# Core dependencies
import requests
import speech_recognition as sr
from gtts import gTTS
import pygame
import pyttsx3

# Import our symptom triage agent
from symp_triage import SymptomTriageAgent, A2AProtocol, load_triage_config

# Enhanced logging with clear formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedSession:
    """Enhanced session container with triage integration"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    data: Dict[str, Any] = field(default_factory=dict)
    conversation: List[Dict] = field(default_factory=list)
    api_calls: List[Dict] = field(default_factory=list)
    
    # Triage integration
    triage_required: bool = False
    triage_complete: bool = False
    triage_data: Dict = field(default_factory=dict)
    current_flow: str = "standard"  # standard, triage, post_triage
    
    def add_message(self, role: str, message: str):
        """Add message with enhanced metadata"""
        self.conversation.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "message": message,
            "turn_number": len(self.conversation) + 1,
            "current_flow": self.current_flow
        })
    
    def add_triage_data(self, triage_handoff: Dict):
        """Add triage data from A2A handoff"""
        self.triage_data = triage_handoff
        self.triage_complete = True
        
        # Extract key triage information
        if triage_handoff.get('urgency_level'):
            self.data['urgency_level'] = triage_handoff['urgency_level']
        if triage_handoff.get('doctor_type'):
            self.data['recommended_doctor'] = triage_handoff['doctor_type']
        if triage_handoff.get('clinical_notes'):
            self.data['clinical_notes'] = triage_handoff['clinical_notes']
    
    def add_api_call(self, api_type: str, request_data: Dict, response_data: Dict, success: bool):
        """Track API calls for debugging and analytics"""
        self.api_calls.append({
            "timestamp": datetime.now().isoformat(),
            "api_type": api_type,
            "request": request_data,
            "response": response_data,
            "success": success,
            "duration_ms": getattr(self, '_last_api_duration', 0)
        })
    
    def get_completion_percentage(self) -> float:
        """Calculate conversation completion percentage"""
        required_fields = ['name', 'phone', 'reason', 'date_of_birth', 'state', 'provider_name', 'preferred_date', 'preferred_time']
        completed = sum(1 for field in required_fields if self.data.get(field))
        return (completed / len(required_fields)) * 100
    
    def save(self) -> str:
        """Enhanced session save with triage data"""
        try:
            os.makedirs("sessions", exist_ok=True)
            
            end_time = self.end_time or datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            session_data = {
                "metadata": {
                    "session_id": self.session_id,
                    "start_time": self.start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": duration,
                    "duration_minutes": round(duration / 60, 2),
                    "completion_percentage": self.get_completion_percentage(),
                    "total_turns": len(self.conversation),
                    "api_calls_made": len(self.api_calls),
                    "triage_required": self.triage_required,
                    "triage_complete": self.triage_complete,
                    "current_flow": self.current_flow
                },
                "patient_data": self.data,
                "conversation_history": self.conversation,
                "api_call_log": self.api_calls,
                "triage_data": self.triage_data,
                "raw_responses": {
                    "discovery_raw": self.data.get('discovery_raw', ''),
                    "eligibility_raw": self.data.get('eligibility_raw', '')
                }
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sessions/enhanced_session_{timestamp}_{self.session_id[:8]}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 ENHANCED SESSION SAVED: {os.path.abspath(filename)}")
            print(f"📊 Completion: {self.get_completion_percentage():.1f}% | Triage: {'✅' if self.triage_complete else '❌'}")
            return filename
        except Exception as e:
            logger.error(f"Session save failed: {e}")
            return ""

class Audio:
    """Production-grade audio system (same as original)"""
    
    def __init__(self):
        print("🎤 Initializing advanced audio system...")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize audio systems
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        self.tts = pyttsx3.init()
        self._configure_tts()
        self._calibrate_microphone()
        
        print("✅ Audio system ready with intelligent processing")
    
    def _configure_tts(self):
        """Configure TTS for optimal healthcare communication"""
        try:
            self.tts.setProperty('rate', 165)
            voices = self.tts.getProperty('voices')
            if voices:
                for voice in voices:
                    if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                        self.tts.setProperty('voice', voice.id)
                        break
        except Exception as e:
            logger.warning(f"TTS configuration issue: {e}")
    
    def _calibrate_microphone(self):
        """Advanced microphone calibration with environment adaptation"""
        try:
            with self.microphone as source:
                print("🎤 CALIBRATING: Analyzing ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2.0)
                
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8
                self.recognizer.phrase_threshold = 0.3
                self.recognizer.non_speaking_duration = 0.8
                
                print(f"✅ CALIBRATED: Energy threshold {self.recognizer.energy_threshold}")
        except Exception as e:
            logger.error(f"Microphone calibration failed: {e}")
    
    async def listen(self) -> str:
        """Intelligent speech recognition with context awareness"""
        def _listen_with_intelligence():
            try:
                with self.microphone as source:
                    print("🎧 LISTENING: Ready for speech...")
                    audio = self.recognizer.listen(
                        source, 
                        timeout=15,
                        phrase_time_limit=10
                    )
                
                print("🧠 PROCESSING: Analyzing speech with AI...")
                
                try:
                    result = self.recognizer.recognize_google(
                        audio, 
                        language='en-US',
                        show_all=False
                    )
                    print(f"🎯 RECOGNIZED: '{result}'")
                    return self._intelligent_post_process(result)
                    
                except sr.UnknownValueError:
                    print("🤔 UNCLEAR: Attempting enhanced recognition...")
                    try:
                        alternatives = self.recognizer.recognize_google(
                            audio, 
                            language='en-US', 
                            show_all=True
                        )
                        if alternatives and 'alternative' in alternatives:
                            best_match = alternatives['alternative'][0]['transcript']
                            print(f"🎯 RECOGNIZED (secondary): '{best_match}'")
                            return self._intelligent_post_process(best_match)
                    except:
                        pass
                    return "UNCLEAR"
                    
                except sr.RequestError as e:
                    logger.error(f"Speech service error: {e}")
                    return "NETWORK_ERROR"
                    
            except sr.WaitTimeoutError:
                print("⏰ TIMEOUT: No speech detected")
                return "TIMEOUT"
            except Exception as e:
                logger.error(f"Listen error: {e}")
                return "ERROR"
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _listen_with_intelligence)
        return result
    
    def _intelligent_post_process(self, text: str) -> str:
        """AI-powered post-processing of speech recognition"""
        if not text:
            return ""
        
        cleaned = ' '.join(text.split()).strip()
        
        # Healthcare-specific corrections
        healthcare_corrections = {
            'hernia': 'California',
            'gloria': 'Florida',
            'taxes': 'Texas',
            'organ': 'Oregon',
            'pencil vania': 'Pennsylvania',
            'connect i cut': 'Connecticut'
        }
        
        for incorrect, correct in healthcare_corrections.items():
            cleaned = re.sub(r'\b' + re.escape(incorrect) + r'\b', correct, cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def convert_numbers_to_digits(self, text: str) -> str:
        """Intelligent number conversion preserving medical context"""
        protected_patterns = [
            r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)\b',
            r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b',
            r'\b\d+\s*(?:mg|ml|cc|units?)\b',
            r'\b\d+/\d+\b'
        ]
        
        protected_ranges = []
        for pattern in protected_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                protected_ranges.append((match.start(), match.end()))
        
        def replace_number(match):
            start, end = match.span()
            for p_start, p_end in protected_ranges:
                if start >= p_start and end <= p_end:
                    return match.group()
            
            number = match.group()
            digit_words = {
                '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
            }
            return ' '.join(digit_words.get(digit, digit) for digit in number)
        
        result = re.sub(r'\b\d+\b', replace_number, text)
        return result
    
    async def speak(self, text: str):
        """Intelligent TTS with healthcare optimization"""
        if not text:
            return
        
        print(f"🗣️  SPEAKING: {text}")
        
        speech_text = self.convert_numbers_to_digits(text)
        speech_text = self._optimize_for_speech(speech_text)
        
        def _speak_intelligently():
            try:
                tts = gTTS(text=speech_text, lang='en', slow=False, tld='com')
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                    tts.save(tmp.name)
                    pygame.mixer.music.load(tmp.name)
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(50)
                    
                    os.unlink(tmp.name)
                    
            except Exception as e:
                logger.warning(f"Primary TTS failed: {e}, using fallback")
                try:
                    self.tts.say(speech_text)
                    self.tts.runAndWait()
                except Exception as e2:
                    logger.error(f"All TTS methods failed: {e2}")
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _speak_intelligently)
    
    def _optimize_for_speech(self, text: str) -> str:
        """Optimize text for natural speech delivery"""
        cleaned = re.sub(r'[{}[\]"|<>\\]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        cleaned = re.sub(r'([.!?])\s*', r'\1 ', cleaned)
        cleaned = re.sub(r'(,)\s*', r'\1 ', cleaned)
        
        return cleaned
    
    def cleanup(self):
        """Cleanup audio resources"""
        try:
            pygame.mixer.quit()
            if hasattr(self.tts, 'stop'):
                self.tts.stop()
        except Exception as e:
            logger.warning(f"Audio cleanup issue: {e}")

class MCPInsuranceAPI:
    """MODIFIED: Production MCP client with updated payload format"""
    
    def __init__(self, mcp_url: str, api_key: str):
        self.mcp_url = mcp_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-INF-API-KEY": api_key
        }
        
        print(f"🔗 MCP CLIENT INITIALIZED: {mcp_url}")
        print(f"🔑 API Key: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '***'}")
    
    def _split_name(self, full_name: str) -> Dict[str, str]:
        """MODIFIED: Split full name into first and last names"""
        if not full_name:
            return {"first_name": "", "last_name": ""}
        
        # Clean and split the name
        cleaned_name = ' '.join(full_name.strip().split())
        name_parts = cleaned_name.split()
        
        if len(name_parts) == 1:
            # Only one name provided
            return {"first_name": name_parts[0], "last_name": ""}
        elif len(name_parts) == 2:
            # First and last name
            return {"first_name": name_parts[0], "last_name": name_parts[1]}
        else:
            # Multiple names - first name is first part, last name is everything else
            return {
                "first_name": name_parts[0], 
                "last_name": " ".join(name_parts[1:])
            }
    
    def _split_provider_name(self, provider_name: str) -> Dict[str, str]:
        """MODIFIED: Split provider name for eligibility API"""
        if not provider_name:
            return {"provider_first_name": "", "provider_last_name": ""}
        
        # Handle common provider name formats
        cleaned = provider_name.strip()
        
        # Remove common titles and suffixes
        titles_suffixes = ['Dr.', 'Dr', 'MD', 'M.D.', 'DO', 'D.O.', 'NP', 'PA', 'RN']
        for title in titles_suffixes:
            cleaned = re.sub(r'\b' + re.escape(title) + r'\b', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        cleaned = ' '.join(cleaned.split())
        
        # Split the name
        name_parts = cleaned.split()
        
        if len(name_parts) == 0:
            return {"provider_first_name": "", "provider_last_name": ""}
        elif len(name_parts) == 1:
            return {"provider_first_name": name_parts[0], "provider_last_name": ""}
        elif len(name_parts) == 2:
            return {"provider_first_name": name_parts[0], "provider_last_name": name_parts[1]}
        else:
            # Multiple names - first is first name, rest is last name
            return {
                "provider_first_name": name_parts[0],
                "provider_last_name": " ".join(name_parts[1:])
            }

    async def call_insurance_api(self, api_type: str, patient_data: Dict) -> Dict:
        """MODIFIED: Intelligent MCP API call with updated payload format"""
        start_time = datetime.now()
        
        print(f"\n🚀 MCP API CALL INITIATED")
        print(f"📋 Type: {api_type.upper()}")
        print(f"👤 Patient: {patient_data.get('name', 'Unknown')}")
        
        try:
            payload = self._construct_mcp_payload(api_type, patient_data)
            
            print(f"📤 MCP PAYLOAD:")
            print(json.dumps(payload, indent=2))
            
            def _execute_request():
                try:
                    response = requests.post(
                        self.mcp_url, 
                        headers=self.headers, 
                        json=payload, 
                        timeout=45
                    )
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.Timeout:
                    return {"error": "REQUEST_TIMEOUT", "message": "Insurance API call timed out"}
                except requests.exceptions.ConnectionError:
                    return {"error": "CONNECTION_ERROR", "message": "Could not connect to insurance API"}
                except requests.exceptions.HTTPError as e:
                    return {"error": "HTTP_ERROR", "message": f"HTTP {e.response.status_code}: {e.response.text}"}
                except Exception as e:
                    return {"error": "REQUEST_ERROR", "message": str(e)}
            
            loop = asyncio.get_event_loop()
            raw_response = await loop.run_in_executor(None, _execute_request)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            print(f"📥 MCP RESPONSE RECEIVED ({duration:.2f}s):")
            print(json.dumps(raw_response, indent=2))
            
            if "error" in raw_response:
                print(f"❌ MCP API ERROR: {raw_response['error']}")
                return {
                    "success": False,
                    "error": raw_response["error"],
                    "message": raw_response.get("message", "Unknown error"),
                    "duration_seconds": duration
                }
            
            if "result" in raw_response:
                result_data = raw_response["result"]
                print(f"✅ MCP API SUCCESS: Data received")
                
                processed_result = self._process_insurance_result(api_type, result_data)
                
                return {
                    "success": True,
                    "data": processed_result,
                    "raw_response": str(result_data),
                    "duration_seconds": duration
                }
            else:
                print(f"⚠️  MCP API WARNING: No result field in response")
                return {
                    "success": False,
                    "error": "NO_RESULT",
                    "message": "MCP response missing result field",
                    "duration_seconds": duration
                }
                
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"MCP API call failed: {e}")
            
            return {
                "success": False,
                "error": "EXCEPTION",
                "message": str(e),
                "duration_seconds": duration
            }
    
    def _construct_mcp_payload(self, api_type: str, patient_data: Dict) -> Dict:
        """MODIFIED: Construct MCP payload with new required format"""
        
        # Split patient name
        patient_names = self._split_name(patient_data.get('name', ''))
        
        # Format timestamp for ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if api_type == "discovery":
            # MODIFIED: Discovery request format
            payload = {
                "jsonrpc": "2.0",
                "id": f"discovery_{timestamp}",
                "method": "tools/call",
                "params": {
                    "name": "insurance_discovery",
                    "arguments": {
                        "patientDateOfBirth": str(patient_data.get('dob', '')),
                        "patientFirstName": patient_names["first_name"],
                        "patientLastName": patient_names["last_name"],
                        "patientState": str(patient_data.get('state', ''))
                    }
                }
            }
            
        elif api_type == "eligibility":
            # MODIFIED: Eligibility request format
            provider_names = self._split_provider_name(patient_data.get('provider_name', ''))
            
            payload = {
                "jsonrpc": "2.0",
                "id": f"eligibility_{timestamp}",
                "method": "tools/call",
                "params": {
                    "name": "benefits_eligibility",
                    "arguments": {
                        "patientFirstName": patient_names["first_name"],
                        "patientLastName": patient_names["last_name"],
                        "patientDateOfBirth": str(patient_data.get('dob', '')),
                        "subscriberId": str(patient_data.get('subscriber_id', patient_data.get('member_id', ''))),
                        "payerName": str(patient_data.get('payer_name', patient_data.get('payer', ''))),
                        "providerFirstName": provider_names["provider_first_name"],
                        "providerLastName": provider_names["provider_last_name"],
                        "providerNpi": "1234567890"  # Default value as specified
                    }
                }
            }
        else:
            # Fallback for unknown API type
            payload = {
                "jsonrpc": "2.0",
                "id": f"{api_type}_{timestamp}",
                "method": "tools/call",
                "params": {
                    "name": api_type,
                    "arguments": patient_data
                }
            }
        
        return payload
    
    def _process_insurance_result(self, api_type: str, result_data: Any) -> str:
        """MODIFIED: Intelligently process insurance API results"""
        return str(result_data)

class EnhancedLLM:
    """Enhanced LLM client with triage integration"""
    
    def __init__(self, jwt_token: str, endpoint_url: str, project_id: str, connection_id: str):
        self.jwt_token = jwt_token
        self.endpoint_url = endpoint_url
        self.project_id = project_id
        self.connection_id = connection_id
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {jwt_token}'
        }
        
        print(f"🧠 ENHANCED LLM INITIALIZED with A2A support")
    
    async def process_conversation(self, user_input: str, session: EnhancedSession, 
                                 api_results: Dict = None, triage_agent = None) -> Dict:
        """Enhanced conversation processing with triage integration"""
        
        print(f"\n🧠 ENHANCED LLM PROCESSING")
        print(f"👤 User Input: '{user_input}'")
        print(f"🔄 Current Flow: {session.current_flow}")
        print(f"🩺 Triage Required: {session.triage_required}")
        
        # Construct enhanced prompt based on current flow
        prompt = self._construct_enhanced_prompt(user_input, session, api_results, triage_agent)
        
        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]
            
            payload = {
                "messages": messages,
                "project_id": self.project_id,
                "connection_id": self.connection_id,
                "max_tokens": 600,
                "temperature": 0.2,
                "top_p": 0.9
            }
            
            def _execute_llm_request():
                try:
                    response = requests.post(
                        self.endpoint_url, 
                        headers=self.headers, 
                        json=payload, 
                        timeout=30
                    )
                    response.raise_for_status()
                    return response.json()
                except Exception as e:
                    logger.error(f"LLM request failed: {e}")
                    return {"error": str(e)}
            
            loop = asyncio.get_event_loop()
            raw_response = await loop.run_in_executor(None, _execute_llm_request)
            
            if "error" in raw_response:
                print(f"❌ LLM ERROR: {raw_response['error']}")
                return self._create_enhanced_fallback(session)
            
            # Extract LLM response
            if 'choices' in raw_response and raw_response['choices']:
                llm_content = raw_response['choices'][0]['message']['content']
            else:
                llm_content = raw_response.get('response', '')
            
            print(f"🧠 LLM RESPONSE: {llm_content}")
            
            if llm_content:
                parsed_response = self._parse_enhanced_response(llm_content)
                return parsed_response
            else:
                return self._create_enhanced_fallback(session)
                
        except Exception as e:
            logger.error(f"Enhanced LLM processing error: {e}")
            return self._create_enhanced_fallback(session)
    
    def _construct_enhanced_prompt(self, user_input: str, session: EnhancedSession, 
                                 api_results: Dict = None, triage_agent = None) -> str:
        """MODIFIED: Construct enhanced prompts with updated API trigger logic"""
        
        # API context
        api_context = ""
        if api_results and api_results.get("success"):
            api_context = f"""
IMPORTANT - API RESULTS JUST RECEIVED:
The {api_results.get('api_type', 'API')} call was successful. 
API Response Data: {api_results.get('data', '')}

YOU MUST:
1. Acknowledge the API success briefly
2. Extract and announce key information (payer name, member ID, copay, etc.)
3. Immediately ask the next logical question to continue the flow
4. Do NOT ask for information you already have in session data

COMBINE your acknowledgment + next question into ONE response.
"""</api_context>
    elif api_results and not api_results.get("success"):
        api_context = f"""
API CALL FAILED - Continue anyway:
{api_results.get('error', 'Unknown error')}

Acknowledge briefly that you couldn't access all insurance details but continue with scheduling.
"""
        
        # Triage context
        triage_context = ""
        if session.triage_complete:
            triage_context = f"""
TRIAGE COMPLETED - CLINICAL INFORMATION AVAILABLE:
- Urgency Level: {session.triage_data.get('urgency_level', 'unknown')}
- Recommended Doctor: {session.triage_data.get('doctor_type', 'unknown')}
- Clinical Notes: {session.triage_data.get('clinical_notes', '')}
- Symptoms: {', '.join(session.triage_data.get('symptoms', []))}
"""
        
        base_prompt = f"""You are an expert healthcare appointment scheduler with A2A triage integration capability.

CURRENT SESSION DATA: {json.dumps(session.data, indent=2)}
RECENT CONVERSATION: {json.dumps(session.conversation[-4:] if session.conversation else [], indent=2)}
CURRENT FLOW: {session.current_flow}
TRIAGE STATUS: {"Complete" if session.triage_complete else "Pending" if session.triage_required else "Not Required"}
USER INPUT: "{user_input}"{api_context}{triage_context}

ENHANCED CONVERSATION FLOW WITH TRIAGE INTEGRATION:

1. GREETING: Professional welcome, ask for full name
2. DATA COLLECTION: Phone number 
3. REASON ASSESSMENT: Ask reason for visit
   - IF MEDICAL CONCERN → Set triage_required=true, trigger A2A triage handoff
   - IF NON-MEDICAL → Continue standard flow
4. TRIAGE INTEGRATION (if medical):
   - Wait for triage completion via A2A protocol
   - Receive triage results and clinical recommendations
   - Announce triage findings and doctor recommendation
5. POST-TRIAGE STANDARD FLOW: 
   - Ask for date of birth (YYYY-MM-DD format) 
   - Ask for state
   - IMMEDIATELY trigger "discovery" API call when you have name+dob+state
   - Wait for discovery results, then extract subscriber_id and payer_name
   - Ask for provider name
   - IMMEDIATELY trigger "eligibility" API call when you have discovery results+provider
   - Announce insurance details (payer name, policy ID, co-pay)
6. APPOINTMENT SCHEDULING: 
    - Ask for Preferred date
    - Ask for Preferred time
    - IMMEDIATELY CONFIRM APPOINTMENT (NO AVAILABILITY CHECK REQUIRED)
    - Generate 5-digit alphanumeric confirmation code
    - Announce confirmation details
7. CONFIRMATION: Confirm all details and provide confirmation code
8. PROFESSIONAL CLOSING: Thank patient and end call

CRITICAL APPOINTMENT BOOKING RULE:
- DO NOT check doctor availability
- DO NOT mention availability or scheduling conflicts
- IMMEDIATELY BOOK the appointment when you have preferred date and time
- ALWAYS generate confirmation code once date/time collected
- Assume all requested appointments are available

MODIFIED API CALL TRIGGERS:
- DISCOVERY: After collecting full_name + date_of_birth + state → ALWAYS call "discovery" API
  - DOB must be in YYYY-MM-DD format
  - Name will be automatically split into first and last names
  - State should be 2-letter code or full state name
- ELIGIBILITY: After collecting provider_name AND having discovery results → ALWAYS call "eligibility" API
  - Provider name will be automatically split into first and last names
  - Uses subscriber_id from discovery response
  - Uses payer_name from discovery response
  - Default providerNpi = 1234567890

TRIAGE INTEGRATION RULES:
- When reason suggests medical concern, immediately set triage_required=true
- Do NOT proceed with DOB/state until triage is complete
- Announce triage results clearly: urgency level and doctor recommendation
- Use triage doctor recommendation for provider selection

INTELLIGENT BEHAVIOR:
- Check CURRENT SESSION DATA before asking questions
- Handle medical emergencies appropriately
- Be conversational and efficient
- Convert dates to YYYY-MM-DD format for APIs
- Generate 5-digit alphanumeric confirmation codes
- Extract subscriber_id and payer_name from discovery results for eligibility call

MEDICAL CONCERN DETECTION:
Symptoms, pain, illness, injury, medical issues, health problems, feeling sick, etc.

REQUIRED JSON RESPONSE FORMAT:
{{
    "response": "your professional response",
    "extract": {{"field_name": "extracted_value"}},
    "triage_required": true/false,
    "api_call": "discovery|eligibility|none",
    "api_data": {{relevant data}},
    "done": false,
    "flow_stage": "greeting|data_collection|reason|triage_wait|post_triage|standard_flow|complete"
}}"""
        
        return base_prompt
    
    def _parse_enhanced_response(self, response_text: str) -> Dict:
        """Parse enhanced LLM response with triage support"""
        try:
            content = response_text.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            parsed = json.loads(content)
            
            # Validate and enhance
            result = {
                "response": parsed.get("response", "").strip(),
                "extract": parsed.get("extract", {}),
                "triage_required": bool(parsed.get("triage_required", False)),
                "api_call": parsed.get("api_call", "none").lower(),
                "api_data": parsed.get("api_data", {}),
                "done": bool(parsed.get("done", False)),
                "flow_stage": parsed.get("flow_stage", "greeting")
            }
            
            if not result["response"]:
                result["response"] = "I understand. Please continue."
            
            if result["api_call"] not in ["discovery", "eligibility", "none"]:
                result["api_call"] = "none"
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ ENHANCED JSON PARSE ERROR: {e}")
            return self._create_enhanced_fallback(None)
        
        except Exception as e:
            print(f"❌ ENHANCED PARSE EXCEPTION: {e}")
            return self._create_enhanced_fallback(None)
    
    def _create_enhanced_fallback(self, session: EnhancedSession = None) -> Dict:
        """Create enhanced fallback response"""
        return {
            "response": "I understand. Could you please repeat that more clearly?",
            "extract": {},
            "triage_required": False,
            "api_call": "none",
            "api_data": {},
            "done": False,
            "flow_stage": "data_collection"
        }

class EnhancedHealthcareVoiceAgent:
    """Enhanced Healthcare Voice Agent with A2A Triage Integration"""
    
    def __init__(self, jwt_token: str, endpoint_url: str, project_id: str, connection_id: str,
                 mcp_url: str, insurance_api_key: str):
        
        print("🏥 ENHANCED HEALTHCARE VOICE AGENT - A2A TRIAGE INTEGRATION")
        print("=" * 70)
        
        self.session = EnhancedSession()
        self.audio = Audio()
        self.llm = EnhancedLLM(jwt_token, endpoint_url, project_id, connection_id)
        self.mcp_api = MCPInsuranceAPI(mcp_url, insurance_api_key)
        
        # Triage integration
        self.triage_agent = None
        self.triage_config = load_triage_config()
        
        print(f"✅ ENHANCED AGENT READY - Session ID: {self.session.session_id}")
        print("=" * 70)
    
    async def _initialize_triage_agent(self):
        """Initialize triage agent when needed"""
        if not self.triage_agent and self.triage_config:
            try:
                self.triage_agent = SymptomTriageAgent(
                    openai_url=self.triage_config['openai_url'],
                    openai_api_key=self.triage_config['openai_api_key'],
                    openai_project_id=self.triage_config.get('openai_project_id', ''),
                    openai_connection_id=self.triage_config.get('openai_connection_id', ''),
                    openai_connection_name=self.triage_config.get('openai_connection_name', ''),
                    openai_provider=self.triage_config.get('openai_provider', 'Azure-OpenAI'),
                    openai_model=self.triage_config.get('openai_model', 'gpt-4o'),
                    symptoms_csv=self.triage_config['symptoms_csv']
                )
                print("✅ TRIAGE AGENT INITIALIZED")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize triage agent: {e}")
                return False
        return self.triage_agent is not None
    
    async def _conduct_triage_session(self, patient_name: str, patient_phone: str, chief_complaint: str) -> Dict:
        """Conduct complete triage session via A2A protocol"""
        
        print(f"🩺 CONDUCTING A2A TRIAGE SESSION")
        print(f"👤 Patient: {patient_name}")
        print(f"📞 Phone: {patient_phone}")
        print(f"🗣️ Complaint: {chief_complaint}")
        
        # Initialize triage agent
        if not await self._initialize_triage_agent():
            return {"error": "Triage agent initialization failed"}
        
        try:
            # Start triage session
            triage_session = await self.triage_agent.start_triage_session(
                patient_name=patient_name,
                patient_phone=patient_phone,
                chief_complaint=chief_complaint
            )
            
            print(f"🆕 TRIAGE SESSION STARTED: {triage_session.session_id}")
            
            # Conduct triage conversation
            await self.audio.speak("I need to ask you some medical questions to properly assess your condition. Let me start with some basic questions.")
            
            # Process triage through multiple rounds
            while not triage_session.current_stage == "complete":
                # Listen for patient response
                user_input = await self.audio.listen()
                
                if user_input in ["TIMEOUT", "UNCLEAR", "NETWORK_ERROR", "ERROR"]:
                    response = await self._handle_speech_issue(user_input, 1)
                    if response:
                        await self.audio.speak(response)
                    continue
                
                if not user_input:
                    continue
                
                print(f"👤 PATIENT: {user_input}")
                
                # Process with triage agent
                triage_result = await self.triage_agent.process_user_input(user_input)
                
                # Handle emergency situations
                if triage_result.get("emergency_alert"):
                    emergency_msg = triage_result.get("response", "This appears to be an emergency. Please call 911 immediately.")
                    print(f"🚨 EMERGENCY: {emergency_msg}")
                    await self.audio.speak(emergency_msg)
                    return {"emergency": True, "end_call": True}
                
                # Speak triage response
                if triage_result.get("response"):
                    print(f"🩺 TRIAGE: {triage_result['response']}")
                    await self.audio.speak(triage_result["response"])
                
                # Check if triage complete
                if triage_result.get("triage_complete"):
                    print("✅ TRIAGE ASSESSMENT COMPLETE")
                    
                    # Get A2A handoff data
                    handoff_data = A2AProtocol.create_triage_handoff(self.triage_agent)
                    
                    # Save triage session
                    self.triage_agent.save_triage_session()
                    
                    return {"success": True, "handoff_data": handoff_data}
            
            # Fallback if loop exits without completion
            handoff_data = A2AProtocol.create_triage_handoff(self.triage_agent)
            return {"success": True, "handoff_data": handoff_data}
            
        except Exception as e:
            logger.error(f"Triage session error: {e}")
            return {"error": str(e)}
    
    def _format_dob_for_api(self, dob_input: str) -> str:
        """MODIFIED: Intelligent DOB formatting for API calls"""
        if not dob_input:
            return ""
        
        dob_str = str(dob_input).strip()
        print(f"🎂 FORMATTING DOB: '{dob_str}'")
        
        # MM/DD/YYYY format
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}', dob_str):
            parts = dob_str.split('/')
            month, day, year = parts[0], parts[1], parts[2]
            formatted = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            print(f"✅ DOB FORMATTED: {dob_str} → {formatted}")
            return formatted
        
        # YYYY-MM-DD format (already correct)
        if re.match(r'^\d{4}-\d{1,2}-\d{1,2}', dob_str):
            parts = dob_str.split('-')
            year, month, day = parts[0], parts[1], parts[2]
            formatted = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            print(f"✅ DOB FORMATTED: {dob_str} → {formatted}")
            return formatted
        
        # Natural language formats
        try:
            cleaned = re.sub(r'[,\s]+', ' ', dob_str).strip()
            parsed = datetime.strptime(cleaned, '%B %d %Y')
            formatted = parsed.strftime('%Y-%m-%d')
            print(f"✅ DOB FORMATTED: {dob_str} → {formatted}")
            return formatted
        except ValueError:
            pass
        
        try:
            cleaned = re.sub(r'[,\s]+', ' ', dob_str).strip()
            parsed = datetime.strptime(cleaned, '%b %d %Y')
            formatted = parsed.strftime('%Y-%m-%d')
            print(f"✅ DOB FORMATTED: {dob_str} → {formatted}")
            return formatted
        except ValueError:
            pass
        
        print(f"⚠️  DOB FORMAT UNCLEAR: Using as-is: {dob_str}")
        return dob_str
    
    async def start_enhanced_conversation(self):
        """Start enhanced conversation with A2A triage integration"""
        try:
            # Enhanced greeting
            greeting = "Hello! I'm your healthcare appointment assistant. I'll help you schedule your appointment today. To get started, could you please tell me your full name?"
            
            print(f"\n🏥 AGENT: {greeting}")
            await self.audio.speak(greeting)
            self.session.add_message("assistant", greeting)
            
            # Enhanced conversation loop
            turn_number = 0
            consecutive_errors = 0
            max_consecutive_errors = 3
            
            while consecutive_errors < max_consecutive_errors:
                try:
                    turn_number += 1
                    print(f"\n{'='*20} TURN {turn_number} {'='*20}")
                    
                    # Intelligent listening
                    user_input = await self.audio.listen()
                    
                    # Handle speech issues
                    if user_input in ["TIMEOUT", "UNCLEAR", "NETWORK_ERROR", "ERROR"]:
                        consecutive_errors += 1
                        response = await self._handle_speech_issue(user_input, consecutive_errors)
                        if response:
                            await self.audio.speak(response)
                        continue
                    
                    if not user_input:
                        consecutive_errors += 1
                        continue
                    
                    consecutive_errors = 0
                    print(f"👤 USER: {user_input}")
                    self.session.add_message("user", user_input)
                    
                    # Check for conversation end
                    if self._is_conversation_ending(user_input):
                        await self._end_conversation("User requested to end")
                        break
                    
                    # Process with enhanced LLM
                    llm_result = await self.llm.process_conversation(
                        user_input, self.session, triage_agent=self.triage_agent
                    )
                    
                    # Update session data
                    if llm_result.get("extract"):
                        old_data = dict(self.session.data)
                        self.session.data.update(llm_result["extract"])
                        new_fields = [k for k in llm_result["extract"].keys() 
                                    if k not in old_data or old_data[k] != llm_result["extract"][k]]
                        if new_fields:
                            print(f"📝 DATA UPDATED: {new_fields}")
                    
                    # Handle triage requirement
                    if llm_result.get("triage_required") and not self.session.triage_complete:
                        print("🩺 TRIAGE REQUIRED - INITIATING A2A HANDOFF")
                        self.session.triage_required = True
                        self.session.current_flow = "triage"
                        
                        # Announce triage transition
                        transition_msg = "I understand you have a medical concern. Let me conduct a quick medical assessment to determine the appropriate care level."
                        await self.audio.speak(transition_msg)
                        self.session.add_message("assistant", transition_msg)
                        
                        # Conduct triage session
                        triage_result = await self._conduct_triage_session(
                            patient_name=self.session.data.get('name', ''),
                            patient_phone=self.session.data.get('phone', ''),
                            chief_complaint=self.session.data.get('reason', user_input)
                        )
                        
                        # Handle triage results
                        if triage_result.get("emergency"):
                            await self._end_conversation("Emergency - referred to 911")
                            break
                        elif triage_result.get("success"):
                            # Process A2A handoff
                            handoff_data = triage_result["handoff_data"]
                            self.session.add_triage_data(handoff_data)
                            self.session.current_flow = "post_triage"
                            
                            # Announce triage results
                            urgency = handoff_data.get('urgency_level', 'unknown')
                            doctor_type = handoff_data.get('doctor_type', 'general practitioner')
                            
                            triage_summary = f"Based on my assessment, your condition appears to be {urgency} priority. I recommend seeing a {doctor_type}. Now let me help you schedule this appointment."
                            
                            print(f"🩺 TRIAGE SUMMARY: {triage_summary}")
                            await self.audio.speak(triage_summary)
                            self.session.add_message("assistant", triage_summary)
                        else:
                            # Triage failed, continue with standard flow
                            fallback_msg = "I'll help you schedule your appointment. Let me continue with the appointment details."
                            await self.audio.speak(fallback_msg)
                            self.session.add_message("assistant", fallback_msg)
                    
                    # Handle API calls
                    api_results = None
                    if llm_result.get("api_call") != "none":
                        print(f"Executing API CALL: {llm_result[api_call']}")
                        
                        api_results = await self._execute_intelligent_api_call(
                            llm_result["api_call"], 
                            llm_result.get("api_data", {})
                        )
                        
                        # Process API results with LLM
                        if api_results and api_results.get("success"):
                            api_announcement = await self.llm.process_conversation(
                                "API_RESULTS_RECEIVED", 
                                self.session, 
                                api_results
                            )
                            
                            if api_announcement.get("response"):
                                api_response = api_announcement["response"]
                                print(f"🏥 AGENT (API Results): {api_response}")
                                await self.audio.speak(api_response)
                                self.session.add_message("assistant", api_response)
                    
                    # Deliver main response
                    main_response = llm_result.get("response", "")
                    if main_response:
                        print(f"🏥 AGENT: {main_response}")
                        await self.audio.speak(main_response)
                        self.session.add_message("assistant", main_response)
                    
                    # Check for conversation completion
                    if llm_result.get("done", False):
                        print("✅ CONVERSATION COMPLETED BY LLM")
                        await self._end_conversation("Completed successfully")
                        break
                    
                    # Brief pause for natural flow
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Turn {turn_number} error: {e}")
                    error_response = "I apologize, I had a technical issue. Could you please repeat that?"
                    print(f"🏥 AGENT (Error): {error_response}")
                    await self.audio.speak(error_response)
            
            # Handle max errors
            if consecutive_errors >= max_consecutive_errors:
                await self._end_conversation("Too many consecutive errors")
                
        except KeyboardInterrupt:
            await self._end_conversation("Manually interrupted")
        except Exception as e:
            logger.error(f"Enhanced conversation error: {e}")
            await self._end_conversation(f"System error: {str(e)}")
        finally:
            self._cleanup_resources()
    
    async def _handle_speech_issue(self, issue_type: str, consecutive_count: int) -> str:
        """Handle speech recognition issues"""
        responses = {
            "TIMEOUT": ["I didn't hear anything. Please try speaking again."],
            "UNCLEAR": ["I couldn't understand that clearly. Could you please speak more slowly?"],
            "NETWORK_ERROR": ["I'm having a connection issue. Please try again in a moment."],
            "ERROR": ["I had a technical issue. Could you please try again?"]
        }
        
        response_list = responses.get(issue_type, responses["ERROR"])
        
        if consecutive_count >= 2:
            return f"{random.choice(response_list)} If you continue to have issues, you may want to call back later."
        
        return random.choice(response_list)
    
    def _is_conversation_ending(self, user_input: str) -> bool:
        """Detect conversation ending signals"""
        ending_phrases = [
            'bye', 'goodbye', 'hang up', 'end call', 'thank you goodbye',
            'that\'s all', 'we\'re done', 'finished', 'end', 'quit',
            'cancel', 'nevermind', 'never mind'
        ]
        
        user_lower = user_input.lower().strip()
        return any(phrase in user_lower for phrase in ending_phrases)
    
    async def _execute_intelligent_api_call(self, api_type: str, api_data: Dict) -> Dict:
        """MODIFIED: Execute intelligent API call with comprehensive data"""
        
        print(f"🚀 EXECUTING INTELLIGENT API CALL: {api_type.upper()}")
        
        # Prepare comprehensive call data
        call_data = {**self.session.data, **api_data}
        
        # MODIFIED: Intelligent DOB formatting
        if 'date_of_birth' in call_data:
            call_data['dob'] = self._format_dob_for_api(call_data['date_of_birth'])
        
        # MODIFIED: Extract data from discovery results for eligibility call
        if api_type == "eligibility":
            # Extract subscriber_id and payer_name from discovery results
            discovery_result = self.session.data.get('discovery_result', {})
            if discovery_result.get('success'):
                # Try to extract from discovery response
                discovery_data = discovery_result.get('data', '')
                discovery_raw = discovery_result.get('raw_response', '')
                
                # Extract subscriber_id (member_id/policy_id)
                if not call_data.get('subscriber_id') and not call_data.get('member_id'):
                    for response_text in [discovery_data, discovery_raw]:
                        if response_text:
                            # Look for member ID patterns
                            id_patterns = [
                                r'member\s*id[:\s]*([a-za-z0-9\-]+)',
                                r'policy\s*number[:\s]*([a-za-z0-9\-]+)',
                                r'policy\s*id[:\s]*([a-za-z0-9\-]+)',
                                r'subscriber\s*id[:\s]*([a-za-z0-9\-]+)'
                            ]
                            
                            for pattern in id_patterns:
                                match = re.search(pattern, str(response_text).lower())
                                if match:
                                    call_data['subscriber_id'] = match.group(1).strip().upper()
                                    print(f"🆔 EXTRACTED SUBSCRIBER ID: {call_data['subscriber_id']}")
                                    break
                            if call_data.get('subscriber_id'):
                                break
                
                # Extract payer_name
                if not call_data.get('payer_name') and not call_data.get('payer'):
                    for response_text in [discovery_data, discovery_raw]:
                        if response_text:
                            # Look for payer patterns
                            payer_patterns = [
                                r'payer[:\s]*([^\n,;]+)',
                                r'insurance[:\s]*([^\n,;]+)',
                                r'carrier[:\s]*([^\n,;]+)',
                                r'plan[:\s]*([^\n,;]+)'
                            ]
                            
                            for pattern in payer_patterns:
                                match = re.search(pattern, str(response_text).lower())
                                if match:
                                    call_data['payer_name'] = match.group(1).strip().title()
                                    print(f"🏢 EXTRACTED PAYER NAME: {call_data['payer_name']}")
                                    break
                            if call_data.get('payer_name'):
                                break
        
        # Debug the call data
        print(f"📋 API CALL DATA FOR {api_type.upper()}:")
        for key, value in call_data.items():
            if 'password' not in key.lower() and 'secret' not in key.lower():
                print(f"   {key}: {value}")
        
        # Execute the API call
        start_time = datetime.now()
        result = await self.mcp_api.call_insurance_api(api_type, call_data)
        
        # Store comprehensive results
        self.session.data[f"{api_type}_result"] = result
        self.session.add_api_call(api_type, call_data, result, result.get("success", False))
        
        # Store raw response for debugging
        if result.get("success"):
            self.session.data[f"{api_type}_raw"] = result.get("raw_response", "")
            
            # MODIFIED: Intelligent data extraction from API response
            self._extract_api_intelligence(api_type, result.get("data", ""))
        else:
            print(f"❌ API CALL FAILED: {result.get('error', 'Unknown error')}")
            print(f"📝 Error Message: {result.get('message', 'No error message')}")
        
        duration = (datetime.now() - start_time).total_seconds()
        status = "✅ SUCCESS" if result.get("success") else "❌ FAILED"
        print(f"{status} - {api_type.upper()} API completed in {duration:.2f}s")
        
        return result
    
    def _extract_api_intelligence(self, api_type: str, api_response: str):
        """MODIFIED: Intelligently extract structured data from API responses"""
        
        print(f"🧠 EXTRACTING INTELLIGENCE FROM {api_type.upper()} RESPONSE")
        
        if not api_response:
            return
        
        response_lower = api_response.lower()
        
        if api_type == "discovery":
            # Extract payer information
            payer_patterns = [
                r'payer[:\s]*([^\n,;]+)',
                r'insurance[:\s]*([^\n,;]+)',
                r'carrier[:\s]*([^\n,;]+)',
                r'plan[:\s]*([^\n,;]+)'
            ]
            
            for pattern in payer_patterns:
                match = re.search(pattern, response_lower)
                if match and not self.session.data.get('payer'):
                    payer = match.group(1).strip().title()
                    self.session.data['payer'] = payer
                    self.session.data['payer_name'] = payer  # Store both for eligibility API
                    print(f"🏢 EXTRACTED PAYER: {payer}")
                    break
            
            # Extract member/policy ID
            id_patterns = [
                r'member\s*id[:\s]*([a-za-z0-9\-]+)',
                r'policy\s*number[:\s]*([a-za-z0-9\-]+)',
                r'policy\s*id[:\s]*([a-za-z0-9\-]+)',
                r'subscriber\s*id[:\s]*([a-za-z0-9\-]+)',
                r'id[:\s]*([a-za-z0-9\-]{5,})'  # At least 5 chars for ID
            ]
            
            for pattern in id_patterns:
                match = re.search(pattern, response_lower)
                if match and not self.session.data.get('member_id'):
                    member_id = match.group(1).strip().upper()
                    self.session.data['member_id'] = member_id
                    self.session.data['subscriber_id'] = member_id  # Store both for eligibility API
                    print(f"🆔 EXTRACTED MEMBER ID: {member_id}")
                    break
        
        elif api_type == "eligibility":
            # Extract financial information
            financial_patterns = {
                'copay': r'co-?pay[:\s]*\$?([0-9,]+\.?[0-9]*)',
                'deductible': r'deductible[:\s]*\$?([0-9,]+\.?[0-9]*)',
                'coinsurance': r'coinsurance[:\s]*([0-9]+)%?'
            }
            
            for key, pattern in financial_patterns.items():
                match = re.search(pattern, response_lower)
                if match:
                    value = match.group(1).strip()
                    self.session.data[f'insurance_{key}'] = value
                    print(f"💰 EXTRACTED {key.upper()}: ${value}")
    
    async def _end_conversation(self, reason: str):
        """End conversation with comprehensive summary"""
        print(f"\n🏁 ENDING ENHANCED CONVERSATION: {reason}")
        
        self.session.end_time = datetime.now()
        
        # Generate confirmation if appropriate
        if self._should_generate_confirmation():
            if not self.session.data.get("confirmation_code"):
                confirmation = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                self.session.data["confirmation_code"] = confirmation
                
                patient_name = self.session.data.get('name', 'Patient')
                preferred_date = self.session.data.get('preferred_date', 'your requested date')
                
                end_message = f"Perfect, {patient_name}! Your appointment is confirmed for {preferred_date}. Your confirmation number is {confirmation}. Please keep this number for your records. Thank you for calling!"
            else:
                end_message = "Thank you for calling. Have a great day!"
        else:
            completion = self.session.get_completion_percentage()
            if completion > 50:
                end_message = "Thank you for the information you provided. Please call back when you're ready to complete your appointment scheduling."
            else:
                end_message = "Thank you for calling. Please call back when you have all the information needed to schedule your appointment."
        
        print(f"🏥 AGENT (Final): {end_message}")
        await self.audio.speak(end_message)
        self.session.add_message("assistant", end_message)
        
        # Save enhanced session
        filename = self.session.save()
        self._display_enhanced_summary(reason, filename)
    
    def _should_generate_confirmation(self) -> bool:
        """Determine if confirmation should be generated"""
        required_for_confirmation = ['name', 'preferred_date']
        return all(self.session.data.get(field) for field in required_for_confirmation)
    
    def _display_enhanced_summary(self, reason: str, filename: str):
        """Display enhanced conversation summary"""
        duration = (self.session.end_time - self.session.start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"🏥 ENHANCED HEALTHCARE CONVERSATION SUMMARY")
        print(f"{'='*70}")
        print(f"📋 Session ID: {self.session.session_id}")
        print(f"⏱️  Duration: {duration/60:.2f} minutes")
        print(f"🎯 Completion: {self.session.get_completion_percentage():.1f}%")
        print(f"💬 Total turns: {len(self.session.conversation)}")
        print(f"🔗 API calls: {len(self.session.api_calls)}")
        print(f"🩺 Triage: {'✅ Complete' if self.session.triage_complete else '❌ Not performed'}")
        print(f"🔄 Flow: {self.session.current_flow}")
        print(f"🏁 End reason: {reason}")
        print(f"📁 Saved to: {filename}")
        
        # Triage summary
        if self.session.triage_complete:
            print(f"\n🩺 TRIAGE SUMMARY:")
            triage = self.session.triage_data
            print(f"   Urgency: {triage.get('urgency_level', 'unknown')}")
            print(f"   Doctor Type: {triage.get('doctor_type', 'unknown')}")
            print(f"   Symptoms: {', '.join(triage.get('symptoms', []))}")
        
        print(f"{'='*70}")
    
    def _cleanup_resources(self):
        """Cleanup all resources"""
        try:
            print("🧹 CLEANING UP ENHANCED RESOURCES...")
            self.audio.cleanup()
            print("✅ Enhanced cleanup completed")
        except Exception as e:
            logger.warning(f"Enhanced cleanup issue: {e}")

def load_enhanced_config():
    """Load enhanced configuration with triage support"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("🔧 Loading enhanced environment configuration...")
    except ImportError:
        print("⚠️  python-dotenv not available, reading from environment...")
    
    config = {}
    required_configs = {
        'jwt_token': 'JWT_TOKEN',
        'endpoint_url': 'ENDPOINT_URL', 
        'project_id': 'PROJECT_ID',
        'connection_id': 'CONNECTION_ID',
        'mcp_url': 'MCP_URL',
        'insurance_api_key': 'X_INF_API_KEY'
    }
    
    print("🔍 VALIDATING ENHANCED CONFIGURATION:")
    missing_configs = []
    
    for key, env_var in required_configs.items():
        value = os.getenv(env_var)
        if value and value.strip():
            config[key] = value.strip()
            display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            print(f"   ✅ {env_var}: {display_value}")
        else:
            missing_configs.append(env_var)
            print(f"   ❌ {env_var}: MISSING")
    
    if missing_configs:
        print(f"\n❌ CONFIGURATION ERROR: Missing required environment variables:")
        for var in missing_configs:
            print(f"   - {var}")
        print(f"\nPlease set these environment variables and try again.")
        return None
    
    print("✅ ENHANCED CONFIGURATION VALIDATED")
    
    # Check triage configuration
    triage_config = load_triage_config()
    if triage_config:
        print("✅ TRIAGE INTEGRATION AVAILABLE")
        config['triage_available'] = True
    else:
        print("⚠️  TRIAGE INTEGRATION UNAVAILABLE - Missing OpenAI configuration")
        config['triage_available'] = False
    
    return config

async def main():
    """Enhanced Healthcare Voice Agent Main Entry Point with A2A Triage"""
    print("🏥 ENHANCED HEALTHCARE VOICE AGENT - A2A TRIAGE INTEGRATION")
    print("🤖 Maximum AI Intelligence | MCP Integration | Symptom Triage | Production Ready")
    print("=" * 80)
    
    # Load and validate configuration
    config = load_enhanced_config()
    if not config:
        print("❌ Cannot start without valid configuration")
        return
    
    try:
        # Initialize enhanced agent
        agent = EnhancedHealthcareVoiceAgent(
            jwt_token=config['jwt_token'],
            endpoint_url=config['endpoint_url'],
            project_id=config['project_id'],
            connection_id=config['connection_id'],
            mcp_url=config['mcp_url'],
            insurance_api_key=config['insurance_api_key']
        )
        
        # Display capabilities
        print(f"\n🎯 AGENT CAPABILITIES:")
        print(f"   🗣️  Voice Recognition & TTS")
        print(f"   🔗 MCP Insurance Integration")
        print(f"   🩺 A2A Symptom Triage: {'✅ Enabled' if config['triage_available'] else '❌ Disabled'}")
        print(f"   🧠 GPT-4o Medical Intelligence")
        print(f"   📋 Complete Appointment Workflow")
        
        # Start enhanced conversation
        await agent.start_enhanced_conversation()
        
    except KeyboardInterrupt:
        print("\n👋 GRACEFUL SHUTDOWN: User interrupted")
    except Exception as e:
        logger.error(f"CRITICAL ERROR: {e}")
        print(f"💥 SYSTEM ERROR: {e}")
    finally:
        print("🏥 Enhanced Healthcare Voice Agent session ended")

# Standalone test function for A2A integration
async def test_a2a_integration():
    """Test A2A integration between supervisor and triage agents"""
    
    print("🧪 TESTING A2A TRIAGE INTEGRATION")
    print("=" * 50)
    
    # Test triage configuration
    triage_config = load_triage_config()
    if not triage_config:
        print("❌ Cannot test A2A integration without triage configuration")
        return
    
    # Initialize triage agent with full Azure OpenAI parameters
    triage_agent = SymptomTriageAgent(
        openai_url=triage_config['openai_url'],
        openai_api_key=triage_config['openai_api_key'],
        openai_project_id=triage_config.get('openai_project_id', ''),
        openai_connection_id=triage_config.get('openai_connection_id', ''),
        openai_connection_name=triage_config.get('openai_connection_name', ''),
        openai_provider=triage_config.get('openai_provider', 'Azure-OpenAI'),
        openai_model=triage_config.get('openai_model', 'gpt-4o'),
        symptoms_csv=triage_config['symptoms_csv']
    )
    
    # Test A2A handoff process
    print("\n🔗 TESTING A2A PROTOCOL...")
    
    # Simulate supervisor request
    supervisor_request = A2AProtocol.process_supervisor_request(
        patient_name="Jane Test",
        patient_phone="555-0456", 
        chief_complaint="I have severe chest pain"
    )
    
    print(f"📤 SUPERVISOR REQUEST:")
    print(json.dumps(supervisor_request, indent=2))
    
    # Start triage session
    session = await triage_agent.start_triage_session(
        patient_name=supervisor_request["patient_name"],
        patient_phone=supervisor_request["patient_phone"],
        chief_complaint=supervisor_request["chief_complaint"]
    )
    
    # Simulate triage conversation
    test_responses = [
        "I have severe chest pain and shortness of breath",
        "About 30 minutes ago", 
        "9 out of 10",
        "It's crushing and radiating to my left arm",
        "Yes, I'm sweating and feel nauseous",
        "No recent injuries but I have high blood pressure"
    ]
    
    print(f"\n🩺 CONDUCTING TRIAGE SIMULATION...")
    for response in test_responses:
        print(f"👤 PATIENT: {response}")
        result = await triage_agent.process_user_input(response)
        print(f"🩺 TRIAGE: {result.get('response', '')}")
        
        if result.get('emergency_alert'):
            print(f"🚨 EMERGENCY DETECTED!")
            break
            
        if result.get('triage_complete'):
            print(f"✅ TRIAGE COMPLETE")
            break
    
    # Generate A2A handoff
    handoff_data = A2AProtocol.create_triage_handoff(triage_agent)
    
    print(f"\n🔗 A2A HANDOFF DATA:")
    print(json.dumps(handoff_data, indent=2))
    
    # Save test session
    filename = triage_agent.save_triage_session()
    print(f"💾 Test triage session saved: {filename}")
    
    print(f"\n✅ A2A INTEGRATION TEST COMPLETE")
    print(f"🎯 Urgency Level: {handoff_data.get('urgency_level')}")
    print(f"🩺 Doctor Recommendation: {handoff_data.get('doctor_type')}")

# Test function for MCP API payload formats
async def test_mcp_api_formats():
    """Test the new MCP API payload formats"""
    
    print("🧪 TESTING MCP API PAYLOAD FORMATS")
    print("=" * 50)
    
    # Create test MCP API instance
    mcp_api = MCPInsuranceAPI("http://test-url", "test-key")
    
    # Test patient data
    test_patient_data = {
        'name': 'John Michael Doe',
        'dob': '1985-03-15',
        'state': 'CA',
        'provider_name': 'Dr. Sarah Johnson MD',
        'subscriber_id': 'ABC123456',
        'payer_name': 'Blue Cross Blue Shield'
    }
    
    print("\n🔍 TESTING NAME SPLITTING:")
    patient_names = mcp_api._split_name(test_patient_data['name'])
    print(f"   Full Name: {test_patient_data['name']}")
    print(f"   First Name: {patient_names['first_name']}")
    print(f"   Last Name: {patient_names['last_name']}")
    
    provider_names = mcp_api._split_provider_name(test_patient_data['provider_name'])
    print(f"   Provider Name: {test_patient_data['provider_name']}")
    print(f"   Provider First: {provider_names['provider_first_name']}")
    print(f"   Provider Last: {provider_names['provider_last_name']}")
    
    print("\n📤 TESTING DISCOVERY PAYLOAD:")
    discovery_payload = mcp_api._construct_mcp_payload("discovery", test_patient_data)
    print(json.dumps(discovery_payload, indent=2))
    
    print("\n📤 TESTING ELIGIBILITY PAYLOAD:")
    eligibility_payload = mcp_api._construct_mcp_payload("eligibility", test_patient_data)
    print(json.dumps(eligibility_payload, indent=2))
    
    print("\n✅ MCP API FORMAT TESTS COMPLETE")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            # Run A2A integration test
            asyncio.run(test_a2a_integration())
        elif sys.argv[1] == "test-mcp":
            # Run MCP API format test
            asyncio.run(test_mcp_api_formats())
        else:
            print("Available test options:")
            print("  python script.py test       - Test A2A integration")
            print("  python script.py test-mcp   - Test MCP API formats")
            print("  python script.py           - Run main agent")
    else:
        # Run main enhanced agent
        asyncio.run(main())
