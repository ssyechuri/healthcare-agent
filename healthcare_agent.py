"""
Healthcare Voice Agent v1.0.0
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

# Core dependencies
import requests
import speech_recognition as sr
from gtts import gTTS
import pygame
import pyttsx3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Session:
    """Simple session container"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # All data in one simple dict
    data: Dict[str, Any] = field(default_factory=dict)
    conversation: List[Dict] = field(default_factory=list)
    
    def add_message(self, role: str, message: str):
        self.conversation.append({
            "time": datetime.now().isoformat(),
            "role": role,
            "message": message
        })
    
    def save(self):
        """Save session to JSON file (not encrypted)"""
        try:
            os.makedirs("sessions", exist_ok=True)
            
            end_time = self.end_time or datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            session_data = {
                "session_id": self.session_id,
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_minutes": round(duration / 60, 2),
                "patient_data": self.data,
                "conversation_history": self.conversation
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sessions/call_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            print(f"💾 Session saved: {os.path.abspath(filename)}")
            return filename
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return ""

class Audio:
    """Enhanced audio manager with better recognition and number digit conversion"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        pygame.mixer.init()
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 165)
        
        # Enhanced microphone setup
        try:
            with self.microphone as source:
                print("🎤 Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1.5)  # Longer calibration
                # Optimize recognition settings
                self.recognizer.energy_threshold = 300  # Adjust for sensitivity
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.8  # How long to wait for pause
                self.recognizer.phrase_threshold = 0.3  # Minimum audio length
                print("✅ Microphone calibrated")
        except Exception as e:
            print(f"⚠️ Microphone setup issue: {e}")
    
    async def listen(self) -> str:
        """Enhanced speech recognition with better accuracy"""
        def _listen():
            try:
                with self.microphone as source:
                    print("🎧 Listening...")
                    # Better audio capture settings
                    audio = self.recognizer.listen(
                        source, 
                        timeout=12,  # Slightly longer timeout
                        phrase_time_limit=8  # Allow longer phrases
                    )
                
                print("🔄 Processing speech...")
                # Try Google Speech Recognition with error handling
                try:
                    result = self.recognizer.recognize_google(audio, language='en-US')
                    print(f"🎯 Recognized: '{result}'")
                    return result
                except sr.UnknownValueError:
                    print("❓ Speech unclear, trying alternative...")
                    # Try with different settings if first attempt fails
                    try:
                        result = self.recognizer.recognize_google(audio, language='en-US', show_all=False)
                        print(f"🎯 Recognized (retry): '{result}'")
                        return result
                    except:
                        return "UNCLEAR"
                except sr.RequestError as e:
                    print(f"🌐 Network error: {e}")
                    return "NETWORK_ERROR"
                    
            except sr.WaitTimeoutError:
                print("⏰ Listening timeout")
                return "TIMEOUT"
            except Exception as e:
                print(f"❌ Listen error: {e}")
                return "ERROR"
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _listen)
        
        # Clean and validate result
        if result in ["TIMEOUT", "UNCLEAR", "NETWORK_ERROR", "ERROR"]:
            return result
        
        # Basic cleaning while preserving meaning
        if result:
            # Remove extra spaces but preserve the actual words
            cleaned = ' '.join(result.split())
            return cleaned.strip()
        
        return ""
    
    def convert_numbers_to_digits(self, text: str) -> str:
        """Convert numbers to digit-by-digit pronunciation except datetime"""
        # Don't convert time expressions like "10:30 AM" or "11 AM"
        time_pattern = r'\b\d{1,2}:\d{2}\s*(AM|PM|am|pm)\b|\b\d{1,2}\s*(AM|PM|am|pm)\b'
        
        # Find all time expressions to preserve
        time_matches = re.finditer(time_pattern, text)
        protected_ranges = [(m.start(), m.end()) for m in time_matches]
        
        # Convert other numbers to digits
        def replace_number(match):
            start, end = match.span()
            # Check if this number is in a protected time range
            for p_start, p_end in protected_ranges:
                if start >= p_start and end <= p_end:
                    return match.group()  # Don't convert
            
            number = match.group()
            # Convert to digit by digit (e.g., "123" -> "one two three")
            digit_words = {
                '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
            }
            return ' '.join(digit_words.get(digit, digit) for digit in number)
        
        # Find standalone numbers (not in time expressions)
        number_pattern = r'\b\d+\b'
        result = re.sub(number_pattern, replace_number, text)
        return result
    
    async def speak(self, text: str):
        """Speak text with number conversion"""
        if not text:
            return
        
        # Convert numbers to digits (except time)
        text_with_digits = self.convert_numbers_to_digits(text)
        
        # Clean text
        clean_text = re.sub(r'[{}[\]"|<>\\]', '', text_with_digits)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        def _speak():
            try:
                tts = gTTS(text=clean_text, lang='en', slow=False)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
                    tts.save(tmp.name)
                    pygame.mixer.music.load(tmp.name)
                    pygame.mixer.music.play()
                    
                    while pygame.mixer.music.get_busy():
                        pygame.time.wait(100)
                    
                    os.unlink(tmp.name)
            except:
                try:
                    self.tts.say(clean_text)
                    self.tts.runAndWait()
                except:
                    pass
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _speak)
    
    def cleanup(self):
        try:
            pygame.mixer.quit()
            self.tts.stop()
        except:
            pass

class API:
    """Simple API client"""
    
    def __init__(self, a2a_url: str, api_key: str):
        self.a2a_url = a2a_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-INF-API-KEY": api_key
        }
    
    async def call(self, api_type: str, data: Dict) -> Dict:
        """Make API call"""
        try:
            if api_type == "discovery":
                text = f"Patient insurance discovery for {data.get('name')} with date of birth {data.get('dob')} and Patient state is {data.get('state')}"
            else:  # eligibility
                text = f"Find eligibility for patient {data.get('name')}. Date of birth is {data.get('dob')} with member ID {data.get('member_id', 'Unknown')}. Payer name is {data.get('payer', 'Unknown')}. Provider name {data.get('provider_name')}. NPI {data.get('provider_npi')}."
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "message/send",
                "params": {
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": text}]
                    }
                }
            }
            
            def _request():
                try:
                    response = requests.post(self.a2a_url, headers=self.headers, json=payload, timeout=30)
                    response.raise_for_status()
                    result = response.json()
                    if 'result' in result:
                        return {"success": True, "data": str(result['result'])}
                    return {"success": False, "error": "No result"}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _request)
            
        except Exception as e:
            return {"success": False, "error": str(e)}

class LLM:
    """Simple LLM client with progress tracking"""
    
    def __init__(self, jwt_token: str, endpoint_url: str, project_id: str, connection_id: str):
        self.jwt_token = jwt_token
        self.endpoint_url = endpoint_url
        self.project_id = project_id
        self.connection_id = connection_id
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {jwt_token}'
        }
    
    async def process(self, user_input: str, session: Session, api_results: Dict = None) -> Dict:
        """Process with LLM"""
        
        # Show current progress
        self._show_progress(session)
        
        # Add API results context if available
        api_context = ""
        if api_results and api_results.get("success"):
            api_context = f"\n\nAPI RESULTS TO ANNOUNCE: Extract and announce ONLY these details from the API response:\n- Payer name (insurance company)\n- Policy ID number  \n- Co-pay amount\nAPI Response: {api_results.get('data', '')}"
        
        # Enhanced prompt for confident, intelligent conversation with speech recognition awareness
        prompt = f"""You are an experienced healthcare appointment scheduler. Be confident, efficient, and intelligent.

CURRENT DATA: {json.dumps(session.data)}
CONVERSATION: {json.dumps(session.conversation[-3:]) if session.conversation else []}
USER INPUT: "{user_input}"{api_context}

CONVERSATION FLOW (maintain context throughout):
1. GREETING: Warm, professional welcome
2. COLLECT: name → phone → reason → date_of_birth → state (smoothly, one at a time)
3. INSURANCE DISCOVERY: When you have name+dob+state, trigger discovery API
4. PROVIDER INFO: Get provider_name and provider_npi
5. ELIGIBILITY CHECK: When you have discovery+provider info, trigger eligibility API
6. ANNOUNCE RESULTS: Extract and announce ONLY: Policy ID, Payer name, Co-pay amount
7. APPOINTMENT: preferred_date → preferred_time → book with 5-digit code
8. END: Professional closing

SMART BEHAVIOR:
- NEVER ask for info you already have - check CURRENT DATA first
- NEVER repeat confirmations unnecessarily 
- Move forward confidently - don't second-guess yourself
- Be natural and conversational, not robotic
- Handle multiple pieces of info in one response when user provides them
- Stay focused on the flow - don't get sidetracked

SPEECH RECOGNITION AWARENESS:
- If user input seems like misheard speech, ask for clarification
- For dates, confirm in a different format (e.g., "November 15th" if you heard "4-16")
- For state names, confirm common misheard ones (California/hernia, Florida/Gloria, etc.)
- If something doesn't make sense in context, politely ask user to repeat

TECHNICAL:
- Numbers: digit by digit (one two three) except times (10 AM)
- Date formats: accept MM/DD/YYYY or YYYY-MM-DD, month names
- Generate 5-digit alphanumeric confirmation code when booking

RESPOND WITH JSON:
{{
    "response": "confident, natural response",
    "extract": {{"field": "value"}},
    "api_call": "discovery|eligibility|none", 
    "api_data": {{}},
    "done": false
}}

Set done=true when confirmation code generated and appointment booked."""

        try:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]
            
            payload = {
                "messages": messages,
                "project_id": self.project_id,
                "connection_id": self.connection_id,
                "max_tokens": 400,
                "temperature": 0.3
            }
            
            def _request():
                try:
                    response = requests.post(self.endpoint_url, headers=self.headers, json=payload, timeout=20)
                    response.raise_for_status()
                    result = response.json()
                    if 'choices' in result and result['choices']:
                        return result['choices'][0]['message']['content']
                    return result.get('response', '')
                except:
                    return ""
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _request)
            
            if response:
                return self._parse_response(response)
            else:
                return {"response": "Could you repeat that?", "extract": {}, "api_call": "none", "done": False}
                
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {"response": "I'm sorry, could you say that again?", "extract": {}, "api_call": "none", "done": False}
    
    def _show_progress(self, session: Session):
        """Show current progress on terminal"""
        required_fields = ['name', 'phone', 'reason', 'date_of_birth', 'state', 'provider_name', 'provider_npi', 'preferred_date', 'preferred_time']
        collected = [field for field in required_fields if session.data.get(field)]
        
        print(f"\n📊 PROGRESS: {len(collected)}/{len(required_fields)} fields collected")
        print(f"✅ Collected: {', '.join(collected) if collected else 'None yet'}")
        
        missing = [field for field in required_fields if not session.data.get(field)]
        if missing:
            print(f"⏳ Still need: {', '.join(missing)}")
        
        # Show API status
        if session.data.get('discovery_result'):
            print("🔍 Insurance discovery: ✅ Done")
        if session.data.get('eligibility_result'):
            print("✅ Eligibility check: ✅ Done")
        
        print("-" * 50)
    
    def _parse_response(self, response_text: str) -> Dict:
        """Parse LLM response"""
        try:
            # Clean response
            content = response_text.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            parsed = json.loads(content)
            
            return {
                "response": parsed.get("response", ""),
                "extract": parsed.get("extract", {}),
                "api_call": parsed.get("api_call", "none"),
                "api_data": parsed.get("api_data", {}),
                "done": parsed.get("done", False)
            }
            
        except:
            # Try to extract just the response
            response_match = re.search(r'"response":\s*"([^"]*)"', response_text)
            if response_match:
                return {
                    "response": response_match.group(1),
                    "extract": {},
                    "api_call": "none",
                    "done": False
                }
            
            return {
                "response": response_text if response_text else "I understand.",
                "extract": {},
                "api_call": "none", 
                "done": False
            }

class HealthcareAgent:
    """ healthcare agent - no complex states"""
    
    def __init__(self, jwt_token: str, endpoint_url: str, project_id: str, connection_id: str,
                 a2a_url: str, insurance_api_key: str):
        
        self.session = Session()
        self.audio = Audio()
        self.llm = LLM(jwt_token, endpoint_url, project_id, connection_id)
        self.api = API(a2a_url, insurance_api_key)
        
        print(f"🏥 Healthcare Agent v1.0.0 ready - Session: {self.session.session_id}")
    
    async def start(self):
        """Start conversation"""
        try:
            # greeting - LLM can handle this
            greeting = "Hello! I'm here to help schedule your medical appointment. Could you please tell me your full name?"
            
            print(f"🏥 Agent: {greeting}")
            await self.audio.speak(greeting)
            self.session.add_message("assistant", greeting)
            
            # Simple conversation loop - no arbitrary turn limit
            turn = 0
            while True:
                try:
                    turn += 1
                    print(f"\n🔄 Turn {turn}")
                    
                    # Listen
                    user_input = await self.audio.listen()
                    
                    # Enhanced handling of different speech recognition results
                    if user_input == "TIMEOUT":
                        print("⏰ No speech detected")
                        timeout_msg = "I didn't hear anything. Please try speaking again."
                        await self.audio.speak(timeout_msg)
                        continue
                    elif user_input == "UNCLEAR":
                        print("❓ Speech was unclear")
                        unclear_msg = "I couldn't understand that clearly. Could you please repeat more slowly?"
                        await self.audio.speak(unclear_msg)
                        continue
                    elif user_input == "NETWORK_ERROR":
                        print("🌐 Network issue with speech recognition")
                        network_msg = "I'm having trouble with my speech recognition. Please try again."
                        await self.audio.speak(network_msg)
                        continue
                    elif user_input == "ERROR":
                        print("❌ Speech recognition error")
                        error_msg = "Sorry, I had a technical issue. Please try again."
                        await self.audio.speak(error_msg)
                        continue
                    elif not user_input:
                        print("👤 User: [No input detected]")
                        continue
                    
                    print(f"👤 User: {user_input}")
                    self.session.add_message("user", user_input)
                    
                    # Check for exit
                    if any(word in user_input.lower() for word in ['bye', 'goodbye', 'hang up', 'end']):
                        await self._end_call("User requested to end")
                        break
                    
                    # Process with LLM
                    result = await self.llm.process(user_input, self.session)
                    
                    # Update data
                    if result.get("extract"):
                        self.session.data.update(result["extract"])
                        print(f"📝 Updated: {list(result['extract'].keys())}")
                    
                    # Make API call if requested
                    api_results = None
                    if result.get("api_call") != "none":
                        print(f"🔄 Making {result['api_call']} API call...")
                        api_results = await self._handle_api(result["api_call"], result.get("api_data", {}))
                        
                        # Let LLM handle the API announcement
                        if api_results and api_results.get("success"):
                            print("📢 Processing API results for announcement...")
                            announcement_result = await self.llm.process("ANNOUNCE_API_RESULTS", self.session, api_results)
                            if announcement_result.get("response"):
                                print(f"🏥 Agent: {announcement_result['response']}")
                                await self.audio.speak(announcement_result["response"])
                                self.session.add_message("assistant", announcement_result["response"])
                    
                    # Speak response
                    response = result.get("response", "")
                    if response:
                        print(f"🏥 Agent: {response}")
                        await self.audio.speak(response)
                        self.session.add_message("assistant", response)
                    
                    # Check if done
                    if result.get("done", False):
                        print("✅ LLM says conversation is complete")
                        await self._end_call("Completed")
                        break
                    
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"Turn error: {e}")
                    error_msg = "I'm sorry, could you repeat that?"
                    print(f"🏥 Agent: {error_msg}")
                    await self.audio.speak(error_msg)
                
        except KeyboardInterrupt:
            await self._end_call("Interrupted")
        except Exception as e:
            logger.error(f"Session error: {e}")
            await self._end_call("Error")
        finally:
            self._cleanup()
    
    async def _handle_api(self, api_type: str, api_data: Dict) -> Dict:
        """Handle API calls"""
        try:
            result = await self.api.call(api_type, api_data)
            
            # Store result
            self.session.data[f"{api_type}_result"] = result
            
            if result.get("success"):
                print(f"✅ {api_type} API successful")
            else:
                print(f"❌ {api_type} API failed: {result.get('error', 'Unknown error')}")
            
            return result
                
        except Exception as e:
            logger.error(f"API error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _end_call(self, reason: str):
        """End the call"""
        self.session.end_time = datetime.now()
        
        # Generate confirmation if we have enough data
        if self.session.data.get("name") and self.session.data.get("preferred_date"):
            if not self.session.data.get("confirmation_code"):
                import random
                import string
                confirmation = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))  # 5-digit as requested
                self.session.data["confirmation_code"] = confirmation
                
                end_msg = f"Perfect! Your appointment is confirmed with confirmation number {confirmation}. Thank you for calling!"
            else:
                end_msg = "Thank you for calling!"
        else:
            end_msg = "Thank you for calling. Please call back to complete your appointment."
        
        print(f"🏥 Agent: {end_msg}")
        await self.audio.speak(end_msg)
        self.session.add_message("assistant", end_msg)
        
        # Save session
        filename = self.session.save()
        
        # Show summary
        duration = (self.session.end_time - self.session.start_time).total_seconds()
        print(f"\n✅ Call ended: {reason}")
        print(f"📞 Duration: {round(duration/60, 2)} minutes")
        print(f"💾 Data: {self.session.data}")
        print(f"📁 Saved: {filename}")
    
    def _cleanup(self):
        """Cleanup"""
        try:
            self.audio.cleanup()
        except:
            pass

# Configuration
def load_config():
    """Load config from environment"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    config = {}
    required = {
        'jwt_token': 'JWT_TOKEN',
        'endpoint_url': 'ENDPOINT_URL',
        'project_id': 'PROJECT_ID',
        'connection_id': 'CONNECTION_ID',
        'a2a_url': 'A2A_URL',
        'insurance_api_key': 'X_INF_API_KEY'
    }
    
    for key, env_var in required.items():
        value = os.getenv(env_var)
        if value:
            config[key] = value.strip()
        else:
            print(f"❌ Missing: {env_var}")
            return None
    
    return config

async def main():
    """Healthcare Voice Agent v1.0.0"""
    print("🏥 Healthcare Voice Agent v1.0.0")
    print("=" * 40)
    
    config = load_config()
    if not config:
        return
    
    try:
        agent = HealthcareAgent(
            config['jwt_token'],
            config['endpoint_url'],
            config['project_id'],
            config['connection_id'],
            config['a2a_url'],
            config['insurance_api_key']
        )
        
        await agent.start()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
