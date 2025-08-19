# Enhanced Healthcare Voice Agent - A2A Triage Integration

🏥 **Healthcare Voice Assistant with Intelligent Medical Triage**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Healthcare](https://img.shields.io/badge/domain-healthcare-green.svg)](https://github.com/healthcare-ai)
[![Voice AI](https://img.shields.io/badge/type-voice--ai-purple.svg)](https://github.com/healthcare-ai)

## 🌟 Overview

The Enhanced Healthcare Voice Agent is a sophisticated AI-powered voice assistant designed specifically for healthcare appointment scheduling with integrated medical triage capabilities. It combines advanced speech processing, intelligent conversation management, and real-time medical assessment to provide a comprehensive healthcare interaction experience.

### 🎯 Key Features

- **🗣️ Advanced Voice Processing**: High-quality speech recognition and text-to-speech with noise adaptation
- **🩺 Medical Triage Integration**: A2A (Agent-to-Agent) protocol for seamless triage handoffs
- **🔗 Insurance API Integration**: MCP protocol support for real-time insurance verification
- **🧠 GPT-4 Powered Intelligence**: Advanced conversation management and medical reasoning
- **🚨 Emergency Detection**: Automatic identification and appropriate referral of emergency situations
- **📋 Complete Appointment Workflow**: End-to-end scheduling with confirmation codes

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Voice Input   │────│  Speech Engine   │────│  LLM Processor  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Audio Output   │────│ Session Manager  │────│ Triage Agent    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   MCP Insurance  │    │  A2A Protocol   │
                       │       API        │    │    Handler      │
                       └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Microphone and speakers/headphones
- Valid API credentials (see Configuration section)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/healthcare-ai/enhanced-voice-agent.git
cd enhanced-voice-agent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment variables:**
```bash
cp .env.example .env
# Edit .env with your API credentials
```

4. **Run the agent:**
```bash
python enhanced_healthcare_agent.py
```

## ⚙️ Configuration

Create a `.env` file with the following required variables:

### Core LLM Configuration
```bash
# Required: Main LLM API Configuration
JWT_TOKEN=your_jwt_token_here
ENDPOINT_URL=https://your-llm-endpoint.com/v1/chat/completions
PROJECT_ID=your_project_id
CONNECTION_ID=your_connection_id
```

### Insurance Integration
```bash
# Required: MCP Insurance API
MCP_URL=https://your-mcp-endpoint.com/api
X_INF_API_KEY=your_insurance_api_key
```

### Triage Integration (Optional)
```bash
# Optional: For Medical Triage Features
OPENAI_URL=https://your-openai-endpoint.azure.com
OPENAI_API_KEY=your_openai_api_key
OPENAI_PROJECT_ID=your_openai_project
OPENAI_CONNECTION_ID=your_openai_connection
OPENAI_CONNECTION_NAME=your_connection_name
OPENAI_PROVIDER=Azure-OpenAI
OPENAI_MODEL=gpt-4o
SYMPTOMS_CSV=path/to/symptoms.csv
```

## 🎭 Usage Examples

### Basic Appointment Scheduling
```
Agent: "Hello! I'm your healthcare appointment assistant. Could you please tell me your full name?"
User: "Hi, my name is John Smith"
Agent: "Thank you John. Could you please provide your phone number?"
User: "555-123-4567"
Agent: "What's the reason for your visit today?"
User: "I need a routine checkup"
# ... continues with appointment scheduling flow
```

### Medical Triage Example
```
Agent: "What's the reason for your visit today?"
User: "I have severe chest pain and trouble breathing"
Agent: "I understand you have a medical concern. Let me conduct a quick medical assessment..."
# Agent initiates A2A triage handoff
Triage: "Can you describe your chest pain? When did it start?"
# ... medical triage assessment continues
# Results in urgency level and doctor recommendation
```

## 🧩 Core Components

### EnhancedSession
Manages conversation state, triage data, and API call tracking:
```python
session = EnhancedSession()
session.add_message("user", "I need an appointment")
session.add_triage_data(triage_handoff)
completion = session.get_completion_percentage()
```

### Audio System
Advanced speech processing with healthcare optimizations:
```python
audio = Audio()
user_input = await audio.listen()  # Intelligent speech recognition
await audio.speak("Your appointment is confirmed")  # Natural TTS
```

### MCP Insurance API
Real-time insurance verification and benefits checking:
```python
mcp_api = MCPInsuranceAPI(mcp_url, api_key)
result = await mcp_api.call_insurance_api("discovery", patient_data)
```

### A2A Triage Integration
Seamless handoff to medical triage specialist:
```python
triage_agent = SymptomTriageAgent(...)
handoff_data = A2AProtocol.create_triage_handoff(triage_agent)
```

## 🔄 Conversation Flow

1. **Greeting & Name Collection**
2. **Phone Number Verification**
3. **Reason Assessment**
   - Medical concerns → Triage handoff
   - Non-medical → Standard flow
4. **Medical Triage (if required)**
   - Symptom assessment
   - Urgency determination
   - Doctor recommendation
5. **Patient Information**
   - Date of birth
   - State/location
   - Insurance discovery API call
6. **Provider Information**
   - Provider name
   - Eligibility verification API call
7. **Appointment Scheduling**
   - Preferred date/time
   - Immediate confirmation
   - Confirmation code generation
8. **Session Completion**

## 🩺 Medical Triage Features

The agent includes sophisticated medical triage capabilities:

- **Symptom Assessment**: Structured medical questioning
- **Emergency Detection**: Automatic 911 referral for critical conditions
- **Urgency Classification**: Priority levels (low, medium, high, emergency)
- **Doctor Recommendations**: Specialist vs. general practitioner routing
- **Clinical Documentation**: Comprehensive triage notes

## 🔗 API Integrations

### Insurance Discovery API
Automatically discovers patient insurance information:
```json
{
  "patientFirstName": "John",
  "patientLastName": "Smith",
  "patientDateOfBirth": "1985-03-15",
  "patientState": "CA"
}
```

### Benefits Eligibility API
Verifies coverage and benefits:
```json
{
  "subscriberId": "ABC123456",
  "payerName": "Blue Cross Blue Shield",
  "providerFirstName": "Sarah",
  "providerLastName": "Johnson",
  "providerNpi": "1234567890"
}
```

## 📊 Session Analytics

Each conversation generates comprehensive analytics:

- **Completion Percentage**: Progress tracking
- **Turn Analysis**: Conversation efficiency metrics
- **API Performance**: Response times and success rates
- **Triage Outcomes**: Medical assessment results
- **Audio Quality**: Speech recognition accuracy

## 🔒 Security & Privacy

- **HIPAA Compliance**: Privacy-focused design patterns
- **Secure API Communications**: Encrypted endpoints
- **Session Isolation**: No cross-session data leakage
- **Audit Logging**: Comprehensive interaction tracking
- **Emergency Protocols**: Appropriate escalation procedures

## 🧪 Testing

### Run A2A Integration Tests
```bash
python enhanced_healthcare_agent.py test
```

### Test MCP API Formats
```bash
python enhanced_healthcare_agent.py test-mcp
```

### Manual Testing
```bash
python enhanced_healthcare_agent.py
```

## 📁 Project Structure

```
enhanced-voice-agent/
├── enhanced_healthcare_agent.py    # Main agent implementation
├── symp_triage.py                 # Triage agent module
├── requirements.txt               # Python dependencies
├── .env.example                  # Environment template
├── README.md                     # This file
├── sessions/                     # Session storage directory
└── tests/                        # Test files
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Test voice interactions thoroughly

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [Wiki](https://github.com/healthcare_agent/supervisor/wiki)
- **Issues**: [GitHub Issues](https://github.com/healthcare_agent/supervisor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/healthcare_agent/supervisor/discussions)

## 🙏 Acknowledgments

- **Speech Recognition**: Built on SpeechRecognition library
- **Text-to-Speech**: Powered by gTTS and pyttsx3
- **Medical Intelligence**: Enhanced by GPT-4 capabilities

## 📈 Roadmap

- [ ] Multi-language support
- [ ] Advanced symptom databases
- [ ] Integration with EHR systems
- [ ] Mobile app companion
- [ ] Telehealth video integration
- [ ] Advanced analytics dashboard

---

**⚠️ Medical Disclaimer**: This system is designed to assist with appointment scheduling and basic triage. It is not a replacement for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers for medical concerns.
