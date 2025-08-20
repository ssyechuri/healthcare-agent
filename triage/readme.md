# Symptom Triage Agent

An AI-powered medical triage system that conducts systematic symptom assessment through structured questioning, determines urgency levels, and provides appropriate medical recommendations. Features A2A (Agent-to-Agent) protocol compatibility for integration with supervisor agents and comprehensive clinical data tracking.

## 🏥 Features

- **Intelligent Symptom Assessment**: AI-powered analysis of patient complaints using GPT-4o
- **Structured Clinical Workflow**: Multi-stage triage process (initial → generic → specific → assessment)
- **Dynamic Question Generation**: Contextual follow-up questions based on identified symptoms
- **Urgency Level Classification**: Three-tier urgency system (low, medium, high) with appropriate recommendations
- **CSV-Based Symptom Database**: Configurable symptom database with customizable questions
- **A2A Protocol Support**: Seamless integration with supervisor agents and other AI systems
- **Comprehensive Session Tracking**: Complete conversation logs and clinical notes
- **Emergency Detection**: Automatic identification of high-urgency situations requiring immediate care
- **Azure OpenAI Integration**: Enterprise-grade AI with configurable endpoints

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Azure OpenAI API access (or compatible OpenAI endpoint)
- Required Python packages (see `requirements.txt`)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ssyechuri/healthcare_agent/main/triage.git
   cd symptom-triage-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Set up environment variables**
   ```bash
   # Required
   export OPENAI_URL="https://your-azure-openai-endpoint.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2023-12-01-preview"
   export OPENAI_API_KEY="your-azure-openai-api-key"
   
   # Optional
   export OPENAI_PROJECT_ID="your-project-id"
   export OPENAI_CONNECTION_ID="your-connection-id"
   export OPENAI_CONNECTION_NAME="your-connection-name"
   export OPENAI_PROVIDER="Azure-OpenAI"
   export OPENAI_MODEL="gpt-4o"
   export SYMPTOMS_CSV="symptoms.csv"
   ```

### Basic Usage

```python
import asyncio
from symptom_triage_agent import SymptomTriageAgent, load_triage_config

async def main():
    # Load configuration
    config = load_triage_config()
    
    # Initialize agent
    agent = SymptomTriageAgent(
        openai_url=config['openai_url'],
        openai_api_key=config['openai_api_key'],
        openai_model=config.get('openai_model', 'gpt-4o')
    )
    
    # Start triage session
    session = await agent.start_triage_session(
        patient_name="John Doe",
        patient_phone="555-0123",
        chief_complaint="I have a headache"
    )
    
    # Process patient responses
    result = await agent.process_user_input("I have a severe headache for 3 hours")
    print(f"Agent: {result['response']}")
    
    # Continue conversation until triage complete
    while not result.get('triage_complete'):
        user_input = input("Patient: ")
        result = await agent.process_user_input(user_input)
        print(f"Agent: {result['response']}")
    
    # Get final assessment
    if result.get('triage_complete'):
        print(f"Urgency Level: {result['urgency_level']}")
        print(f"Recommendation: {result['recommendation']}")

if __name__ == "__main__":
    asyncio.run(main())
```

## 📋 Symptom Database

The agent uses a CSV-based symptom database (`symptoms.csv`) with the following structure:

```csv
symptoms,question1,question2,question3
headache,Is the headache throbbing or constant?,Do you have any visual changes or sensitivity to light?,Have you had any recent head injuries or changes in medication?
chest pain,Is the pain sharp crushing or burning?,Does the pain radiate to your arm jaw or back?,Do you have shortness of breath or sweating?
```

### Adding New Symptoms

1. **Edit symptoms.csv**: Add new rows with symptom name and three follow-up questions
2. **Restart the agent**: The database is loaded on initialization
3. **Test new symptoms**: Verify the agent recognizes and asks appropriate questions

### Symptom Synonyms

The agent automatically recognizes common synonyms:
- `headache` → head pain, migraine, head ache
- `chest pain` → chest hurt, heart pain, chest discomfort
- `abdominal pain` → stomach pain, belly pain, stomach ache

## 🔄 Triage Workflow

### 1. Initial Assessment
- Extract chief complaint and identify symptoms
- Determine if medical triage is required
- Extract patient demographics

### 2. Generic Questions
- Ask about symptom duration
- Assess severity on 1-10 scale
- Validate and standardize responses

### 3. Specific Questions
- Ask symptom-specific clinical questions
- Process up to 3 follow-up questions per symptom
- Record detailed clinical responses

### 4. Final Assessment
- Analyze all collected data using AI
- Determine urgency level (low/medium/high)
- Provide specific medical recommendations
- Generate comprehensive clinical notes

## 🔗 A2A Protocol Integration

The agent supports Agent-to-Agent protocol for seamless integration with supervisor systems:

```python
from symptom_triage_agent import A2AProtocol, integrate_with_supervisor

# Integration with supervisor agent
result = await integrate_with_supervisor(
    patient_name="Jane Smith",
    patient_phone="555-0456",
    chief_complaint="chest pain"
)

# Get handoff data for supervisor
handoff_data = A2AProtocol.create_triage_handoff(triage_agent)
```

### Handoff Data Structure

```json
{
  "triage_complete": true,
  "patient_name": "Jane Smith",
  "urgency_level": "high",
  "doctor_type": "EMERGENCY - Call 911 immediately",
  "recommendation": "Immediate emergency care required",
  "clinical_notes": "TRIAGE ASSESSMENT:\nChief Complaint: chest pain...",
  "session_id": "uuid-string",
  "a2a_protocol_version": "1.0"
}
```

## ⚠️ Safety Features

### Emergency Detection
- Automatic identification of high-urgency symptoms
- Immediate 911 recommendations for life-threatening conditions
- Clear emergency alerts in responses

### Clinical Validation
- Structured data collection and validation
- Comprehensive session logging
- Audit trail for medical compliance

### Privacy Protection
- No persistent storage of sensitive data by default
- Configurable data retention policies

## 🧪 Testing

Run the built-in test suite:

```bash
python triage_agent.py
```

This will execute a complete triage simulation with sample patient data.

### Example Test Scenarios

1. **Emergency Scenario**: Chest pain with radiation
2. **Medium Urgency**: Severe headache with visual symptoms
3. **Low Urgency**: Minor back pain from lifting

## 📊 Session Management

### Automatic Session Saving
- Sessions saved to `triage_sessions/` directory
- JSON format with complete conversation history
- Clinical notes and assessment data included

### Session Data Structure
```json
{
  "session_metadata": {
    "session_id": "uuid",
    "start_time": "2025-08-20T14:30:00",
    "duration_minutes": 5.2
  },
  "clinical_assessment": {
    "symptoms": ["headache"],
    "urgency_level": "medium",
    "recommendation": "See neurologist within 24 hours"
  },
  "conversation_log": [...]
}
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_URL` | ✅ | - | Azure OpenAI endpoint URL |
| `OPENAI_API_KEY` | ✅ | - | API key for authentication |
| `OPENAI_MODEL` | ❌ | `gpt-4o` | AI model to use |
| `OPENAI_PROVIDER` | ❌ | `Azure-OpenAI` | Provider type |
| `SYMPTOMS_CSV` | ❌ | `symptoms.csv` | Path to symptom database |

### Azure OpenAI Setup

1. **Create Azure OpenAI Resource**
2. **Deploy GPT-4o Model**
3. **Get Endpoint URL and API Key**
4. **Configure Environment Variables**

Example endpoint URL format:
```
https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions?api-version=2023-12-01-preview
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-capability`
3. **Make your changes**: Follow existing code style and patterns
4. **Add tests**: Ensure new features are tested
5. **Submit a pull request**: Include description of changes

### Development Guidelines

- Follow Python PEP 8 style guidelines
- Add comprehensive docstrings to new functions
- Include error handling and logging
- Test with multiple symptom scenarios
- Maintain HIPAA compliance considerations

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚖️ Medical Disclaimer

**IMPORTANT**: This software is for demonstration and research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about medical conditions. In case of medical emergency, call 911 immediately.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/ssyechuri/healthcare_agent/main/triage/issues)
- **Documentation**: [Wiki](https://github.com/ssyechuri/healthcare_agent/main/triage/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/ssyechuri/healthcare_agent/main/triage/discussions)

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Patient Input  │───▶│  Triage Agent    │───▶│  Medical AI     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  Symptom DB      │    │  Clinical       │
                       │  (CSV)           │    │  Assessment     │
                       └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │  A2A Protocol    │    │  Session        │
                       │  Integration     │    │  Management     │
                       └──────────────────┘    └─────────────────┘
```

## 📈 Roadmap

- [ ] **Multi-language Support**: Expand to Spanish, French, and other languages
- [ ] **Voice Integration**: Add speech-to-text and text-to-speech capabilities
- [ ] **Clinical Decision Trees**: Advanced symptom-specific decision algorithms
- [ ] **Integration APIs**: REST API for external system integration
- [ ] **Dashboard UI**: Web interface for session monitoring and management
- [ ] **Machine Learning**: Continuous improvement from triage outcomes

---

**Made with ❤️ for healthcare innovation**
