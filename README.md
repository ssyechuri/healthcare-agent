# Healthcare Voice Agent

An AI-powered voice assistant for medical appointment scheduling that handles patient intake, insurance verification, eligibility checks, and appointment booking through natural voice interactions.

## Features

- **Voice Interface**: Natural speech recognition and text-to-speech responses
- **Patient Data Collection**: Automated intake of patient information
- **Insurance Processing**: Real-time insurance discovery and eligibility verification via A2A APIs
- **Appointment Scheduling**: Complete booking workflow with confirmation codes
- **Session Management**: Comprehensive conversation logging and data persistence
- **LLM Integration**: Intelligent conversation flow management

## Prerequisites

- Python 3.8 or higher
- macOS, Windows, or Linux
- Microphone and speakers/headphones
- Active internet connection
- LLM API access (configured endpoint)
- Insurance A2A API access

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ssyechuri/healthcare-agent
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system audio dependencies:**

   **macOS:**
   ```bash
   # Install system audio packages if needed
   brew install portaudio
   ```

   **Ubuntu/Debian:**
   ```bash
   sudo apt-get install python3-pyaudio portaudio19-dev
   ```

   **Windows:**
   ```bash
   # PyAudio should install automatically with pip
   ```

## Dependencies

### Core Python Packages
```
speech_recognition>=3.10.0
gtts>=2.3.0
pygame>=2.5.0
pyttsx3>=2.90
requests>=2.28.0
asyncio
tempfile
logging
json
uuid
dataclasses
datetime
re
os
```

### Audio System Requirements
- **Microphone access** for speech input
- **Audio output** (speakers/headphones) for TTS responses
- **Internet connection** for Google Speech Recognition and TTS services

### API Requirements
- **LLM Endpoint**: Configured JWT token and endpoint URL
- **Insurance A2A API**: API key and endpoint for insurance verification

## Configuration

Create a `.env` file in the project root with the following variables:

```env
# LLM Configuration
JWT_TOKEN=your_jwt_token_here
ENDPOINT_URL=your_llm_endpoint_url
PROJECT_ID=your_project_id
CONNECTION_ID=your_connection_id

# Insurance API Configuration
A2A_URL=your_a2a_api_url
X_INF_API_KEY=your_insurance_api_key
```

## Usage

1. **Set up your environment variables** (see Configuration above)

2. **Run the agent:**
   ```bash
   python healthcare_agent.py
   ```

3. **Interact with the voice agent:**
   - The agent will greet you and ask for your name
   - Speak clearly into your microphone
   - Follow the prompts to provide:
     - Full name
     - Phone number
     - Date of birth
     - State
     - Reason for appointment
     - Provider information
     - Preferred appointment date/time

4. **Complete the flow:**
   - The agent will verify your insurance
   - Process eligibility checks
   - Book your appointment
   - Provide a confirmation code

## Session Management

- All conversations are automatically saved to `sessions/` directory
- Each session includes:
  - Patient data collected
  - Full conversation history
  - API call results
  - Session duration and metadata

## Audio Configuration

The agent uses multiple TTS engines for reliability:
- **Primary**: Google Text-to-Speech (gTTS) 
- **Fallback**: pyttsx3 for offline operation

Speech recognition uses Google Speech Recognition API with:
- 15-second listening timeout
- 6-second phrase time limit
- Automatic ambient noise adjustment

## API Integration

### Insurance Discovery API
```python
# Example usage in the agent
api_call_result = await self.api.call("discovery", {
    "name": patient_name,
    "dob": date_of_birth,
    "state": patient_state
})
```

### Eligibility Verification API
```python
# Example usage in the agent  
api_call_result = await self.api.call("eligibility", {
    "name": patient_name,
    "dob": date_of_birth,
    "member_id": member_id,
    "payer": payer_name,
    "provider_name": provider_name,
    "provider_npi": provider_npi
})
```

## Troubleshooting

### Audio Issues
- **No microphone input**: Check system permissions for microphone access
- **No audio output**: Verify speakers/headphones are connected and volume is up
- **Poor recognition**: Ensure quiet environment and speak clearly

### API Issues
- **Authentication errors**: Verify your JWT token and API keys are correct
- **Timeout errors**: Check internet connection and API endpoint availability
- **Rate limiting**: Implement delays between requests if needed

### Performance
- **Slow responses**: Check internet speed for real-time API calls
- **Memory usage**: Monitor for long sessions, restart if needed

## Development

### Project Structure
```
healthcare-voice-agent/
├── healthcare_agent.py              # Main agent application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── .env.example         # Example environment file
├── sessions/            # Generated session logs
├── README.md           # This file
└── .gitignore         # Git ignore file
```

### Adding Features
The agent uses a simple LLM-driven architecture where:
- All conversation logic is handled by the LLM
- State is maintained in the `Session` object
- API calls are triggered based on LLM decisions

### Testing
```bash
# Run with debug logging
python healthcare_agent.py --debug

# Test audio components separately
python -c "import speech_recognition as sr; print('Speech Recognition OK')"
python -c "import pyttsx3; print('TTS OK')"
```

## Security Considerations

- **Environment Variables**: Never commit `.env` file to version control
- **Session Data**: Contains sensitive patient information - secure appropriately
- **API Keys**: Rotate regularly and use least-privilege access
- **Audio Data**: Speech is processed via Google APIs - review privacy policies

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section above
- Review session logs in `sessions/` directory

## Changelog

### v1.0.0
- Initial release
- Voice interface with speech recognition and TTS
- Insurance discovery and eligibility verification
- Appointment scheduling with confirmation codes
- Session logging and data persistence# healthcare-agent
healthcare voice agent
