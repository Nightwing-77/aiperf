# WER/CER Evaluation Service

Standalone service for evaluating TTS audio quality using Whisper and jiwer.

## Overview

This service provides HTTP endpoints for computing Word Error Rate (WER) and Character Error Rate (CER) for TTS audio responses. It uses OpenAI's Whisper model for speech-to-text transcription and jiwer for WER/CER calculation.

## Installation

Install the required dependencies:

```bash
uv pip install 'aiperf[wer_cer]'
```

Or install manually:

```bash
uv pip install openai-whisper jiwer
```

## Running the Service

### Standalone Mode

Run the service as a standalone FastAPI server:

```bash
python -m aiperf.tts.wer_cer_service
```

Or using the module:

```bash
uv run python -c "from aiperf.tts.wer_cer_service import main; main()"
```

### Environment Variables

Configure the service with environment variables:

- `WER_CER_HOST`: Host to bind to (default: `0.0.0.0`)
- `WER_CER_PORT`: Port to bind to (default: `8001`)
- `WHISPER_MODEL_SIZE`: Whisper model size (default: `base`, options: `tiny`, `base`, `small`, `medium`, `large`)
- `WHISPER_DEVICE`: Device to run on (default: `cpu`, options: `cpu`, `cuda`)

Example:

```bash
export WHISPER_MODEL_SIZE=small
export WHISPER_DEVICE=cuda
python -m aiperf.tts.wer_cer_service
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns service health status.

### Evaluate Audio (File Upload)

```bash
POST /evaluate
Content-Type: multipart/form-data

Parameters:
- audio_file: Audio file (WAV, MP3, etc.)
- reference_text: Reference/transcript text
- language: Language code (default: "en")
```

Example using curl:

```bash
curl -X POST "http://localhost:8001/evaluate" \
  -F "audio_file=@audio.wav" \
  -F "reference_text=hello world" \
  -F "language=en"
```

### Evaluate Audio (JSON)

```bash
POST /evaluate_json
Content-Type: application/json

{
  "audio_data": "base64-encoded-audio-bytes",
  "audio_format": "wav",
  "reference_text": "hello world",
  "language": "en"
}
```

## Response Format

```json
{
  "wer": 25.0,
  "cer": 10.5,
  "transcription": "hello word",
  "reference_text": "hello world",
  "language": "en"
}
```

- `wer`: Word Error Rate as percentage (0-100)
- `cer`: Character Error Rate as percentage (0-100)
- `transcription`: Transcribed text from Whisper
- `reference_text`: Reference text that was compared
- `language`: Language used for transcription

## Integration with AIPerf

Enable WER/CER evaluation in AIPerf benchmarks:

### CLI Flags

```bash
aiperf --wer-cer-enabled \
       --wer-cer-service-url http://localhost:8001 \
       --wer-cer-language en \
       --wer-cer-timeout 30.0 \
       ...
```

### YAML Configuration

```yaml
benchmark:
  # ... other config ...

wer_cer:
  enabled: true
  service_url: http://localhost:8001
  language: en
  timeout: 30.0
```

## How It Works

1. **Audio Input**: Accepts audio file (WAV, MP3, etc.) or base64-encoded audio bytes
2. **Transcription**: Uses Whisper to transcribe audio to text
3. **Text Normalization**: Removes punctuation, converts to lowercase
4. **WER Calculation**: Computes Word Error Rate using jiwer
5. **CER Calculation**: Computes Character Error Rate using jiwer
6. **Response**: Returns WER, CER, transcription, and metadata

## Whisper Model Sizes

- `tiny`: Fastest, lowest accuracy (~39M params)
- `base`: Good balance (~74M params) - **default**
- `small`: Better accuracy (~244M params)
- `medium`: High accuracy (~769M params)
- `large`: Best accuracy (~1.5B params)

Larger models provide better accuracy but require more memory and compute time.

## Language Support

Whisper supports multiple languages. Common language codes:

- `en`: English
- `es`: Spanish
- `fr`: French
- `de`: German
- `it`: Italian
- `pt`: Portuguese
- `ru`: Russian
- `ja`: Japanese
- `zh`: Chinese
- `ko`: Korean
- `ar`: Arabic
- `hi`: Hindi

Use `auto` for automatic language detection (may be less accurate).

## Performance Considerations

- **Model Size**: Larger models are slower but more accurate
- **Device**: Use `cuda` for GPU acceleration if available
- **Timeout**: Set timeout based on expected audio length (default: 30s)
- **Batch Processing**: The service processes one request at a time; scale horizontally for concurrent evaluation

## Error Handling

The service returns HTTP 500 on errors with error details in the response body. Common errors:

- Missing audio data or reference text
- Invalid base64 encoding
- Whisper transcription failure
- Timeout exceeded

## Development

### Running Tests

```bash
uv run pytest tests/unit/tts/test_wer_cer_service.py
```

### Code Structure

- `wer_cer_service.py`: FastAPI service and WERCERService class
- `wer_cer_record_processor.py`: AIPerf record processor integration
- `wer_cer.py`: Configuration model

## License

SPDX-License-Identifier: Apache-2.0
