#!/bin/bash

mkdir -p backend/api
mkdir -p backend/services
mkdir -p backend/models
mkdir -p backend/utils
mkdir -p backend/data/sample_resumes

touch backend/main.py
touch backend/requirements.txt

touch backend/api/resume.py
touch backend/api/interview.py
touch backend/api/voice.py

touch backend/services/resume_parser.py
touch backend/services/ats_scorer.py
touch backend/services/question_generator.py
touch backend/services/feedback_generator.py
touch backend/services/voice_engine.py

touch backend/models/schemas.py

touch backend/utils/text_cleaner.py
