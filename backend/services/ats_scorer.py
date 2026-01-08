import os
from typing import Dict, List, Optional
import google.generativeai as genai
from dotenv import load_dotenv
import random
load_dotenv()
class AtsScorer:
    def __init__(self):
        self.api_key=os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=self.api_key)
        self.model=genai.GenerativeModel("gemini-pro")
        self.criteria_weights={
            'skills_match': 30,
            'experience_level': 25,
            'education': 15,
            'keyword_density': 15,
            'formatting': 10,
            'achievements': 5
        }
        self.prompt_template="""
        You are an expert ATS (Applicant Tracking System) analyst with 10+ years of experience in resume evaluation.
        Analyze the following resume against the job description and provide a detailed scoring and feedback.
        
        JOB TITLE: {job_title}
        COMPANY: {company_name}
        
        JOB DESCRIPTION:
        {job_description}
        
        RESUME CONTENT:
        {resume_content}
        
        Please provide your analysis in the following JSON format:
        {{
            "scores": {{
                "skills_match": 0-100,  // How well the skills match the job requirements
                "experience_level": 0-100,  // Relevance of experience to the role
                "education": 0-100,  // Education level and relevance
                "keyword_density": 0-100,  // Presence of relevant keywords
                "formatting": 0-100,  // Resume structure and readability
                "achievements": 0-100  // Quantifiable achievements
            }},
            "overall_score": 0-100,
            "strengths": ["list", "of", "strengths"],
            "weaknesses": ["list", "of", "weaknesses"],
            "suggestions": ["list", "of", "suggestions"],
            "missing_keywords": ["list", "of", "missing", "keywords"],
            "ats_optimization_tips": ["list", "of", "tips"]
        }}
        """
    async def score_resume(
self,
        resume_data: Dict[str, Any],
        job_title: str,
        company_name: str,
        job_description: str
    )->Dict[str,Any]:
        """
        Score a resume against a job description using the Gemini API.
        
        Args:
            resume_data: Parsed resume data from resume_parser.py
            job_title: The target job title
            company_name: The target company name
            job_description: The job description text
            
        Returns:
            Dictionary containing scoring results and feedback
        """
        try:
            # Prepare resume content for analysis
            resume_content = self._prepare_resume_content(resume_data)
            
            # Generate the prompt
            prompt = self.scoring_prompt.format(
                job_title=job_title,
                company_name=company_name,
                job_description=job_description,
                resume_content=resume_content
            )
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse the response
            result = self._parse_ats_response(response.text)
            
            # Calculate weighted score
            if 'scores' in result:
                result['overall_score'] = self._calculate_weighted_score(result['scores'])
            
            return {
                "success": True,
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    def _prepare_resume_content(self,resume_data:Dict[str,Any])->str:
        """Convert parsed resume data into a structured content summary."""
        sections=[]
        if resume_data.get('name'):
            sections.append(f"Name: {resume_data['name']}")
        contact_info = []
        if resume_data.get('email'):
            contact_info.append(f"Email: {resume_data['email']}")
        if resume_data.get('phone'):
            contact_info.append(f"Phone: {resume_data['phone']}")
        if contact_info:
            sections.append("CONTACT: " + " | ".join(contact_info))
        
        # Professional Summary
        if resume_data.get('summary'):
            sections.append(f"SUMMARY:\n{resume_data['summary']}")
        
        # Skills
        if resume_data.get('skills'):
            sections.append(f"SKILLS:\n{', '.join(resume_data['skills'])}")
        
        # Experience
        if resume_data.get('experience'):
            sections.append("EXPERIENCE:")
            for exp in resume_data['experience']:
                exp_text = f"- {exp.get('title', '')}"
                if exp.get('company'):
                    exp_text += f" at {exp['company']}"
                if exp.get('duration'):
                    exp_text += f" ({exp['duration']})"
                sections.append(exp_text)
                
                if exp.get('description'):
                    if isinstance(exp['description'], list):
                        for desc in exp['description']:
                            sections.append(f"  • {desc}")
                    else:
                        sections.append(f"  • {exp['description']}")
        
        # Education
        if resume_data.get('education'):
            sections.append("EDUCATION:")
            for edu in resume_data['education']:
                edu_text = f"- {edu.get('degree', '')}"
                if edu.get('institution'):
                    edu_text += f" from {edu['institution']}"
                if edu.get('year'):
                    edu_text += f" ({edu['year']})"
                sections.append(edu_text)
        
        # Projects
        if resume_data.get('projects'):
            sections.append("PROJECTS:")
            for proj in resume_data['projects']:
                proj_text = f"- {proj.get('title', 'Untitled Project')}"
                if proj.get('description'):
                    if isinstance(proj['description'], list):
                        proj_text += ": " + "; ".join(proj['description'])
                    else:
                        proj_text += f": {proj['description']}"
                sections.append(proj_text)
        
        # Certifications
        if resume_data.get('certifications'):
            sections.append("CERTIFICATIONS:")
            for cert in resume_data['certifications']:
                if isinstance(cert, dict):
                    cert_text = f"- {cert.get('name', '')}"
                    if cert.get('issuer'):
                        cert_text += f" from {cert['issuer']}"
                    if cert.get('year'):
                        cert_text += f" ({cert['year']})"
                    sections.append(cert_text)
                else:
                    sections.append(f"- {cert}")
        
        return "\n\n".join(sections)
    def _parse_ats_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the ATS response into a structured dictionary."""
        try:
            # Try to extract JSON from markdown code block if present
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()
            
            # Parse the JSON string
            result = json.loads(json_str)
            
            # Ensure all required fields are present
            required_fields = ['scores', 'overall_score', 'strengths', 'weaknesses', 'suggestions']
            for field in required_fields:
                if field not in result:
                    if field == 'scores':
                        result[field] = {k: 0 for k in self.criteria_weights}
                    elif field == 'overall_score':
                        result[field] = 0
                    else:
                        result[field] = []
            
            return result
            
        except Exception as e:
            raise ValueError(f"Failed to parse ATS response: {str(e)}")
    def _calculate_weighted_score(self, scores: Dict[str, int]) -> float:
        """Calculate the weighted overall score based on criteria weights."""
        if not scores:
            return 0.0
            
        total_weight = sum(self.criteria_weights.values())
        if total_weight == 0:
            return 0.0
            
        weighted_sum = 0
        for criterion, weight in self.criteria_weights.items():
            score = scores.get(criterion, 0)
            weighted_sum += (score * weight)
            
        return round(weighted_sum / total_weight, 1)
# Singleton instance
ats_scorer = AtsScorer()
# Public API
async def score_resume_ats(
    resume_data: Dict[str, Any],
    job_title: str,
    company_name: str,
    job_description: str
) -> Dict[str, Any]:
    """
    Score a resume against a job description using ATS criteria.
    
    Args:
        resume_data: Parsed resume data from resume_parser.py
        job_title: The target job title
        company_name: The target company name
        job_description: The job description text
        
    Returns:
        Dictionary containing ATS scoring results and feedback
    """
    return await ats_scorer.score_resume(
        resume_data, job_title, company_name, job_description
    )
            