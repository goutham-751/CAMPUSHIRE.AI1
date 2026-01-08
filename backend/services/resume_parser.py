import os
import re
import json
from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import PyPDF2
import docx
import spacy
from datetime import datetime
import datefinder

class ResumeParser:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "English language model not found. "
                "Run 'python -m spacy download en_core_web_sm'"
            )
        
        self.skills = self._load_skills_list()
        self.phone_regex = re.compile(
            r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        )
        self.email_regex = re.compile(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        )
        self.linkedin_regex = re.compile(
            r'(?:https?:\/\/)?(?:www\.)?linkedin\.com\/in\/[a-zA-Z0-9-]+\/?'
        )
        self.github_regex = re.compile(
            r'(?:https?:\/\/)?(?:www\.)?github\.com\/[a-zA-Z0-9-]+\/?'
        )
        self.date_pattern = re.compile(
            r'((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})|'
            r'(\d{1,2}[/-]\d{4})|'
            r'((19|20)\d{2})'
        )

    async def parse_resume(self, file_path: str) -> Dict[str, Any]:
        """Parse a resume file and return structured data"""
        try:
            text = self._extract_text(file_path)
            if not text:
                raise ValueError("Could not extract text from the file")

            doc = self.nlp(text)
            contact_info = self._extract_contact_info(text)
            name = self._extract_name(doc, contact_info.get('email', ''))
            
            return {
                "name": name,
                **contact_info,
                "skills": self._extract_skills(text),
                "experience": self._extract_experience(doc),
                "education": self._extract_education(doc),
                "projects": self._extract_projects(doc),
                "languages": self._extract_languages(text),
                "certifications": self._extract_certifications(doc),
                "raw_text": text[:5000]  # Store first 5000 chars for further processing
            }

        except Exception as e:
            raise Exception(f"Error parsing resume: {str(e)}")

    def _extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from resume text"""
        email_match = self.email_regex.search(text)
        phone_match = self.phone_regex.search(text)
        linkedin_match = self.linkedin_regex.search(text)
        github_match = self.github_regex.search(text)

        return {
            "email": email_match.group(0) if email_match else "",
            "phone": phone_match.group(0) if phone_match else "",
            "linkedin": linkedin_match.group(0) if linkedin_match else "",
            "github": github_match.group(0) if github_match else ""
        }

    def _extract_name(self, doc, email: str) -> str:
        """Extract candidate name using NER and email prefix"""
        # First try to get name from email
        if email:
            name_from_email = email.split('@')[0]
            name_parts = re.sub(r'[^a-zA-Z]', ' ', name_from_email).split()
            if len(name_parts) >= 2:
                return ' '.join(name_parts).title()
        
        # Fall back to NER
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text.strip()
        
        return ""

    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills using keyword matching and NLP"""
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Find skills using keyword matching
        found_skills = set()
        for skill in self.skills:
            if skill.lower() in text_lower:
                found_skills.add(skill)
        
        return sorted(list(found_skills))

    def _extract_experience(self, doc) -> List[Dict[str, Any]]:
        """Extract work experience with improved parsing"""
        experience = []
        current_exp = {}
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Look for company names
            for ent in sent.ents:
                if ent.label_ == "ORG" and len(ent.text) > 2:  # Filter out short ORG names
                    if current_exp and 'company' in current_exp:
                        experience.append(current_exp)
                        current_exp = {}
                    current_exp = {
                        'company': ent.text,
                        'title': '',
                        'duration': self._extract_duration(sent_text),
                        'description': []
                    }
                    break
            
            # Look for position titles (simplified)
            if ' - ' in sent_text and not current_exp.get('title'):
                parts = sent_text.split(' - ', 1)
                if len(parts[0].split()) < 5:  # Reasonable title length
                    current_exp['title'] = parts[0].strip()
            
            # Add bullet points to description
            if sent_text.startswith(('•', '-', '•', '▪', '▸')):
                current_exp['description'].append(sent_text[1:].strip())
        
        # Add the last experience if exists
        if current_exp:
            experience.append(current_exp)
            
        return experience[:5]  # Return max 5 most recent

    def _extract_education(self, doc) -> List[Dict[str, str]]:
        """Extract education information with improved parsing"""
        education = []
        current_edu = {}
        edu_keywords = ['university', 'college', 'institute', 'school']
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Look for educational institutions
            for ent in sent.ents:
                if (ent.label_ == "ORG" and 
                    any(keyword in ent.text.lower() for keyword in edu_keywords)):
                    
                    if current_edu:  # Save previous education entry
                        education.append(current_edu)
                    
                    current_edu = {
                        'institution': ent.text,
                        'degree': '',
                        'field_of_study': '',
                        'year': self._extract_year(sent_text)
                    }
                    break
            
            # Look for degree information
            if not current_edu.get('degree'):
                degree_indicators = ['bachelor', 'master', 'phd', 'bsc', 'msc', 'b.tech', 'm.tech']
                if any(indicator in sent_text.lower() for indicator in degree_indicators):
                    current_edu['degree'] = sent_text
        
        if current_edu:  # Add the last education entry
            education.append(current_edu)
            
        return education

    def _extract_projects(self, doc) -> List[Dict[str, str]]:
        """Extract projects section with improved parsing"""
        projects = []
        current_project = {}
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            
            # Look for project titles (lines that are in title case and not too long)
            if (sent_text.istitle() and 
                10 < len(sent_text) < 100 and 
                not any(word in sent_text.lower() for word in ['experience', 'education'])):
                
                if current_project:  # Save previous project
                    projects.append(current_project)
                
                current_project = {
                    'title': sent_text,
                    'description': []
                }
            elif current_project:  # Add to current project description
                current_project['description'].append(sent_text)
        
        if current_project:  # Add the last project
            projects.append(current_project)
            
        return projects[:10]  # Return max 10 projects

    def _extract_languages(self, text: str) -> List[str]:
        """Extract known programming languages and human languages"""
        languages = set()
        common_languages = {
            'english', 'spanish', 'french', 'german', 'chinese', 'japanese',
            'hindi', 'arabic', 'russian', 'portuguese'
        }
        
        # Check for language proficiency patterns
        lang_pattern = r'\b(' + '|'.join(common_languages) + r')\s*:\s*(fluent|native|proficient|intermediate|beginner)'
        matches = re.findall(lang_pattern, text.lower())
        languages.update([match[0].capitalize() for match in matches])
        
        return sorted(list(languages))

    def _extract_certifications(self, doc) -> List[str]:
        """Extract certifications and licenses"""
        certs = set()
        cert_keywords = ['certified', 'certification', 'license', 'certificate']
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if any(keyword in sent_text.lower() for keyword in cert_keywords):
                # Clean up and add the certification
                cert = re.sub(r'[^\w\s-]', '', sent_text).strip()
                if len(cert.split()) < 10:  # Reasonable length for a certification name
                    certs.add(cert)
        
        return sorted(list(certs))

    def _extract_duration(self, text: str) -> str:
        """Extract date ranges from text"""
        try:
            dates = list(datefinder.find_dates(text))
            if len(dates) >= 2:
                return f"{dates[0].strftime('%b %Y')} - {dates[1].strftime('%b %Y')}"
            elif dates:
                return f"{dates[0].strftime('%b %Y')} - Present"
        except:
            pass
        
        # Fallback to regex if datefinder fails
        matches = self.date_pattern.findall(text)
        if matches:
            return ' - '.join(match[0] for match in matches[:2])
        
        return ""

    def _extract_year(self, text: str) -> str:
        """Extract year from text"""
        matches = re.findall(r'(19|20)\d{2}', text)
        return matches[0] if matches else ""

    def _extract_text(self, file_path: str) -> str:
        """Extract text from different file formats"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_path.suffix.lower() == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                return self._extract_text_from_docx(file_path)
            elif file_path.suffix.lower() == '.txt':
                return file_path.read_text(encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
        except Exception as e:
            raise Exception(f"Error reading file: {str(e)}")

    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file with error handling"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
                return '\n'.join(text)
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def _extract_text_from_docx(self, file_path: Path) -> str:
        """Extract text from DOCX file with error handling"""
        try:
            doc = docx.Document(file_path)
            return '\n'.join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")

    def _load_skills_list(self) -> Set[str]:
        """Load skills from a JSON file or return default set"""
        skills_path = Path("data/skills.json")
        if skills_path.exists():
            try:
                with open(skills_path, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            except Exception:
                pass
        
        # Return a comprehensive default skills list
        return {
            # Programming Languages
            'Python', 'JavaScript', 'Java', 'C++', 'C#', 'Ruby', 'PHP', 'Swift', 'Kotlin', 'Go',
            'TypeScript', 'Rust', 'Scala', 'Dart', 'R', 'MATLAB', 'Perl', 'Haskell', 'Lua', 'Julia',
            
            # Web Development
            'HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js', 'Django', 'Flask',
            'FastAPI', 'Spring', 'Ruby on Rails', 'ASP.NET', 'jQuery', 'Bootstrap', 'Tailwind CSS',
            
            # Mobile Development
            'React Native', 'Flutter', 'Android', 'iOS', 'SwiftUI', 'Kotlin Multiplatform', 'Xamarin',
            
            # Database
            'SQL', 'PostgreSQL', 'MySQL', 'MongoDB', 'Redis', 'SQLite', 'Oracle', 'Microsoft SQL Server',
            'Cassandra', 'Elasticsearch', 'Firebase', 'DynamoDB', 'Neo4j',
            
            # DevOps & Cloud
            'Docker', 'Kubernetes', 'AWS', 'Azure', 'Google Cloud', 'Terraform', 'Ansible', 'Jenkins',
            'GitHub Actions', 'GitLab CI', 'CircleCI', 'Prometheus', 'Grafana', 'Nginx', 'Apache',
            
            # Data Science & AI/ML
            'Python', 'R', 'Pandas', 'NumPy', 'scikit-learn', 'TensorFlow', 'PyTorch', 'Keras',
            'OpenCV', 'NLTK', 'spaCy', 'Hugging Face', 'MLflow', 'PySpark', 'Dask', 'Jupyter',
            
            # Other Tools & Technologies
            'Git', 'Linux', 'Bash', 'PowerShell', 'REST API', 'GraphQL', 'gRPC', 'WebSockets',
            'OAuth', 'JWT', 'OAuth2', 'OIDC', 'OAuth 2.0', 'OIDC', 'SAML', 'OpenID'
        }

# Singleton instance
resume_parser = ResumeParser()

# Public API
async def parse_resume(file_path: str) -> Dict[str, Any]:
    """Parse a resume file and return structured data"""
    return await resume_parser.parse_resume(file_path)