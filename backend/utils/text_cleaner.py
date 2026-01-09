import re
import string
import html
import unicodedata
from typing import Optional, List, Callable
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

class TextCleaner:
    def __init__(self, 
                 remove_stopwords: bool = True, 
                 lemmatize: bool = True,
                 custom_stopwords: Optional[List[str]] = None):
        """
        Initialize the TextCleaner with configuration options.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            custom_stopwords: Additional stopwords to include
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)
        
        # Common resume/job description specific stopwords
        self.resume_specific_stopwords = {
            'resume', 'cv', 'curriculum vitae', 'contact', 'email', 'phone', 'linkedin',
            'github', 'portfolio', 'references', 'available upon request', 'page', 'http',
            'https', 'www', 'com', 'org', 'net', 'linkedin', 'github', 'github.com'
        }
        self.stop_words.update(self.resume_specific_stopwords)
        
        # Common patterns to remove
        self.patterns = {
            'url': re.compile(r'https?://\S+|www\.\S+'),
            'email': re.compile(r'\S+@\S+'),
            'phone': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'),
            'special_chars': re.compile(r'[^\w\s]'),
            'numbers': re.compile(r'\b\d+\b'),
            'whitespace': re.compile(r'\s+'),
            'single_char': re.compile(r'\b\w\b')
        }

    def clean_text(self, text: str, 
                  steps: Optional[List[str]] = None) -> str:
        """
        Clean and preprocess the input text using specified cleaning steps.
        
        Args:
            text: Input text to clean
            steps: List of cleaning steps to apply. If None, applies all steps.
                   Possible steps: 'lowercase', 'remove_urls', 'remove_emails',
                   'remove_phones', 'remove_special_chars', 'remove_numbers',
                   'remove_stopwords', 'lemmatize', 'normalize_whitespace',
                   'remove_single_chars'
        
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Default steps if none provided
        if steps is None:
            steps = [
                'lowercase',
                'remove_urls',
                'remove_emails',
                'remove_phones',
                'remove_special_chars',
                'remove_numbers',
                'remove_stopwords',
                'lemmatize',
                'normalize_whitespace',
                'remove_single_chars'
            ]
        
        # Apply each cleaning step
        for step in steps:
            if step == 'lowercase':
                text = text.lower()
            elif step == 'remove_urls':
                text = self._remove_pattern(text, self.patterns['url'])
            elif step == 'remove_emails':
                text = self._remove_pattern(text, self.patterns['email'])
            elif step == 'remove_phones':
                text = self._remove_pattern(text, self.patterns['phone'])
            elif step == 'remove_special_chars':
                text = self._remove_special_chars(text)
            elif step == 'remove_numbers':
                text = self.patterns['numbers'].sub(' ', text)
            elif step == 'remove_stopwords' and self.remove_stopwords:
                text = self._remove_stopwords(text)
            elif step == 'lemmatize' and self.lemmatize:
                text = self._lemmatize_text(text)
            elif step == 'normalize_whitespace':
                text = self.patterns['whitespace'].sub(' ', text).strip()
            elif step == 'remove_single_chars':
                text = self.patterns['single_char'].sub('', text)
                text = self.patterns['whitespace'].sub(' ', text).strip()
        
        return text

    def clean_resume_text(self, text: str) -> str:
        """
        Specialized cleaning for resume text.
        Removes sections, headers, and other resume-specific noise.
        """
        if not text:
            return ""
            
        # Remove common resume section headers
        section_headers = [
            r'(?i)\b(?:professional\s+summary|summary|experience|work\s+history|'
            r'education|certifications|skills|projects|technical\s+skills|'
            r'professional\s+experience|work\s+experience|internship|awards|'
            r'publications|volunteer|references|additional\s+information)\s*:?\s*\n'
        ]
        
        for pattern in section_headers:
            text = re.sub(pattern, '\n', text, flags=re.IGNORECASE)
        
        # Clean the text using standard cleaning steps
        return self.clean_text(text)

    def clean_job_description(self, text: str) -> str:
        """
        Specialized cleaning for job descriptions.
        Removes company boilerplate and focuses on requirements and responsibilities.
        """
        if not text:
            return ""
            
        # Remove common job description boilerplate
        boilerplate_phrases = [
            r'(?i)about\s+us:.*?(?=\n\n|\n\w)',
            r'(?i)company\s+description:.*?(?=\n\n|\n\w)',
            r'(?i)equal\s+opportunity\s+employer.*',
            r'(?i)we\s+are\s+an\s+equal\s+opportunity\s+employer.*',
            r'(?i)all\s+qualified\s+applicants.*',
        ]
        
        for pattern in boilerplate_phrases:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Clean the text using standard cleaning steps
        return self.clean_text(text)

    def tokenize(self, text: str, clean: bool = True) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            clean: Whether to clean the text before tokenization
            
        Returns:
            List of tokens
        """
        if clean:
            text = self.clean_text(text)
        return word_tokenize(text)

    def _remove_pattern(self, text: str, pattern: re.Pattern) -> str:
        """Helper method to remove patterns from text."""
        return pattern.sub(' ', text)

    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters but keep basic punctuation."""
        # Keep basic sentence punctuation
        text = re.sub(r'[^\w\s.,;:!?]', ' ', text)
        # Normalize unicode characters
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        return text

    def _remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text."""
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)

    def _lemmatize_text(self, text: str) -> str:
        """Lemmatize words in the text."""
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def get_keywords(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract top N keywords from text based on frequency.
        
        Args:
            text: Input text
            top_n: Number of top keywords to return
            
        Returns:
            List of top keywords
        """
        from collections import Counter
        
        # Clean and tokenize the text
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize(cleaned_text, clean=False)
        
        # Count word frequencies
        word_freq = Counter(tokens)
        
        # Get top N words
        return [word for word, _ in word_freq.most_common(top_n)]

# Create a default instance for convenience
default_cleaner = TextCleaner()

def clean_text(text: str, **kwargs) -> str:
    """
    Convenience function to clean text using default settings.
    
    Args:
        text: Input text to clean
        **kwargs: Additional arguments to pass to TextCleaner.clean_text()
        
    Returns:
        Cleaned text
    """
    return default_cleaner.clean_text(text, **kwargs)