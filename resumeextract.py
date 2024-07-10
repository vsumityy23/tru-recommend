from utility import skills,tools,certifications
import fitz  
import spacy
from spacy.matcher import PhraseMatcher

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to read PDF and extract text
def extract_text_from_pdf(pdf_content):
    doc = fitz.open(stream=pdf_content, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to create a matcher for a list of phrases
def create_phrase_matcher(nlp, phrases):
    matcher = PhraseMatcher(nlp.vocab)
    patterns = [nlp(text) for text in phrases]
    matcher.add("PhraseMatcher", None, *patterns)
    return matcher

# Function to extract matches from text
def extract_matches(text, matcher):
    doc = nlp(text)
    matches = matcher(doc)
    extracted = [doc[start:end].text for match_id, start, end in matches]
    return extracted

# Main extraction function
def extract_resume_info(pdf_content):
    text = extract_text_from_pdf(pdf_content)
    
    # Create matchers
    skill_matcher = create_phrase_matcher(nlp, skills)
    tool_matcher = create_phrase_matcher(nlp, tools)
    cert_matcher = create_phrase_matcher(nlp, certifications)
    
    # Extract information
    extracted_skills = extract_matches(text, skill_matcher)
    extracted_tools = extract_matches(text, tool_matcher)
    extracted_certs = extract_matches(text, cert_matcher)
    
    return {
        "skills": list(set(extracted_skills)),
        "tools": list(set(extracted_tools)),
        "certifications": list(set(extracted_certs))
    }
