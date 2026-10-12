"""
Text cleaner for job descriptions.

Strips HTML, removes boilerplate sections (disclaimers, benefits, EEO),
and isolates technical requirements sections.
"""
import re
import logging
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Patterns indicating non-technical boilerplate
BOILERPLATE_PATTERNS = [
    r'equal\s+opportunity\s+employer',
    r'we\s+are\s+an?\s+(?:equal|inclusive)',
    r'(?:diversity|inclusion)\s+(?:statement|policy)',
    r'we\s+do\s+not\s+discriminate',
    r'accommodation\s+for\s+(?:people|individuals)\s+with\s+disabilities',
    r'background\s+check',
    r'drug\s+(?:test|screen)',
    r'benefits?\s+(?:include|package)',
    r'(?:health|dental|vision)\s+insurance',
    r'401\s*\(?k\)?',
    r'paid\s+time\s+off',
    r'unlimited\s+(?:pto|vacation)',
    r'free\s+(?:lunch|snacks|food)',
    r'gym\s+(?:membership|reimbursement)',
    r'company\s+(?:swag|retreats?)',
]

# Section headers that indicate technical requirements
REQUIREMENTS_HEADERS = [
    r'(?:required|minimum|must[\s-]have)\s+(?:skills?|qualifications?|experience)',
    r'requirements?',
    r'qualifications?',
    r'what\s+(?:you\'ll|we)\s+(?:need|expect|look\s+for)',
    r'(?:technical|key)\s+(?:skills?|requirements?|competenc)',
    r'you\s+(?:should|must|will)\s+have',
    r'(?:nice[\s-]to[\s-]have|preferred|bonus)',
    r'tech\s*(?:nical)?\s*stack',
    r'responsibilities',
    r'what\s+you\'ll\s+(?:do|work\s+on)',
]


def clean_html(text: str) -> str:
    """Strip HTML tags and decode entities, preserving meaningful text structure."""
    if not text:
        return ""
    
    soup = BeautifulSoup(text, "html.parser")
    
    # Remove script and style elements
    for element in soup.find_all(["script", "style", "nav", "footer", "header"]):
        element.decompose()
    
    # Get text with newlines preserving block-level structure
    lines = []
    for element in soup.stripped_strings:
        lines.append(element)
    
    text = "\n".join(lines)
    
    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def remove_boilerplate(text: str) -> str:
    """Remove EEO statements, benefits lists, and other non-technical content."""
    if not text:
        return ""
    
    lines = text.split('\n')
    cleaned_lines = []
    skip_section = False
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if this line starts a boilerplate section
        if any(re.search(pattern, line_lower) for pattern in BOILERPLATE_PATTERNS):
            skip_section = True
            continue
        
        # Reset skip if we hit a new section header
        if skip_section and line_lower and (
            line_lower.endswith(':') or 
            re.match(r'^#{1,3}\s+', line) or
            any(re.search(pat, line_lower) for pat in REQUIREMENTS_HEADERS)
        ):
            skip_section = False
        
        if not skip_section:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines).strip()


def extract_requirements_section(text: str) -> str:
    """
    Try to isolate the requirements/qualifications section.
    Falls back to the full cleaned text if no section is found.
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    in_requirements = False
    requirements_lines = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if this line is a requirements header
        if any(re.search(pat, line_lower) for pat in REQUIREMENTS_HEADERS):
            in_requirements = True
            requirements_lines.append(line)
            continue
        
        if in_requirements:
            # Check if we've hit a different section
            if line_lower and (
                re.match(r'^#{1,3}\s+', line) and
                not any(re.search(pat, line_lower) for pat in REQUIREMENTS_HEADERS)
            ):
                # Check if this is a non-requirement section header
                if any(keyword in line_lower for keyword in [
                    'about us', 'about the company', 'benefits', 'perks',
                    'how to apply', 'application', 'salary', 'compensation'
                ]):
                    in_requirements = False
                    continue
            
            requirements_lines.append(line)
    
    if requirements_lines:
        return '\n'.join(requirements_lines).strip()
    
    # Fallback: return entire cleaned text (it may contain inline requirements)
    return text


def clean_description(raw_description: str) -> str:
    """
    Full cleaning pipeline for a job description.
    
    1. Strip HTML
    2. Remove boilerplate
    3. Try to extract requirements section
    
    Returns the cleaned text suitable for skill extraction.
    """
    if not raw_description:
        return ""
    
    # Step 1: Strip HTML
    text = clean_html(raw_description)
    
    # Step 2: Remove boilerplate
    text = remove_boilerplate(text)
    
    # Step 3: Extract requirements section
    requirements = extract_requirements_section(text)
    
    # If requirements section is too short, use the full cleaned text
    if len(requirements) < 100 and len(text) > len(requirements):
        return text
    
    return requirements


if __name__ == "__main__":
    # Test with sample HTML job description
    sample = """
    <div>
    <h2>Senior Python Developer</h2>
    <p>We are looking for an experienced Python developer to join our team.</p>
    <h3>Requirements</h3>
    <ul>
        <li>5+ years of Python experience</li>
        <li>Experience with Django or FastAPI</li>
        <li>PostgreSQL and Redis</li>
        <li>Docker and Kubernetes</li>
        <li>AWS or GCP</li>
    </ul>
    <h3>Benefits</h3>
    <ul>
        <li>Health insurance</li>
        <li>401(k) matching</li>
        <li>Unlimited PTO</li>
    </ul>
    <p>We are an equal opportunity employer.</p>
    </div>
    """
    result = clean_description(sample)
    print("Cleaned output:")
    print(result)
