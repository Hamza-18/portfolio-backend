"""
RAG Chatbot Engine
This module implements a Retrieval-Augmented Generation chatbot using:
- Rule-based response generation
- FAISS for vector similarity search
- Sentence Transformers for embedding generation
"""

import os
import json
import pickle
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sys
from util import knowledge_base

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Document:
    """Represents a single document with content and metadata"""
    
    def __init__(self, content: str, metadata: Dict = None, doc_id: str = None, doc_type: str = None):
        """
        Initialize a document
        
        Args:
            content: Text content of the document
            metadata: Optional metadata about the document
            doc_id: Optional document ID
            doc_type: Optional document type
        """
        self.content = content
        self.metadata = metadata or {}
        self.doc_id = doc_id or str(hash(content))[:10]
        self.doc_type = doc_type or metadata.get('source', 'unknown') if metadata else 'unknown'
from dataclasses import dataclass
import logging

# Import knowledge base
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from util.knowledge_base import (
    get_about_me, get_experience, get_projects, 
    skills, education, faqs, contacts, personality
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document in the knowledge base"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str
    doc_type: str  # 'about', 'experience', 'project', 'skill', 'education', 'faq'

class KnowledgeProcessor:
    """Processes and prepares knowledge base for RAG"""
    
    def __init__(self):
        self.documents: List[Document] = []
        
    def prepare_knowledge_base(self) -> List[Document]:
        """Convert knowledge base into documents for RAG"""
        documents = []
        
        # Process about me
        about_data = get_about_me()
        documents.append(Document(
            content=about_data['description'],
            metadata={'section': 'about_me'},
            doc_id='about_me_1',
            doc_type='about'
        ))
        
        # Process experience
        experiences = get_experience()
        for i, exp in enumerate(experiences):
            # Main experience description
            exp_content = f"Position: {exp['position']} at {exp['company']} ({exp['duration']})\n"
            exp_content += "\n".join(exp['description'])
            exp_content += f"\nSkills: {', '.join(exp['skills'])}"
            
            documents.append(Document(
                content=exp_content,
                metadata={
                    'section': 'experience',
                    'company': exp['company'],
                    'position': exp['position'],
                    'duration': exp['duration'],
                    'skills': exp['skills']
                },
                doc_id=f'experience_{i}',
                doc_type='experience'
            ))
            
            # Individual skill documents for better retrieval
            for skill in exp['skills']:
                skill_content = f"Skill: {skill}\nUsed at {exp['company']} as {exp['position']}\n"
                skill_content += f"Context: {' '.join(exp['description'][:2])}"  # First 2 bullet points
                
                documents.append(Document(
                    content=skill_content,
                    metadata={
                        'section': 'skill_experience',
                        'skill': skill,
                        'company': exp['company'],
                        'context': 'work_experience'
                    },
                    doc_id=f'skill_{skill.lower().replace(" ", "_")}_{i}',
                    doc_type='skill'
                ))
        
        # Process projects
        projects = get_projects()
        for i, project in enumerate(projects):
            project_content = f"Project: {project['title']}\n"
            project_content += "\n".join(project['description'])
            project_content += f"\nTechnologies: {', '.join(project['technologies'])}"
            project_content += f"\nCategory: {project['category']}"
            if project.get('githubUrl'):
                project_content += f"\nGitHub: {project['githubUrl']}"
            
            documents.append(Document(
                content=project_content,
                metadata={
                    'section': 'projects',
                    'title': project['title'],
                    'technologies': project['technologies'],
                    'category': project['category'],
                    'featured': project.get('featured', False)
                },
                doc_id=f'project_{i}',
                doc_type='project'
            ))
        
        # Process skills
        for category, skill_list in skills.items():
            skill_content = f"Skill Category: {category.replace('_', ' ').title()}\n"
            skill_content += f"Skills: {', '.join(skill_list)}"
            
            documents.append(Document(
                content=skill_content,
                metadata={
                    'section': 'skills',
                    'category': category,
                    'skills': skill_list
                },
                doc_id=f'skills_{category}',
                doc_type='skill'
            ))
        
        # Process education
        for i, edu in enumerate(education):
            edu_content = f"Education: {edu['degree']} from {edu['university']}"
            if 'graduation_year' in edu:
                edu_content += f" (Class of {edu['graduation_year']})"
            if 'minor' in edu:
                edu_content += f"\nMinor: {edu['minor']}"
            if 'focus' in edu:
                edu_content += f"\nFocus Areas: {', '.join(edu['focus'])}"
            if 'achievements' in edu:
                edu_content += f"\nAchievements: {', '.join(edu['achievements'])}"
            
            documents.append(Document(
                content=edu_content,
                metadata={
                    'section': 'education',
                    'degree': edu['degree'],
                    'university': edu['university']
                },
                doc_id=f'education_{i}',
                doc_type='education'
            ))
        
        # Process FAQs
        for i, faq in enumerate(faqs):
            faq_content = f"Question: {faq['question']}\nAnswer: {faq['answer']}"
            
            documents.append(Document(
                content=faq_content,
                metadata={
                    'section': 'faq',
                    'question': faq['question']
                },
                doc_id=f'faq_{i}',
                doc_type='faq'
            ))
        
        # Process contact information
        contact_content = "Contact Information:\n"
        for key, value in contacts.items():
            contact_content += f"{key.title()}: {value}\n"
        
        documents.append(Document(
            content=contact_content,
            metadata={'section': 'contact'},
            doc_id='contact_info',
            doc_type='contact'
        ))
        
        self.documents = documents
        logger.info(f"Prepared {len(documents)} documents for knowledge base")
        return documents

class VectorStore:
    """Manages vector embeddings and similarity search"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store with sentence transformer model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.documents = []
        self.embeddings = None
        
    def load_model(self):
        """Load the sentence transformer model"""
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.encoder = SentenceTransformer(self.model_name)
        
    def create_embeddings(self, documents: List[Document]):
        """Create embeddings for all documents"""
        if not self.encoder:
            self.load_model()
            
        logger.info("Creating embeddings for documents...")
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        self.embeddings = self.encoder.encode(texts, show_progress_bar=True)
        self.documents = documents
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"Created FAISS index with {len(documents)} documents")
        
    def save_index(self, index_path: str, documents_path: str):
        """Save FAISS index and documents to disk"""
        faiss.write_index(self.index, index_path)
        with open(documents_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"Saved index to {index_path} and documents to {documents_path}")
        
    def load_index(self, index_path: str, documents_path: str):
        """Load FAISS index and documents from disk"""
        if not self.encoder:
            self.load_model()
            
        self.index = faiss.read_index(index_path)
        with open(documents_path, 'rb') as f:
            self.documents = pickle.load(f)
        logger.info(f"Loaded index from {index_path} and documents from {documents_path}")
        
    def search(self, query: str, k: int = 5, score_threshold: float = 0.3) -> List[Dict]:
        """
        Search for relevant documents
        
        Args:
            query: Search query
            k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant documents with scores
        """
        if not self.encoder or not self.index:
            raise ValueError("Model and index must be loaded first")
            
        # Encode query
        query_embedding = self.encoder.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        logger.info(f"FAISS search for '{query}' found {len(indices[0])} results")
        logger.info(f"Scores: {scores[0]}")
        logger.info(f"Indices: {indices[0]}")
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                logger.warning(f"Index {idx} out of range (0-{len(self.documents)-1})")
                continue
                
            if score >= score_threshold:
                doc = self.documents[idx]
                logger.info(f"Match: score={score:.3f}, type={doc.doc_type}, content={doc.content[:50]}...")
                results.append({
                    'document': doc,
                    'score': float(score),
                    'content': doc.content,
                    'metadata': doc.metadata
                })
            else:
                logger.info(f"Below threshold: score={score:.3f} < {score_threshold}")
                
        return results

class RAGChatbot:
    """Retrieval Augmented Generation chatbot for portfolio website"""
    
    def __init__(self, use_llama=False):
        """
        Initialize chatbot with knowledge base
        
        Args:
            use_llama: Flag for Llama integration (kept for compatibility but non-functional)
        """
        self.use_llama = False  # Always set to False regardless of parameter
        self.llama_model = None
        self.tokenizer = None
        
        # Load knowledge directly from the module
        self.about_me = knowledge_base.about_me
        self.experience = knowledge_base.experience
        self.projects = knowledge_base.projects
        self.personality = knowledge_base.personality
        
        # Initialize embedding model and index
        self.load_knowledge_base()
        
        logging.info("RAG Chatbot initialized")
        
    def load_llama3(self):
        """
        Placeholder method for compatibility - no longer loads Llama models
        """
        logging.info("Llama loading skipped - feature disabled")
        self.use_llama = False
        return False
    
    def load_knowledge_base(self):
        """
        Initialize embedding model and create vector store from knowledge
        """
        try:
            # Initialize the embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create document chunks from all knowledge sources
            documents = []
            
            # Add about me info
            documents.append({
                'content': self.about_me['description'],
                'metadata': {'source': 'about_me'}
            })
            
            # Add experience info
            for exp in self.experience:
                exp_text = f"{exp['company']} - {exp['position']} ({exp['duration']})\n"
                exp_text += "\n".join([f"• {item}" for item in exp['description']])
                documents.append({
                    'content': exp_text,
                    'metadata': {'source': 'experience', 'company': exp['company']}
                })
            
            # Add project info
            for proj in self.projects:
                proj_text = f"{proj['title']}: {' '.join(proj['description'])}\n"
                proj_text += f"Technologies: {', '.join(proj['technologies'])}"
                documents.append({
                    'content': proj_text,
                    'metadata': {'source': 'project', 'name': proj['title']}
                })
            
            # Create vector store
            self.create_vector_store(documents)
            logging.info(f"Knowledge base initialized with {len(documents)} documents")
        
        except Exception as e:
            logging.error(f"Error loading knowledge base: {str(e)}")
            raise
    
    def create_vector_store(self, documents_data):
        """
        Create vector store from document data
        
        Args:
            documents_data: List of dictionaries with content and metadata
        """
        # Convert document dictionaries to Document objects
        documents = []
        for i, doc_data in enumerate(documents_data):
            doc_type = doc_data['metadata'].get('source', 'unknown')
            doc_id = f"{doc_type}_{i}"
            doc = Document(
                content=doc_data['content'], 
                metadata=doc_data['metadata'],
                doc_id=doc_id,
                doc_type=doc_type
            )
            documents.append(doc)
        
        # Initialize vector store
        self.vector_store = VectorStore()
        self.vector_store.load_model()
        self.vector_store.create_embeddings(documents)
        logging.info(f"Vector store created with {len(documents)} documents")
            
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve relevant context for the query
        
        Args:
            query: User query
            k: Number of top results to return
            
        Returns:
            String containing relevant context
        """
        if not hasattr(self, 'vector_store') or self.vector_store is None:
            logging.error("Vector store not initialized")
            return "I don't have specific information about that topic in my knowledge base."
            
        try:
            # Use a lower threshold for better recall
            results = self.vector_store.search(query, k=k, score_threshold=0.15)
            
            # Debug log the search results
            logging.info(f"Search results for '{query}': {len(results)} items found")
            for i, result in enumerate(results):
                logging.info(f"Result {i+1}: Score={result['score']:.3f}, Source={result.get('metadata', {}).get('source', 'unknown')}")
            
            if not results:
                return "I don't have specific information about that topic in my knowledge base."
            
            context_parts = []
            for result in results:
                context_parts.append(f"{result['content']}")
                
            return "\n\n".join(context_parts)
        except Exception as e:
            logging.error(f"Error retrieving context: {str(e)}")
            return "I encountered an error retrieving information from my knowledge base."
    
    def generate_llama_response(self, prompt: str) -> str:
        """
        No longer generates response using Llama3 - keeping method for compatibility
        but now just returns None to fall back to rule-based generation
        """
        return None
    def generate_prompt(self, user_query: str, context: str) -> str:
        """
        Generate simplified prompt for response generation (kept for compatibility)
        
        Args:
            user_query: The user's question
            context: Retrieved context from knowledge base
            
        Returns:
            Formatted prompt string
        """
        system_prompt = f"""You are Hamza's personal portfolio chatbot.

Your role is to help visitors learn about Hamza Bashir's background, experience, projects, and skills. 

Context:
{context}

User Question: {user_query}

Answer based on the context above:"""

        return system_prompt

        return system_prompt
    
    def chat(self, user_query: str) -> str:
        """
        Main chat function that retrieves context and generates response
        
        Args:
            user_query: User's question or message
            
        Returns:
            Generated response
        """
        user_query_lower = user_query.lower()
        logging.info(f"Processing query: {user_query_lower}")

        # Handle about/introduction queries for better routing
        about_keywords = ['about', 'introduction', 'overview', 'profile', 'summary', 'describe', 'who is']
        if any(keyword in user_query_lower for keyword in about_keywords):
            # Check if it's not a specific topic query (education, project, etc.)
            specific_topic_keywords = [
                'education', 'project', 'experience', 'contact', 'github', 'linkedin', 
                'skills', 'programming', 'technologies', 'languages', 'work', 'job', 
                'company', 'internship', 'degree', 'university', 'email', 'reach',
                'wavelet', 'poladrone', 'gunfire'
            ]
            if not any(word in user_query_lower for word in specific_topic_keywords):
                logging.info(f"Handling about/introduction request: {user_query}")
                return self._format_about_response()

        # Handle project queries first for better routing
        project_keywords = ['project', 'projects', 'built', 'developed', 'created']
        if any(keyword in user_query_lower for keyword in project_keywords):
            logging.info(f"Handling project request: {user_query}")
            return self._format_project_response()

        # Handle education queries first for better routing
        education_keywords = ['education', 'university', 'college', 'degree', 'school', 'academic']
        if any(keyword in user_query_lower for keyword in education_keywords):
            logging.info(f"Handling education request: {user_query}")
            return self._format_education_response()

        # Handle experience/work queries including company names
        experience_keywords = ['experience', 'work', 'job', 'company', 'internship', 'position', 'career', 'wavelet', 'poladrone', 'gunfire']
        if any(keyword in user_query_lower for keyword in experience_keywords):
            logging.info(f"Handling experience request: {user_query}")
            # Retrieve context for experience and use contextual response
            context = self.retrieve_context(user_query)
            return self._format_experience_response(context)

        # Handle contact information directly
        contact_keywords = ['contact', 'email', 'reach out', 'reach', 'connect', 'social media', 'linkedin', 'github']
        if any(keyword in user_query_lower for keyword in contact_keywords):
            logging.info(f"Handling contact request: {user_query}")
            return self._format_contact_response()

        # Handle skills/programming queries
        skills_keywords = ['skill', 'skills', 'programming', 'technology', 'technologies', 'language', 'languages', 'framework', 'frameworks']
        if any(keyword in user_query_lower for keyword in skills_keywords):
            logging.info(f"Handling skills request: {user_query}")
            # Retrieve context for skills and use contextual response
            context = self.retrieve_context(user_query)
            return self._format_skills_response(context)

        # Handle greetings (with word boundaries to avoid false matches)
        import re
        greeting_keywords = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        # Use more flexible pattern for single words like "hello"
        greeting_pattern = r'(?:^|\s)(?:' + '|'.join(re.escape(kw) for kw in greeting_keywords) + r')(?:\s|$|[!?.])'
        if re.search(greeting_pattern, user_query_lower, re.IGNORECASE):
            greeting = np.random.choice(self.personality['greetings'])
            intro = "I'm Hamza's portfolio chatbot. I can help you learn about his experience, projects, skills, and background. What would you like to know?"
            return f"{greeting} {intro}"
        
        # Handle farewells
        farewell_keywords = ['bye', 'goodbye', 'see you', 'thanks', 'thank you']
        farewell_pattern = r'(?:^|\s)(?:' + '|'.join(re.escape(kw) for kw in farewell_keywords) + r')(?:\s|$|[!?.])'
        if re.search(farewell_pattern, user_query_lower, re.IGNORECASE):
            farewell = np.random.choice(self.personality['farewells'])
            return f"{farewell} Feel free to ask if you have any more questions about Hamza!"
        
        # Retrieve relevant context
        context = self.retrieve_context(user_query)
        
        # We no longer use Llama, so we always use rule-based response generation
        if "I don't have specific information" in context:
            available_topics = [
                "Hamza's work experience at Wavelet and other companies",
                "Technical projects like real-time object detection and AI traffic control",
                "Programming skills and technologies",
                "Education background",
                "Contact information"
            ]
            return f"I don't have specific information about that topic. However, I can help you learn about:\n\n" + "\n".join([f"• {topic}" for topic in available_topics])
        
        # Extract key information from context for response
        return self._generate_contextual_response(user_query, context)
    
    def _generate_contextual_response(self, query: str, context: str) -> str:
        """Generate a response based on retrieved context (fallback when Llama3 is not available)"""
        
        # Simple keyword-based response generation (fallback for when Llama3 is not available)
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['experience', 'work', 'job', 'company']):
            return self._format_experience_response(context)
        elif any(word in query_lower for word in ['project', 'projects', 'built', 'developed']):
            return self._format_project_response(context)
        elif any(word in query_lower for word in ['skill', 'skills', 'technology', 'programming']):
            return self._format_skills_response(context)
        elif any(word in query_lower for word in ['education', 'university', 'degree']):
            return self._format_education_response()
        elif any(word in query_lower for word in ['contact', 'email', 'linkedin', 'reach']):
            return self._format_contact_response()
        else:
            # Generic response using context
            return f"Based on Hamza's background:\n\n{context[:500]}..."
    
    def _format_experience_response(self, context: str) -> str:
        """Format experience-related response"""
        return f"Here's information about Hamza's professional experience:\n\n{context}"
    
    def _format_project_response(self, context: str) -> str:
        """Format project-related response"""
        return f"Hamza has worked on several interesting projects:\n\n{context}"
    
    def _format_skills_response(self, context: str) -> str:
        """Format skills-related response"""
        return f"Here are Hamza's technical skills and expertise:\n\n{context}"
    
    def _format_about_response(self) -> str:
        """Format about me/introduction response"""
        return f"Here's an introduction to Hamza:\n\n{knowledge_base.about_me['description']}"
    
    def _format_project_response(self) -> str:
        """Format project-related response"""
        project_info = []
        for proj in knowledge_base.projects:
            title = proj.get('title', 'N/A')
            description = proj.get('description', [])
            technologies = proj.get('technologies', [])
            category = proj.get('category', 'N/A')
            github_url = proj.get('githubUrl', '')
            
            proj_str = f"• {title} ({category})"
            if description:
                proj_str += f"\n  {' '.join(description)}"
            if technologies:
                proj_str += f"\n  Technologies: {', '.join(technologies)}"
            if github_url:
                proj_str += f"\n  GitHub: {github_url}"
                
            project_info.append(proj_str)
            
        return "Here are some of Hamza's key projects:\n\n" + "\n\n".join(project_info)
    
    def _format_education_response(self) -> str:
        """Format education-related response"""
        education_info = []
        for edu in knowledge_base.education:
            degree = edu.get('degree', 'N/A')
            university = edu.get('university', 'N/A')
            grad_year = edu.get('graduation_year')
            minor = edu.get('minor')
            focus = edu.get('focus', [])
            achievements = edu.get('achievements', [])
            
            edu_str = f"• {degree} from {university}"
            if grad_year:
                edu_str += f" (Class of {grad_year})"
            if minor:
                edu_str += f"\n  Minor: {minor}"
            if focus:
                edu_str += f"\n  Focus Areas: {', '.join(focus)}"
            if achievements:
                edu_str += f"\n  Achievements: {', '.join(achievements)}"
                
            education_info.append(edu_str)
            
        return "Here is Hamza's educational background:\n\n" + "\n\n".join(education_info)
    
    def _format_contact_response(self) -> str:
        """Format contact information response"""
        # Access contacts from knowledge_base module directly
        contacts = knowledge_base.contacts
        contact_info = []
        for key, value in contacts.items():
            contact_info.append(f"• {key.title()}: {value}")
        
        return f"You can reach Hamza through:\n\n" + "\n".join(contact_info)


# Main test function for the module
def main():
    """Main function to test the chatbot"""
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*50)
    print("INITIALIZING RAG CHATBOT")
    print("="*50)
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    print("\n" + "="*50)
    print("TESTING CHATBOT QUERIES")
    print("="*50)
    
    # Test queries
    test_queries = [
        "Hello!",
        "Tell me about Hamza's work experience",
        "What projects has he worked on?",
        "What programming languages does he know?",
        "How can I contact him?",
        "What technologies does he use?",
        "Tell me about his education"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = chatbot.chat(query)
        print(f"Response: {response}")
        print("-" * 80)

if __name__ == "__main__":
    main()
