"""
Documentation generation module using Hugging Face's Gemma model to generate
detailed, software-level documentation for code chunks.
"""

import logging
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GemmaDocGenerator:
    """Generate documentation using Hugging Face's Gemma model."""
    
    # Comprehensive documentation prompt template
    DOC_PROMPT_TEMPLATE = """You are an **expert software engineer and technical documentation specialist**.  
Your task is to produce **software-level documentation** for the given code chunk and its related context (retrieved from a knowledge base).  
The documentation must be **detailed, technically sound, and developer-friendly**, intended for professional-level project handover, onboarding, and architectural understanding.

---

### 🔍 INPUT CONTEXT
You are given:
1. A **primary code chunk** from a Python project.
2. **5 retrieved related chunks** from a knowledge base (CodexGLUE or similar) for contextual understanding.
3. Metadata about file paths, function names, and dependencies.

Use this information to infer the **design intent, function purpose, and architectural relationship** between components.

---

### 🧾 OUTPUT FORMAT
All output should be in **structured Markdown** and contain the following sections (skip any not applicable):

#### 1. Introduction
- Summarize what the given file or code chunk does.
- Explain its **role** within the overall system.
- Highlight its **dependencies, purpose, and integration points** with other modules.

#### 2. Function and Class Documentation
For each function/class defined:
- **Name:** Function or class name.
- **Purpose:** Its goal and role in the system.
- **Parameters:** Describe each parameter, including type, validation, and edge cases.
- **Returns:** Explain what is returned and under what conditions.
- **Exceptions:** Note any handled or unhandled exceptions.
- **Algorithm / Logic Overview:** Describe the implementation flow in plain English.
- **Code Quality Evaluation:**
  - Is it efficient and maintainable?
  - Can it be simplified?
  - Does it adhere to principles like SRP, OCP, and Dependency Injection?
  - Are there redundant or unsafe patterns?

#### 3. Inter-Module Relationships
- Explain **how this chunk interacts with other components**.
- Document **function call hierarchies** (use the provided call graph context if available).
- Identify **shared dependencies**, imports, or data flows.

#### 4. Design & Architecture Insights
- What **design patterns** are being used (e.g., Singleton, Factory, MVC)?
- Are they implemented correctly?
- Are there **alternative patterns or frameworks** that would improve clarity, performance, or scalability?
- Does the abstraction level fit the module's purpose?

#### 5. Performance & Optimization Notes
- Are there performance bottlenecks or unnecessary computations?
- Could algorithmic complexity be improved?
- Suggest measurable performance optimizations.

#### 6. Security & Reliability
- Note potential **security vulnerabilities** (e.g., hardcoded credentials, unsafe I/O).
- Discuss **data validation**, **error handling**, and **exception safety**.
- Suggest secure coding or input sanitization improvements.

#### 7. Maintainability & Scalability
- Evaluate **modularity**, **readability**, and **extensibility**.
- Identify code that violates **separation of concerns**.
- Suggest how the module can be adapted for future scaling.

#### 8. Testing & QA Recommendations
- Suggest **unit test cases** for key functions.
- Recommend **mocking/stubbing strategies** for external dependencies.
- Identify untested branches or error conditions.

#### 9. Environment & Dependencies
- List dependencies, frameworks, and APIs used.
- Explain why they are used, and suggest alternatives if more optimal ones exist.
- Highlight any **unwanted compile-time or runtime dependencies**.

#### 10. File-Level Summary
- Describe how this file contributes to the entire project.
- Note **integration points** (e.g., database, external services, internal APIs).
- Provide a **one-paragraph summary** of the file's purpose.

---

## Additional Instructions
- Use **clear subheadings**, bullet points, and formatting to improve readability.
- Prefer **technical precision** with beginner-friendly tone.
- Avoid restating code verbatim; instead, **explain what it does and why**.
- Include **architecture diagrams or pseudocode snippets** when needed (in text form).
- Always generate Markdown-safe output.

---

## Input Code Chunk and Context:

{code_context}

---

Please generate the documentation following the structure above.
"""
    
    def __init__(self, token: Optional[str] = None, model_name: str = "google/gemma-2b"):
        """
        Initialize the Gemma documentation generator.
        
        Args:
            token: Hugging Face token (or use HF_TOKEN env var)
            model_name: Gemma model name to use
        """
        self.token = token or os.getenv("HF_TOKEN")
        
        if not self.token:
            logger.warning("Hugging Face token not found. Some features may not work.")
            logger.warning("Set HF_TOKEN environment variable or pass token parameter.")
        
        self.model_name = model_name
        
        # Initialize the model and tokenizer
        try:
            logger.info(f"Loading Gemma model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.token,
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.token,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info("✓ Successfully loaded Gemma model")
        except Exception as e:
            logger.error(f"Failed to initialize Gemma model: {e}")
            logger.error("Make sure you have installed transformers and torch")
            raise
    
    def generate_documentation(self, code_chunk: str, context_chunks: List[Dict[str, Any]] = None) -> str:
        """
        Generate documentation for a code chunk using Gemma.
        
        Args:
            code_chunk: The code to document
            context_chunks: List of related context chunks from RAG
        
        Returns:
            Generated documentation in Markdown format
        """
        # Prepare context
        context_text = f"```python\n{code_chunk}\n```"
        
        if context_chunks:
            context_text += "\n\n## Related Context:\n\n"
            for i, ctx in enumerate(context_chunks, 1):
                context_text += f"### Context {i}\n\n"
                context_text += f"```python\n{ctx.get('text', '')[:500]}\n```\n\n"
        
        # Format prompt
        prompt = self.DOC_PROMPT_TEMPLATE.format(code_context=context_text)
        
        # Generate documentation using Gemma
        try:
            logger.info("Generating documentation with Gemma...")
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    top_k=50
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (remove prompt)
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            if not generated_text:
                logger.warning("Empty response from Gemma")
                return "# Documentation\n\nNo documentation generated."
            
            logger.info("✓ Successfully generated documentation")
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating documentation with Gemma: {e}")
            return f"# Documentation\n\nError generating documentation: {e}"
    
    def generate_file_documentation(self, file_path: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generate documentation for an entire file from its chunks.
        
        Args:
            file_path: Path to the file
            chunks: List of chunk dictionaries
        
        Returns:
            Complete file documentation
        """
        logger.info(f"Generating documentation for {file_path}")
        
        # Combine all chunks for this file
        file_text = "\n".join([chunk['text'] for chunk in chunks])
        
        # Generate documentation
        doc = self.generate_documentation(file_text)
        
        # Add file header
        file_header = f"# File: {file_path}\n\n"
        
        return file_header + doc
    
    def generate_project_overview(self, project_path: str, file_docs: List[Dict[str, str]]) -> str:
        """
        Generate project-level overview documentation.
        
        Args:
            project_path: Path to project root
            file_docs: List of file documentation dictionaries
        
        Returns:
            Project overview documentation
        """
        overview_template = """# Project Overview

## Purpose
{project_purpose}

## Architecture
{architecture_summary}

## Installation
{installation_steps}

## Structure
{project_structure}

## Known Issues and Improvements
{issues}

## Recommendations
{recommendations}
"""
        
        # Generate placeholders (could be enhanced with Gemini)
        project_purpose = f"This project is located at {project_path}"
        architecture_summary = "Project consists of multiple modules working together."
        installation_steps = "pip install -r requirements.txt"
        
        project_structure = "\n".join([f"- {doc.get('filepath', 'unknown')}" for doc in file_docs])
        issues = "None identified at this time."
        recommendations = "Follow best practices for code organization and documentation."
        
        return overview_template.format(
            project_purpose=project_purpose,
            architecture_summary=architecture_summary,
            installation_steps=installation_steps,
            project_structure=project_structure,
            issues=issues,
            recommendations=recommendations
        )


def generate_docs_for_chunk(chunk: Dict[str, Any], context_chunks: List[Dict[str, Any]] = None, token: Optional[str] = None) -> str:
    """
    Generate documentation for a single chunk using Gemma.
    
    Args:
        chunk: Chunk dictionary
        context_chunks: Related context from RAG
        token: Hugging Face token
    
    Returns:
        Generated documentation
    """
    generator = GemmaDocGenerator(token=token)
    return generator.generate_documentation(chunk['text'], context_chunks)
