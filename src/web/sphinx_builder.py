"""
Sphinx builder module for generating HTML documentation from Markdown files.
"""

import logging
import os
import shutil
from pathlib import Path
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SphinxBuilder:
    """Build Sphinx documentation from generated Markdown files."""
    
    def __init__(self, source_dir: str = "./generated_docs", 
                 output_dir: str = "./docs"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.build_dir = self.output_dir / "_build" / "html"
        
        # Ensure directories exist
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_sphinx(self):
        """Setup Sphinx configuration."""
        logger.info("Setting up Sphinx...")
        
        # Create conf.py if it doesn't exist
        conf_file = self.output_dir / "conf.py"
        if not conf_file.exists():
            self._create_conf_py(conf_file)
        
        # Create index.rst if it doesn't exist
        index_file = self.output_dir / "index.rst"
        if not index_file.exists():
            self._create_index_rst(index_file)
        
        # Create _static and _templates directories
        (self.output_dir / "_static").mkdir(exist_ok=True)
        (self.output_dir / "_templates").mkdir(exist_ok=True)
    
    def _create_conf_py(self, conf_file: Path):
        """Create Sphinx configuration file."""
        conf_content = '''# -*- coding: utf-8 -*-
#
import sphinx_rtd_theme

extensions = [
    'myst_parser',  # For Markdown support
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

project = 'Generated Documentation'
copyright = '2025'
author = 'Auto-Documentation System'

html_title = 'Project Documentation'
html_short_title = 'Docs'

# Markdown configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
    "linkify",
]
'''
        conf_file.write_text(conf_content, encoding='utf-8')
        logger.info(f"Created Sphinx config at {conf_file}")
    
    def _create_index_rst(self, index_file: Path):
        """Create index.rst file."""
        index_content = '''Generated Documentation
========================

This documentation was automatically generated from the project source code.

Contents
--------

.. toctree::
   :maxdepth: 2
   :glob:

   ../generated_docs/*
'''
        index_file.write_text(index_content, encoding='utf-8')
        logger.info(f"Created index.rst at {index_file}")
    
    def copy_markdown_files(self):
        """Copy generated Markdown files to docs directory."""
        logger.info(f"Copying Markdown files from {self.source_dir}")
        
        # Copy all .md files
        for md_file in self.source_dir.rglob("*.md"):
            rel_path = md_file.relative_to(self.source_dir)
            dest_path = self.output_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(md_file, dest_path)
            logger.info(f"Copied {rel_path}")
    
    def build_html(self):
        """Build HTML documentation using Sphinx."""
        logger.info("Building HTML documentation...")
        
        # Check if required packages are installed
        try:
            import myst_parser
            import sphinx_rtd_theme
        except ImportError:
            logger.error("Required Sphinx extensions not installed.")
            logger.error("Run: pip install myst-parser sphinx-rtd-theme")
            return False
        
        # Run sphinx-build
        try:
            cmd = [
                "sphinx-build",
                "-b", "html",
                str(self.output_dir),
                str(self.build_dir)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Sphinx build failed: {result.stderr}")
                return False
            
            logger.info(f"Documentation built successfully at {self.build_dir}")
            logger.info(f"Open {self.build_dir / 'index.html'} in your browser")
            
            return True
            
        except FileNotFoundError:
            logger.error("sphinx-build command not found. Install Sphinx:")
            logger.error("pip install sphinx")
            return False
        except Exception as e:
            logger.error(f"Error building documentation: {e}")
            return False
    
    def build_all(self):
        """Run the complete Sphinx build process."""
        logger.info("Starting Sphinx documentation build...")
        
        # Setup Sphinx
        self.setup_sphinx()
        
        # Copy markdown files
        self.copy_markdown_files()
        
        # Build HTML
        success = self.build_html()
        
        if success:
            logger.info(f"\n✅ Documentation built successfully!")
            logger.info(f"📁 Location: {self.build_dir}")
            logger.info(f"🌐 Open in browser: file://{self.build_dir.absolute() / 'index.html'}")
        
        return success


def build_sphinx_docs(source_dir: str = "./generated_docs", 
                     output_dir: str = "./docs"):
    """
    Build Sphinx HTML documentation from generated Markdown.
    
    Args:
        source_dir: Directory containing generated Markdown files
        output_dir: Output directory for Sphinx build
    
    Returns:
        True if build succeeded
    """
    builder = SphinxBuilder(source_dir=source_dir, output_dir=output_dir)
    return builder.build_all()

