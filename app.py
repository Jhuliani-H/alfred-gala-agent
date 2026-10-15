"""Gradio chat interface for Alfred.

Run with `python app.py`, then open http://127.0.0.1:7860
"""

from smolagents import GradioUI

from agent import build_alfred

if __name__ == "__main__":
    # share=False keeps the UI local; smolagents defaults to a public share link.
    GradioUI(build_alfred()).launch(share=False)
