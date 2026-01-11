import json
import os
import shutil
from datetime import datetime

class ProjectManager:
    """
    Handles saving and loading of Dalil AI project states.
    A project is essentially a JSON file containing metadata and configuration.
    """
    
    @staticmethod
    def save_project(state: dict, filepath: str) -> bool:
        """
        Saves the current state dictionary to a .dalil (JSON) file.
        """
        try:
            # Add metadata
            state['saved_at'] = datetime.now().isoformat()
            state['version'] = "2.0"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=4)
            return True
        except Exception as e:
            print(f"Error saving project: {e}")
            return False

    @staticmethod
    def load_project(filepath: str) -> dict:
        """
        Loads a state dictionary from a .dalil file.
        """
        try:
            if not os.path.exists(filepath):
                return None
                
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            return state
        except Exception as e:
            print(f"Error loading project: {e}")
            return {}

    @staticmethod
    def get_recent_projects(limit=5):
        """
        (Placeholder) intended to read a local config file 
        tracking recently opened paths.
        """
        # In a real impl, we'd read from a standard config location
        # e.g. ~/.dalil_ai/config.json
        return []
