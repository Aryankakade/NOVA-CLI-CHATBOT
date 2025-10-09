"""
🔗 Simple Shared State Manager for Repository Context
"""
import os
import json
from datetime import datetime

class SimpleRepoState:
    def __init__(self):
        self.state_file = "nova_repo_state.json"
        self._ensure_state()
    
    def _ensure_state(self):
        if not os.path.exists(self.state_file):
            self._save_state({"active_repo": None, "db_path": None})
    
    def _save_state(self, data):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except:
            pass
    
    def _load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except:
            return {"active_repo": None, "db_path": None}
    
    def set_repository(self, repo_url, db_path):
        self._save_state({
            "active_repo": repo_url,
            "db_path": os.path.abspath(db_path),
            "timestamp": datetime.now().isoformat()
        })
        print(f"✅ Repository state saved: {repo_url} -> {db_path}")
    
    def get_repository(self):
        return self._load_state()

# Global instance
repo_state = SimpleRepoState()
