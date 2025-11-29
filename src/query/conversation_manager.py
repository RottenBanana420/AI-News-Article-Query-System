"""
Conversation Manager for Multi-turn Dialogues

Manages conversation sessions with history tracking, context window management,
and optional persistence to disk.
"""

import json
import uuid
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation sessions for multi-turn dialogues.
    
    Features:
    - Session creation and management
    - Conversation history tracking with sliding window
    - Optional persistence to JSON files
    - Support for multiple concurrent sessions
    """
    
    def __init__(
        self,
        max_history_turns: int = 10,
        enable_persistence: bool = False,
        storage_dir: Optional[str] = None
    ):
        """
        Initialize the conversation manager.
        
        Args:
            max_history_turns: Maximum number of conversation turns to keep
            enable_persistence: Whether to enable session persistence
            storage_dir: Directory for storing session files
        """
        self.max_history_turns = max_history_turns
        self.enable_persistence = enable_persistence
        
        # Session storage: {session_id: [messages]}
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        
        # Setup storage directory
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            project_root = Path(__file__).parent.parent.parent
            self.storage_dir = project_root / 'data' / 'conversations'
        
        if self.enable_persistence:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def create_session(self) -> str:
        """
        Create a new conversation session.
        
        Returns:
            Unique session ID
        """
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = []
        
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str
    ) -> None:
        """
        Add a conversation turn (question + answer) to a session.
        
        Args:
            session_id: Session identifier
            question: User's question
            answer: Assistant's answer
        """
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found, creating new session")
            self.sessions[session_id] = []
        
        # Add user message
        self.sessions[session_id].append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add assistant message
        self.sessions[session_id].append({
            'role': 'assistant',
            'content': answer,
            'timestamp': datetime.now().isoformat()
        })
        
        # Enforce max history limit (keep last N turns = 2N messages)
        max_messages = self.max_history_turns * 2
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]
        
        logger.debug(f"Added turn to session {session_id}")
    
    def get_history(
        self,
        session_id: str,
        max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            max_turns: Maximum number of turns to return (overrides default)
            
        Returns:
            List of message dictionaries with 'role' and 'content'
        """
        if session_id not in self.sessions:
            return []
        
        history = self.sessions[session_id]
        
        # Apply max_turns limit if specified
        if max_turns is not None:
            max_messages = max_turns * 2
            history = history[-max_messages:]
        
        return history
    
    def format_history_for_prompt(self, session_id: str) -> str:
        """
        Format conversation history as a string for LLM prompts.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Formatted history string
        """
        history = self.get_history(session_id)
        
        if not history:
            return ""
        
        formatted_parts = []
        for message in history:
            role = message['role'].capitalize()
            content = message['content']
            formatted_parts.append(f"{role}: {content}")
        
        return "\n".join(formatted_parts)
    
    def clear_session(self, session_id: str) -> None:
        """
        Clear all history for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.sessions:
            self.sessions[session_id] = []
            logger.info(f"Cleared session {session_id}")
    
    def save_session(self, session_id: str) -> None:
        """
        Save a session to disk.
        
        Args:
            session_id: Session identifier
        """
        if not self.enable_persistence:
            logger.warning("Persistence is not enabled")
            return
        
        if session_id not in self.sessions:
            logger.warning(f"Session {session_id} not found")
            return
        
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            
            session_data = {
                'session_id': session_id,
                'created_at': datetime.now().isoformat(),
                'messages': self.sessions[session_id]
            }
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Saved session {session_id} to {session_file}")
        
        except Exception as e:
            logger.error(f"Error saving session {session_id}: {e}")
    
    def load_session(self, session_id: str) -> bool:
        """
        Load a session from disk.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enable_persistence:
            logger.warning("Persistence is not enabled")
            return False
        
        try:
            session_file = self.storage_dir / f"{session_id}.json"
            
            if not session_file.exists():
                logger.warning(f"Session file not found: {session_file}")
                return False
            
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            self.sessions[session_id] = session_data['messages']
            
            logger.info(f"Loaded session {session_id} from {session_file}")
            return True
        
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding session file {session_id}: {e}")
            return False
        
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return False
