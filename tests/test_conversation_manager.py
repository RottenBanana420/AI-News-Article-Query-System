"""
Comprehensive Test Suite for Conversation Manager

These tests are designed to FAIL initially and drive the implementation.
Tests are NEVER modified - only the implementation code is updated to pass them.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil

from src.query.conversation_manager import ConversationManager


class TestConversationManagerInitialization:
    """Test conversation manager initialization."""
    
    def test_initialization_with_defaults(self):
        """Test initialization with default parameters."""
        manager = ConversationManager()
        
        assert manager.max_history_turns == 10
        assert manager.enable_persistence is False
        assert manager.sessions == {}
    
    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConversationManager(
                max_history_turns=5,
                enable_persistence=True,
                storage_dir=tmpdir
            )
            
            assert manager.max_history_turns == 5
            assert manager.enable_persistence is True
            assert str(manager.storage_dir) == tmpdir


class TestSessionManagement:
    """Test session creation and management."""
    
    def test_create_session(self):
        """Test creating a new session."""
        manager = ConversationManager()
        
        session_id = manager.create_session()
        
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        assert session_id in manager.sessions
    
    def test_create_multiple_sessions(self):
        """Test creating multiple independent sessions."""
        manager = ConversationManager()
        
        session_id1 = manager.create_session()
        session_id2 = manager.create_session()
        
        assert session_id1 != session_id2
        assert session_id1 in manager.sessions
        assert session_id2 in manager.sessions
    
    def test_session_id_uniqueness(self):
        """Test that session IDs are unique."""
        manager = ConversationManager()
        
        session_ids = set()
        for _ in range(100):
            session_id = manager.create_session()
            assert session_id not in session_ids
            session_ids.add(session_id)


class TestConversationHistory:
    """Test conversation history management."""
    
    def test_add_turn_to_session(self):
        """Test adding a conversation turn."""
        manager = ConversationManager()
        session_id = manager.create_session()
        
        manager.add_turn(
            session_id,
            question="What is AI?",
            answer="AI is artificial intelligence."
        )
        
        history = manager.get_history(session_id)
        
        assert len(history) == 2  # Question + answer
        assert history[0]['role'] == 'user'
        assert history[0]['content'] == "What is AI?"
        assert history[1]['role'] == 'assistant'
        assert history[1]['content'] == "AI is artificial intelligence."
    
    def test_add_multiple_turns(self):
        """Test adding multiple conversation turns."""
        manager = ConversationManager()
        session_id = manager.create_session()
        
        manager.add_turn(session_id, "Question 1", "Answer 1")
        manager.add_turn(session_id, "Question 2", "Answer 2")
        manager.add_turn(session_id, "Question 3", "Answer 3")
        
        history = manager.get_history(session_id)
        
        assert len(history) == 6  # 3 questions + 3 answers
        assert history[0]['content'] == "Question 1"
        assert history[1]['content'] == "Answer 1"
        assert history[4]['content'] == "Question 3"
    
    def test_history_ordering(self):
        """Test that history maintains chronological order."""
        manager = ConversationManager()
        session_id = manager.create_session()
        
        for i in range(5):
            manager.add_turn(session_id, f"Q{i}", f"A{i}")
        
        history = manager.get_history(session_id)
        
        # Should be in order: Q0, A0, Q1, A1, ...
        for i in range(5):
            assert history[i * 2]['content'] == f"Q{i}"
            assert history[i * 2 + 1]['content'] == f"A{i}"
    
    def test_max_history_turns_enforcement(self):
        """Test that history respects max_history_turns limit."""
        manager = ConversationManager(max_history_turns=3)
        session_id = manager.create_session()
        
        # Add 5 turns (10 messages)
        for i in range(5):
            manager.add_turn(session_id, f"Q{i}", f"A{i}")
        
        history = manager.get_history(session_id)
        
        # Should only keep last 3 turns (6 messages)
        assert len(history) == 6
        assert history[0]['content'] == "Q2"  # Oldest should be Q2
        assert history[-1]['content'] == "A4"  # Newest should be A4
    
    def test_get_history_with_max_turns_parameter(self):
        """Test retrieving history with custom max_turns."""
        manager = ConversationManager(max_history_turns=10)
        session_id = manager.create_session()
        
        for i in range(5):
            manager.add_turn(session_id, f"Q{i}", f"A{i}")
        
        # Get only last 2 turns
        history = manager.get_history(session_id, max_turns=2)
        
        assert len(history) == 4  # 2 turns = 4 messages
        assert history[0]['content'] == "Q3"
    
    def test_get_history_nonexistent_session(self):
        """Test getting history for nonexistent session."""
        manager = ConversationManager()
        
        history = manager.get_history("nonexistent_session_id")
        
        assert history == []


class TestHistoryFormatting:
    """Test conversation history formatting for prompts."""
    
    def test_format_history_for_prompt(self):
        """Test formatting history as a string for LLM prompts."""
        manager = ConversationManager()
        session_id = manager.create_session()
        
        manager.add_turn(session_id, "What is AI?", "AI is artificial intelligence.")
        manager.add_turn(session_id, "What about ML?", "ML is machine learning.")
        
        formatted = manager.format_history_for_prompt(session_id)
        
        assert isinstance(formatted, str)
        assert "What is AI?" in formatted
        assert "AI is artificial intelligence." in formatted
        assert "What about ML?" in formatted
        assert "ML is machine learning." in formatted
    
    def test_format_empty_history(self):
        """Test formatting when no history exists."""
        manager = ConversationManager()
        session_id = manager.create_session()
        
        formatted = manager.format_history_for_prompt(session_id)
        
        assert formatted == ""


class TestSessionPersistence:
    """Test session persistence to disk."""
    
    def test_save_session(self):
        """Test saving a session to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConversationManager(
                enable_persistence=True,
                storage_dir=tmpdir
            )
            
            session_id = manager.create_session()
            manager.add_turn(session_id, "Test question", "Test answer")
            
            manager.save_session(session_id)
            
            # Check that file was created
            session_file = Path(tmpdir) / f"{session_id}.json"
            assert session_file.exists()
    
    def test_load_session(self):
        """Test loading a session from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save a session
            manager1 = ConversationManager(
                enable_persistence=True,
                storage_dir=tmpdir
            )
            session_id = manager1.create_session()
            manager1.add_turn(session_id, "Question 1", "Answer 1")
            manager1.add_turn(session_id, "Question 2", "Answer 2")
            manager1.save_session(session_id)
            
            # Create new manager and load session
            manager2 = ConversationManager(
                enable_persistence=True,
                storage_dir=tmpdir
            )
            success = manager2.load_session(session_id)
            
            assert success is True
            assert session_id in manager2.sessions
            
            history = manager2.get_history(session_id)
            assert len(history) == 4
            assert history[0]['content'] == "Question 1"
            assert history[3]['content'] == "Answer 2"
    
    def test_load_nonexistent_session(self):
        """Test loading a session that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConversationManager(
                enable_persistence=True,
                storage_dir=tmpdir
            )
            
            success = manager.load_session("nonexistent_id")
            
            assert success is False
    
    def test_load_corrupted_session_file(self):
        """Test handling of corrupted session files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConversationManager(
                enable_persistence=True,
                storage_dir=tmpdir
            )
            
            # Create corrupted file
            session_id = "corrupted_session"
            session_file = Path(tmpdir) / f"{session_id}.json"
            session_file.write_text("invalid json {{{")
            
            success = manager.load_session(session_id)
            
            assert success is False


class TestSessionClearance:
    """Test clearing session data."""
    
    def test_clear_session(self):
        """Test clearing a session's history."""
        manager = ConversationManager()
        session_id = manager.create_session()
        
        manager.add_turn(session_id, "Question", "Answer")
        assert len(manager.get_history(session_id)) == 2
        
        manager.clear_session(session_id)
        
        history = manager.get_history(session_id)
        assert len(history) == 0
    
    def test_clear_nonexistent_session(self):
        """Test clearing a session that doesn't exist."""
        manager = ConversationManager()
        
        # Should not raise an error
        manager.clear_session("nonexistent_id")


class TestConcurrentSessions:
    """Test managing multiple concurrent sessions."""
    
    def test_independent_session_histories(self):
        """Test that different sessions maintain independent histories."""
        manager = ConversationManager()
        
        session1 = manager.create_session()
        session2 = manager.create_session()
        
        manager.add_turn(session1, "Session 1 Q1", "Session 1 A1")
        manager.add_turn(session2, "Session 2 Q1", "Session 2 A1")
        manager.add_turn(session1, "Session 1 Q2", "Session 1 A2")
        
        history1 = manager.get_history(session1)
        history2 = manager.get_history(session2)
        
        assert len(history1) == 4
        assert len(history2) == 2
        assert history1[0]['content'] == "Session 1 Q1"
        assert history2[0]['content'] == "Session 2 Q1"
    
    def test_multiple_sessions_persistence(self):
        """Test saving and loading multiple sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager1 = ConversationManager(
                enable_persistence=True,
                storage_dir=tmpdir
            )
            
            # Create multiple sessions
            session_ids = []
            for i in range(3):
                session_id = manager1.create_session()
                session_ids.append(session_id)
                manager1.add_turn(session_id, f"Q{i}", f"A{i}")
                manager1.save_session(session_id)
            
            # Load all sessions in new manager
            manager2 = ConversationManager(
                enable_persistence=True,
                storage_dir=tmpdir
            )
            
            for i, session_id in enumerate(session_ids):
                success = manager2.load_session(session_id)
                assert success is True
                
                history = manager2.get_history(session_id)
                assert len(history) == 2
                assert history[0]['content'] == f"Q{i}"
