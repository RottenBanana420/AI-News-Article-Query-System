"""
CLI Tests for AI News Article Query System

Tests all CLI commands using subprocess to ensure they work correctly
from the command line.
"""

import pytest
import subprocess
import tempfile
import os
import json
from pathlib import Path


class TestCLICommands:
    """Test CLI command execution."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for CLI tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test URL file
            url_file = os.path.join(tmpdir, 'test_urls.txt')
            with open(url_file, 'w') as f:
                f.write("https://www.bbc.com/news/technology\n")
            
            yield tmpdir, url_file
    
    def run_cli(self, *args):
        """Helper to run CLI commands."""
        cmd = ['python', '-m', 'src.cli'] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result
    
    def test_cli_help(self):
        """Test that CLI help works."""
        result = self.run_cli('--help')
        
        assert result.returncode == 0
        assert 'AI News Article Query System' in result.stdout
        assert 'ingest' in result.stdout
        assert 'query' in result.stdout
        assert 'ask' in result.stdout
    
    def test_cli_no_command(self):
        """Test CLI with no command shows help."""
        result = self.run_cli()
        
        assert result.returncode == 1
        assert 'AI News Article Query System' in result.stdout
    
    def test_ingest_help(self):
        """Test ingest command help."""
        result = self.run_cli('ingest', '--help')
        
        assert result.returncode == 0
        assert '--url' in result.stdout
        assert '--file' in result.stdout
    
    def test_query_help(self):
        """Test query command help."""
        result = self.run_cli('query', '--help')
        
        assert result.returncode == 0
        assert 'query' in result.stdout
        assert '--top-k' in result.stdout
    
    def test_ask_help(self):
        """Test ask command help."""
        result = self.run_cli('ask', '--help')
        
        assert result.returncode == 0
        assert 'question' in result.stdout
        assert '--session' in result.stdout
    
    def test_stats_help(self):
        """Test stats command help."""
        result = self.run_cli('stats', '--help')
        
        assert result.returncode == 0
    
    def test_save_help(self):
        """Test save command help."""
        result = self.run_cli('save', '--help')
        
        assert result.returncode == 0
        assert 'name' in result.stdout
    
    def test_load_help(self):
        """Test load command help."""
        result = self.run_cli('load', '--help')
        
        assert result.returncode == 0
        assert 'name' in result.stdout
    
    def test_list_states_help(self):
        """Test list-states command help."""
        result = self.run_cli('list-states', '--help')
        
        assert result.returncode == 0


class TestCLIIntegration:
    """Integration tests for CLI workflows."""
    
    def run_cli(self, *args):
        """Helper to run CLI commands."""
        cmd = ['python', '-m', 'src.cli'] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # Longer timeout for integration tests
        )
        return result
    
    @pytest.mark.slow
    def test_stats_command(self):
        """Test stats command execution."""
        result = self.run_cli('stats')
        
        assert result.returncode == 0
        assert 'System Statistics' in result.stdout
        assert 'Total Articles' in result.stdout
    
    @pytest.mark.slow
    def test_list_states_empty(self):
        """Test listing states when none exist."""
        result = self.run_cli('list-states')
        
        assert result.returncode == 0
        # Should handle empty state gracefully
    
    @pytest.mark.slow
    def test_ingest_invalid_url(self):
        """Test ingesting an invalid URL fails gracefully."""
        result = self.run_cli('ingest', '--url', 'https://invalid-url-12345.com')
        
        # Should fail but not crash
        assert result.returncode == 1
        assert 'Failed' in result.stdout or 'Error' in result.stdout
    
    @pytest.mark.slow
    def test_query_empty_index(self):
        """Test querying an empty index."""
        result = self.run_cli('query', 'test query')
        
        # Should succeed but return no results
        assert result.returncode == 0
    
    @pytest.mark.slow
    def test_load_nonexistent_state(self):
        """Test loading a non-existent state."""
        result = self.run_cli('load', 'nonexistent_state_12345')
        
        assert result.returncode == 1
        assert 'Failed' in result.stdout or 'Error' in result.stdout


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def run_cli(self, *args):
        """Helper to run CLI commands."""
        cmd = ['python', '-m', 'src.cli'] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result
    
    def test_ingest_missing_arguments(self):
        """Test ingest without URL or file."""
        result = self.run_cli('ingest')
        
        assert result.returncode == 1
        assert 'Error' in result.stdout
    
    def test_ingest_nonexistent_file(self):
        """Test ingest with non-existent file."""
        result = self.run_cli('ingest', '--file', '/nonexistent/file.txt')
        
        assert result.returncode == 1
        assert 'not found' in result.stdout or 'Error' in result.stdout
    
    def test_query_missing_argument(self):
        """Test query without query text."""
        result = self.run_cli('query')
        
        # Should show error or help
        assert result.returncode == 2  # argparse error code
    
    def test_ask_missing_argument(self):
        """Test ask without question."""
        result = self.run_cli('ask')
        
        # Should show error or help
        assert result.returncode == 2  # argparse error code
    
    def test_save_missing_argument(self):
        """Test save without name."""
        result = self.run_cli('save')
        
        # Should show error or help
        assert result.returncode == 2  # argparse error code
    
    def test_load_missing_argument(self):
        """Test load without name."""
        result = self.run_cli('load')
        
        # Should show error or help
        assert result.returncode == 2  # argparse error code


class TestCLIVerboseMode:
    """Test CLI verbose mode."""
    
    def run_cli(self, *args):
        """Helper to run CLI commands."""
        cmd = ['python', '-m', 'src.cli'] + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        return result
    
    def test_verbose_flag(self):
        """Test that verbose flag is accepted."""
        result = self.run_cli('--verbose', 'stats')
        
        # Should not fail
        assert result.returncode == 0
    
    def test_verbose_short_flag(self):
        """Test that -v flag is accepted."""
        result = self.run_cli('-v', 'stats')
        
        # Should not fail
        assert result.returncode == 0
