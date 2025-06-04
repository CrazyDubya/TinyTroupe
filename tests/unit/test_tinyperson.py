import pytest
import logging
logger = logging.getLogger("tinytroupe")

import sys
sys.path.insert(0, '../../tinytroupe/') # ensures that the package is imported from the parent directory, not the Python installation
sys.path.insert(0, '../../') # ensures that the package is imported from the parent directory, not the Python installation
sys.path.insert(0, '..') # ensures that the package is imported from the parent directory, not the Python installation

from tinytroupe.agent.tiny_person import TinyPerson
from tinytroupe.agent.memory import EpisodicMemory, SemanticMemory
from tinytroupe.examples import create_oscar_the_architect, create_lisa_the_data_scientist, create_oscar_the_architect_2, create_lisa_the_data_scientist_2
import tinytroupe.openai_utils as openai_utils # For mocking

from testing_utils import * # Assumes get_relative_to_test_path, EXPORT_BASE_FOLDER etc. are here
import os # For test_save_specification

# Imports for new tests
import unittest
from unittest.mock import patch, MagicMock, call
import datetime

# Existing tests ... (truncated for brevity in thought process, but will be in the final file)

def test_act(setup):

    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:

        actions = agent.listen_and_act("Tell me a bit about your life.", return_actions=True)

        logger.info(agent.pp_current_interactions())

        assert len(actions) >= 1, f"{agent.name} should have at least one action to perform (even if it is just DONE)."
        assert contains_action_type(actions, "TALK"), f"{agent.name} should have at least one TALK action to perform, since we asked him to do so."
        assert terminates_with_action_type(actions, "DONE"), f"{agent.name} should always terminate with a DONE action."

def test_listen(setup):
    # test that the agent listens to a speech stimulus and updates its current messages
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.listen("Hello, how are you?")

        assert len(agent.current_messages) > 0, f"{agent.name} should have at least one message in its current messages."
        assert agent.episodic_memory.retrieve_all()[-1]['role'] == 'user', f"{agent.name} should have the last message as 'user'."
        assert agent.episodic_memory.retrieve_all()[-1]['content']['stimuli'][0]['type'] == 'CONVERSATION', f"{agent.name} should have the last message as a 'CONVERSATION' stimulus."
        assert agent.episodic_memory.retrieve_all()[-1]['content']['stimuli'][0]['content'] == 'Hello, how are you?', f"{agent.name} should have the last message with the correct content."

def test_define(setup):
    # test that the agent defines a value to its configuration and resets its prompt
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        # save the original prompt
        original_prompt = agent.current_messages[0]['content']

        # define a new value
        agent.define('age', 25)

        # check that the configuration has the new value
        assert agent._persona['age'] == 25, f"{agent.name} should have the age set to 25."

        # check that the prompt has changed
        assert agent.current_messages[0]['content'] != original_prompt, f"{agent.name} should have a different prompt after defining a new value."

        # check that the prompt contains the new value
        assert '25' in agent.current_messages[0]['content'], f"{agent.name} should have the age in the prompt."

def test_define_several(setup):
    # Test that defining several values to a group works as expected
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.define("skills", ["Python", "Machine learning", "GPT-3"])
        
        assert "Python" in agent._persona["skills"], f"{agent.name} should have Python as a skill."
        assert "Machine learning" in agent._persona["skills"], f"{agent.name} should have Machine learning as a skill."
        assert "GPT-3" in agent._persona["skills"], f"{agent.name} should have GPT-3 as a skill."

def test_socialize(setup):
    # Test that socializing with another agent works as expected
    an_oscar = create_oscar_the_architect()
    a_lisa = create_lisa_the_data_scientist()
    for agent in [an_oscar, a_lisa]:
        other = a_lisa if agent.name == "Oscar" else an_oscar
        agent.make_agent_accessible(other, relation_description="My friend")
        agent.listen(f"Hi {agent.name}, I am {other.name}.")
        actions = agent.act(return_actions=True)
        assert len(actions) >= 1, f"{agent.name} should have at least one action to perform."
        assert contains_action_type(actions, "TALK"), f"{agent.name} should have at least one TALK action to perform, since we started a conversation."
        assert contains_action_content(actions, agent_first_name(other)), f"{agent.name} should mention {other.name}'s first name in the TALK action, since they are friends."

def test_see(setup):
    # Test that seeing a visual stimulus works as expected
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.see("A beautiful sunset over the ocean.")
        actions = agent.act(return_actions=True)
        assert len(actions) >= 1, f"{agent.name} should have at least one action to perform."
        assert contains_action_type(actions, "THINK"), f"{agent.name} should have at least one THINK action to perform, since they saw something interesting."
        assert contains_action_content(actions, "sunset"), f"{agent.name} should mention the sunset in the THINK action, since they saw it."

def test_think(setup):
    # Test that thinking about something works as expected
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.think("I will tell everyone right now how awesome life is!")
        actions = agent.act(return_actions=True)
        assert len(actions) >= 1, f"{agent.name} should have at least one action to perform."
        assert contains_action_type(actions, "TALK"), f"{agent.name} should have at least one TALK action to perform, since they are eager to talk."
        assert contains_action_content(actions, "life"), f"{agent.name} should mention life in the TALK action, since they thought about it."

def test_internalize_goal(setup):
    # Test that internalizing a goal works as expected
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.internalize_goal("I want to compose in my head a wonderful poem about how cats are glorious creatures.")
        actions = agent.act(return_actions=True)
        assert len(actions) >= 1, f"{agent.name} should have at least one action to perform."
        assert contains_action_type(actions, "THINK"), f"{agent.name} should have at least one THINK action to perform, since they internalized a goal."
        assert contains_action_content(actions, "cats"), f"{agent.name} should mention cats in the THINK action, since they internalized a goal about them."


def test_move_to(setup):
    # Test that moving to a new location works as expected
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.move_to("New York", context=["city", "busy", "diverse"])
        assert agent._mental_state["location"] == "New York", f"{agent.name} should have New York as the current location."
        assert "city" in agent._mental_state["context"], f"{agent.name} should have city as part of the current context."
        assert "busy" in agent._mental_state["context"], f"{agent.name} should have busy as part of the current context."
        assert "diverse" in agent._mental_state["context"], f"{agent.name} should have diverse as part of the current context."

def test_change_context(setup):
    # Test that changing the context works as expected
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.change_context(["home", "relaxed", "comfortable"])
        assert "home" in agent._mental_state["context"], f"{agent.name} should have home as part of the current context."
        assert "relaxed" in agent._mental_state["context"], f"{agent.name} should have relaxed as part of the current context."
        assert "comfortable" in agent._mental_state["context"], f"{agent.name} should have comfortable as part of the current context."

def test_save_specification(setup):   
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        # save to a file
        agent.save_specification(get_relative_to_test_path(f"{EXPORT_BASE_FOLDER}/serialization/{agent.name}.tinyperson.json"), include_memory=True)

        # check that the file exists
        assert os.path.exists(get_relative_to_test_path(f"{EXPORT_BASE_FOLDER}/serialization/{agent.name}.tinyperson.json")), f"{agent.name} should have saved the file."

        # load the file to see if the agent is the same. The agent name should be different because it TinyTroupe does not allow two agents with the same name.
        loaded_name = f"{agent.name}_loaded"
        loaded_agent = TinyPerson.load_specification(get_relative_to_test_path(f"{EXPORT_BASE_FOLDER}/serialization/{agent.name}.tinyperson.json"), new_agent_name=loaded_name)

        # check that the loaded agent is the same as the original
        assert loaded_agent.name == loaded_name, f"{agent.name} should have the same name as the loaded agent."
        
        assert agents_personas_are_equal(agent, loaded_agent, ignore_name=True), f"{agent.name} should have the same configuration as the loaded agent, except for the name."
        
def test_programmatic_definitions(setup):
    for agent in [create_oscar_the_architect_2(), create_lisa_the_data_scientist_2()]:
        agent.listen_and_act("Tell me a bit about your life.")


# New Test Class for _extract_and_store_semantic_insight
class TestTinyPersonSemanticExtraction(unittest.TestCase):

    def setUp(self):
        # Basic TinyPerson setup for these specific tests
        # We can use a real SemanticMemory instance as we're mocking the LLM call
        # and the actual store method of SemanticMemory.
        self.agent = TinyPerson(name="TestInsightAgent")
        # Ensure semantic_memory is initialized if not done by default in TinyPerson constructor
        if not hasattr(self.agent, 'semantic_memory') or self.agent.semantic_memory is None:
            self.agent.semantic_memory = SemanticMemory()


    @patch('tinytroupe.openai_utils.client') # Mocking the LLM client
    def test_extract_and_store_semantic_insight_success(self, mock_openai_client):
        # 1. Successful insight extraction
        mock_llm_instance = MagicMock()
        mock_llm_instance.send_message.return_value = {'content': "Meaningful insight about the event."}
        mock_openai_client.return_value = mock_llm_instance

        self.agent.semantic_memory.store = MagicMock() # Mock the store method of semantic memory

        episodic_entry = {
            'type': 'CONVERSATION',
            'content': 'A user said hello.',
            'simulation_timestamp': '2023-01-01T12:00:00Z'
        }

        self.agent._extract_and_store_semantic_insight(episodic_entry)

        # Verify LLM call
        mock_llm_instance.send_message.assert_called_once()
        # print(mock_llm_instance.send_message.call_args) # For debugging the call if needed

        # Verify semantic_memory.store call
        self.agent.semantic_memory.store.assert_called_once()
        call_args = self.agent.semantic_memory.store.call_args

        self.assertIsNotNone(call_args)
        payload = call_args[0][0] # First positional argument of the call

        self.assertEqual(payload['type'], 'semantic_insight')
        self.assertEqual(payload['content'], "Meaningful insight about the event.")
        self.assertEqual(payload['source_event_type'], 'CONVERSATION')
        self.assertEqual(payload['source_event_timestamp'], '2023-01-01T12:00:00Z')
        self.assertTrue('simulation_timestamp' in payload) # Timestamp of insight extraction


    @patch('tinytroupe.openai_utils.client')
    @patch('tinytroupe.agent.tiny_person.logger') # Mock logger in TinyPerson module
    def test_extract_and_store_semantic_insight_no_insight(self, mock_logger, mock_openai_client):
        # 2. Handling of "None" or empty insight from LLM
        mock_llm_instance = MagicMock()
        # Test with "None"
        mock_llm_instance.send_message.return_value = {'content': "None"}
        mock_openai_client.return_value = mock_llm_instance

        self.agent.semantic_memory.store = MagicMock()

        episodic_entry = {'type': 'TEST_EVENT', 'content': 'Test content', 'simulation_timestamp': '2023-01-01T13:00:00Z'}
        self.agent._extract_and_store_semantic_insight(episodic_entry)

        self.agent.semantic_memory.store.assert_not_called()
        mock_logger.debug.assert_any_call(f"[{self.agent.name}] No distinct semantic insight extracted from event: Event Type: TEST_EVENT, Content: Test content, Timestamp: 2023-01-01T13:00:00Z")

        # Test with empty string
        mock_llm_instance.send_message.return_value = {'content': "  "} # Whitespace only
        mock_openai_client.return_value = mock_llm_instance
        self.agent.semantic_memory.store.reset_mock() # Reset for the next check
        mock_logger.reset_mock()

        self.agent._extract_and_store_semantic_insight(episodic_entry)
        self.agent.semantic_memory.store.assert_not_called()
        mock_logger.debug.assert_any_call(f"[{self.agent.name}] No distinct semantic insight extracted from event: Event Type: TEST_EVENT, Content: Test content, Timestamp: 2023-01-01T13:00:00Z")


    @patch('tinytroupe.openai_utils.client')
    @patch('tinytroupe.agent.tiny_person.logger') # Mock logger in TinyPerson module
    def test_extract_and_store_semantic_insight_llm_error(self, mock_logger, mock_openai_client):
        # 3. Error handling
        mock_llm_instance = MagicMock()
        mock_llm_instance.send_message.side_effect = Exception("LLM API Error")
        mock_openai_client.return_value = mock_llm_instance

        self.agent.semantic_memory.store = MagicMock()

        episodic_entry = {'type': 'ERROR_EVENT', 'content': 'Error test', 'simulation_timestamp': '2023-01-01T14:00:00Z'}

        self.agent._extract_and_store_semantic_insight(episodic_entry)

        self.agent.semantic_memory.store.assert_not_called()
        mock_logger.error.assert_called_once()
        # Check if the error message contains the exception
        args, kwargs = mock_logger.error.call_args
        self.assertIn(f"[{self.agent.name}] Error in _extract_and_store_semantic_insight: LLM API Error", args[0])


if __name__ == '__main__':
    # This allows running the tests with `python test_tinyperson.py`
    # For pytest, it will discover tests automatically.
    unittest.main()
