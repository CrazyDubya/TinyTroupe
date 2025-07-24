import pytest
import logging
logger = logging.getLogger("tinytroupe")

import sys
sys.path.insert(0, '../../tinytroupe/')  # ensures that the package is imported from the parent directory, not the Python installation
sys.path.insert(0, '../../')
sys.path.insert(0, '..')

from tinytroupe.agent.tiny_person import TinyPerson
from tinytroupe.agent.memory import EpisodicMemory, SemanticMemory
from tinytroupe.examples import (
    create_oscar_the_architect,
    create_lisa_the_data_scientist,
    create_oscar_the_architect_2,
    create_lisa_the_data_scientist_2,
)
import tinytroupe.openai_utils as openai_utils  # For mocking

from testing_utils import *  # Assumes get_relative_to_test_path, EXPORT_BASE_FOLDER etc. are here
import os  # For test_save_specification

# Imports for new tests
import unittest
from unittest.mock import patch, MagicMock, call
import datetime


def test_act(setup):
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        actions = agent.listen_and_act("Tell me a bit about your life.", return_actions=True)
        logger.info(agent.pp_current_interactions())

        assert len(actions) >= 1, f"{agent.name} should have at least one action to perform (even if it is just DONE)."
        assert contains_action_type(actions, "TALK"), f"{agent.name} should have at least one TALK action to perform."
        assert terminates_with_action_type(actions, "DONE"), f"{agent.name} should always terminate with a DONE action."

        for action in actions:
            if action["action"]["type"] == "TALK":
                talk_content = action["action"]["content"]
                assert proposition_holds(
                    f"The following text is someone talking about their personal life, background, or experiences: '{talk_content}'"
                ), f"Agent should be talking about their life but said: {talk_content}"
                break


def test_listen(setup):
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.listen("Hello, how are you?")
        last = agent.episodic_memory.retrieve_all()[-1]

        assert len(agent.current_messages) > 0, f"{agent.name} should have at least one message in its current messages."
        assert last["role"] == "user", f"{agent.name} should have the last message as 'user'."
        stim = last["content"]["stimuli"][0]
        assert stim["type"] == "CONVERSATION", f"{agent.name} should have the last message as a 'CONVERSATION' stimulus."
        assert stim["content"] == "Hello, how are you?", f"{agent.name} should have the last message with the correct content."


def test_define(setup):
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        original_prompt = agent.current_messages[0]["content"]
        agent.define("age", 25)
        assert agent._persona["age"] == 25
        assert agent.current_messages[0]["content"] != original_prompt
        assert "25" in agent.current_messages[0]["content"]


def test_define_several(setup):
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.define("skills", ["Python", "Machine learning", "GPT-3"])
        skills = agent._persona["skills"]
        assert "Python" in skills
        assert "Machine learning" in skills
        assert "GPT-3" in skills


def test_socialize(setup):
    an_oscar = create_oscar_the_architect()
    a_lisa = create_lisa_the_data_scientist()
    for agent in [an_oscar, a_lisa]:
        other = a_lisa if agent.name == "Oscar" else an_oscar
        agent.make_agent_accessible(other, relation_description="My friend")
        agent.listen(f"Hi {agent.name}, I am {other.name}.")
        actions = agent.act(return_actions=True)
        assert len(actions) >= 1
        assert contains_action_type(actions, "TALK")
        assert contains_action_content(actions, agent_first_name(other))


def test_see(setup):
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.see("A beautiful sunset over the ocean.")
        actions = agent.act(return_actions=True)
        assert len(actions) >= 1
        assert contains_action_type(actions, "THINK")
        assert contains_action_content(actions, "sunset")
        for action in actions:
            if action["action"]["type"] == "THINK":
                think_content = action["action"]["content"]
                assert proposition_holds(
                    f"The following text is someone thinking about or reacting to seeing a beautiful sunset over the ocean: '{think_content}'"
                )
                break


def test_think(setup):
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.think("I will tell everyone right now how awesome life is!")
        actions = agent.act(return_actions=True)
        assert len(actions) >= 1
        assert contains_action_type(actions, "TALK")
        assert contains_action_content(actions, "life")
        for action in actions:
            if action["action"]["type"] == "TALK":
                talk_content = action["action"]["content"]
                assert proposition_holds(
                    f"The following text expresses enthusiasm or positive feelings about life: '{talk_content}'"
                )
                break


def test_internalize_goal(setup):
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.internalize_goal("I want to compose in my head a wonderful poem about how cats are glorious creatures.")
        actions = agent.act(return_actions=True)
        assert len(actions) >= 1
        assert contains_action_type(actions, "THINK")
        assert contains_action_content(actions, "cats")
        for action in actions:
            if action["action"]["type"] == "THINK":
                think_content = action["action"]["content"]
                assert proposition_holds(
                    f"The following text is someone thinking about cats, poetry, or composing a poem about cats: '{think_content}'"
                )
                break


def test_move_to(setup):
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.move_to("New York", context=["city", "busy", "diverse"])
        ctx = agent._mental_state["context"]
        assert agent._mental_state["location"] == "New York"
        assert "city" in ctx and "busy" in ctx and "diverse" in ctx


def test_change_context(setup):
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        agent.change_context(["home", "relaxed", "comfortable"])
        ctx = agent._mental_state["context"]
        assert "home" in ctx and "relaxed" in ctx and "comfortable" in ctx


def test_save_specification(setup):
    for agent in [create_oscar_the_architect(), create_lisa_the_data_scientist()]:
        path = get_relative_to_test_path(f"{EXPORT_BASE_FOLDER}/serialization/{agent.name}.tinyperson.json")
        agent.save_specification(path, include_memory=True)
        assert os.path.exists(path)
        loaded_name = f"{agent.name}_loaded"
        loaded_agent = TinyPerson.load_specification(path, new_agent_name=loaded_name)
        assert loaded_agent.name == loaded_name
        assert agents_personas_are_equal(agent, loaded_agent, ignore_name=True)


def test_programmatic_definitions(setup):
    for agent in [create_oscar_the_architect_2(), create_lisa_the_data_scientist_2()]:
        agent.listen_and_act("Tell me a bit about your life.")


# New Test Class for _extract_and_store_semantic_insight
class TestTinyPersonSemanticExtraction(unittest.TestCase):

    def setUp(self):
        self.agent = TinyPerson(name="TestInsightAgent")
        if not hasattr(self.agent, "semantic_memory") or self.agent.semantic_memory is None:
            self.agent.semantic_memory = SemanticMemory()

    @patch("tinytroupe.openai_utils.client")
    def test_extract_and_store_semantic_insight_success(self, mock_openai_client):
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = {"content": "Meaningful insight about the event."}
        mock_openai_client.return_value = mock_llm

        self.agent.semantic_memory.store = MagicMock()

        episodic_entry = {
            "type": "CONVERSATION",
            "content": "A user said hello.",
            "simulation_timestamp": "2023-01-01T12:00:00Z",
        }
        self.agent._extract_and_store_semantic_insight(episodic_entry)

        mock_llm.send_message.assert_called_once()
        self.agent.semantic_memory.store.assert_called_once()
        payload = self.agent.semantic_memory.store.call_args[0][0]
        self.assertEqual(payload["type"], "semantic_insight")
        self.assertEqual(payload["content"], "Meaningful insight about the event.")
        self.assertEqual(payload["source_event_type"], "CONVERSATION")
        self.assertEqual(payload["source_event_timestamp"], "2023-01-01T12:00:00Z")
        self.assertIn("simulation_timestamp", payload)

    @patch("tinytroupe.openai_utils.client")
    @patch("tinytroupe.agent.tiny_person.logger")
    def test_extract_and_store_semantic_insight_no_insight(self, mock_logger, mock_openai_client):
        mock_llm = MagicMock()
        mock_llm.send_message.return_value = {"content": "None"}
        mock_openai_client.return_value = mock_llm
        self.agent.semantic_memory.store = MagicMock()

        episodic_entry = {
            "type": "TEST_EVENT",
            "content": "Test content",
            "simulation_timestamp": "2023-01-01T13:00:00Z",
        }
        self.agent._extract_and_store_semantic_insight(episodic_entry)
        self.agent.semantic_memory.store.assert_not_called()
        mock_logger.debug.assert_any_call(
            f"[{self.agent.name}] No distinct semantic insight extracted from event: "
            f"Event Type: TEST_EVENT, Content: Test content, Timestamp: 2023-01-01T13:00:00Z"
        )

        mock_llm.send_message.return_value = {"content": "  "}
        self.agent._extract_and_store_semantic_insight(episodic_entry)
        self.agent.semantic_memory.store.assert_not_called()
        mock_logger.debug.assert_any_call(
            f"[{self.agent.name}] No distinct semantic insight extracted from event: "
            f"Event Type: TEST_EVENT, Content: Test content, Timestamp: 2023-01-01T13:00:00Z"
        )

    @patch("tinytroupe.openai_utils.client")
    @patch("tinytroupe.agent.tiny_person.logger")
    def test_extract_and_store_semantic_insight_llm_error(self, mock_logger, mock_openai_client):
        mock_llm = MagicMock()
        mock_llm.send_message.side_effect = Exception("LLM API Error")
        mock_openai_client.return_value = mock_llm
        self.agent.semantic_memory.store = MagicMock()

        episodic_entry = {
            "type": "ERROR_EVENT",
            "content": "Error test",
            "simulation_timestamp": "2023-01-01T14:00:00Z",
        }
        self.agent._extract_and_store_semantic_insight(episodic_entry)
        self.agent.semantic_memory.store.assert_not_called()
        mock_logger.error.assert_called_once()
        self.assertIn(
            f"[{self.agent.name}] Error in _extract_and_store_semantic_insight: LLM API Error",
            mock_logger.error.call_args[0][0],
        )


def test_loop_detection_stops_after_threshold(setup, caplog):
    agent = TinyPerson(name="LoopTestAgent")
    original = TinyPerson.LOOP_DETECTION_THRESHOLD
    TinyPerson.LOOP_DETECTION_THRESHOLD = 3

    looping_action = {"type": "TALK", "content": "I am stuck in a loop."}
    mock_resp = {
        "role": "assistant",
        "content": {
            "cognitive_state": {"goals": [], "attention": "", "emotions": ""},
            "action": looping_action,
        },
    }
    agent._produce_message = MagicMock(return_value=(mock_resp["role"], mock_resp["content"]))
    caplog.set_level(logging.WARNING)

    actions = agent.act(until_done=True, return_actions=True)
    assert len(actions) == TinyPerson.LOOP_DETECTION_THRESHOLD
    for act in actions:
        assert act["action"] == looping_action
    assert "acting in a loop" in caplog.text

    TinyPerson.LOOP_DETECTION_THRESHOLD = original


def test_loop_detection_does_not_trigger_below_threshold(setup):
    agent = TinyPerson(name="NoLoopTestAgent")
    original = TinyPerson.LOOP_DETECTION_THRESHOLD
    TinyPerson.LOOP_DETECTION_THRESHOLD = 3

    action = {"type": "THINK", "content": "Thinking..."}
    done = {"type": "DONE", "content": ""}
    responses = [
        ("assistant", {"cognitive_state": {}, "action": action}),
        ("assistant", {"cognitive_state": {}, "action": action}),
        ("assistant", {"cognitive_state": {}, "action": done}),
    ]
    agent._produce_message = MagicMock(side_effect=responses)

    actions = agent.act(until_done=True, return_actions=True)
    assert len(actions) == 3
    assert actions[2]["action"] == done

    TinyPerson.LOOP_DETECTION_THRESHOLD = original


def test_loop_detection_with_different_actions(setup, caplog):
    agent = TinyPerson(name="DifferentActionsAgent")
    original = TinyPerson.LOOP_DETECTION_THRESHOLD
    TinyPerson.LOOP_DETECTION_THRESHOLD = 3

    a1 = {"type": "TALK", "content": "Hello."}
    a2 = {"type": "THINK", "content": "Hmm."}
    a3 = {"type": "TALK", "content": "Goodbye."}
    done = {"type": "DONE", "content": ""}
    responses = [
        ("assistant", {"cognitive_state": {}, "action": a1}),
        ("assistant", {"cognitive_state": {}, "action": a2}),
        ("assistant", {"cognitive_state": {}, "action": a3}),
        ("assistant", {"cognitive_state": {}, "action": done}),
    ]
    agent._produce_message = MagicMock(side_effect=responses)
    caplog.set_level(logging.WARNING)

    actions = agent.act(until_done=True, return_actions=True)
    assert len(actions) == 4
    assert "acting in a loop" not in caplog.text

    TinyPerson.LOOP_DETECTION_THRESHOLD = original


if __name__ == "__main__":
    unittest.main()
