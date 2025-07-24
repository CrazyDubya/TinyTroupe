from tinytroupe.agent import logger, default, Self, AgentOrWorld, CognitiveActionModel
from tinytroupe.agent.memory import EpisodicMemory, SemanticMemory, EpisodicConsolidator
import tinytroupe.openai_utils as openai_utils
from tinytroupe.utils import JsonSerializableRegistry, repeat_on_error, name_or_empty
import tinytroupe.utils as utils
from tinytroupe.control import transactional, current_simulation
from tinytroupe import config_manager

import os
import json
import copy
import datetime # Added for timestamping semantic extraction
import textwrap  # to dedent strings
import chevron  # to parse Mustache templates
from typing import Any
from rich import print
import threading
from tinytroupe.utils import LLMChat  # Import LLMChat from the appropriate module

import tinytroupe.utils.llm

# to protect from race conditions when running agents in parallel
concurrent_agent_action_lock = threading.Lock()

#######################################################################################################################
# TinyPerson itself
#######################################################################################################################
@utils.post_init
class TinyPerson(JsonSerializableRegistry):
    """A simulated person in the TinyTroupe universe."""

    # The maximum number of actions that an agent is allowed to perform before DONE.
    # This prevents the agent from acting without ever stopping.
    MAX_ACTIONS_BEFORE_DONE = 15
    REFLECTION_TRIGGER_ACTION_COUNT = 5 # Number of actions in a single 'act' call to trigger reflection


    # The number of identical consecutive actions to detect a loop.
    LOOP_DETECTION_THRESHOLD = 5

    PP_TEXT_WIDTH = 100

    # The maximum similarity between consecutive actions. If the similarity is too high, the action is discarded and replaced by a DONE.
    # Set this to None to disable the check.
    MAX_ACTION_SIMILARITY = 0.85

    MIN_EPISODE_LENGTH = config_manager.get("min_episode_length", 15)  # The minimum number of messages in an episode before it is considered valid.
    MAX_EPISODE_LENGTH = config_manager.get("max_episode_length", 50)  # The maximum number of messages in an episode before it is considered valid.



    serializable_attributes = ["_persona", "_mental_state", "_mental_faculties", "_current_episode_event_count", "episodic_memory", "semantic_memory"]
    serializable_attributes_renaming = {"_mental_faculties": "mental_faculties", "_persona": "persona", "_mental_state": "mental_state", "_current_episode_event_count": "current_episode_event_count"}

    # A dict of all agents instantiated so far.
    all_agents = {}  # name -> agent
   
    # Whether to display the communication or not. True is for interactive applications, when we want to see simulation
    # outputs as they are produced.
    communication_display:bool=True
    

    def __init__(self, name:str=None, 
                 action_generator=None,
                 episodic_memory=None,
                 semantic_memory=None,
                 mental_faculties:list=None,
                 enable_basic_action_repetition_prevention:bool=True):
        """
        Creates a TinyPerson.

        Args:
            name (str): The name of the TinyPerson. Either this or spec_path must be specified.
            action_generator (ActionGenerator, optional): The action generator to use. Defaults to ActionGenerator().
            episodic_memory (EpisodicMemory, optional): The memory implementation to use. Defaults to EpisodicMemory().
            semantic_memory (SemanticMemory, optional): The memory implementation to use. Defaults to SemanticMemory().
            mental_faculties (list, optional): A list of mental faculties to add to the agent. Defaults to None.
            enable_basic_action_repetition_prevention (bool, optional): Whether to enable basic action repetition prevention. Defaults to True.
        """

        # NOTE: default values will be given in the _post_init method, as that's shared by 
        #       direct initialization as well as via deserialization.

        if action_generator is not None:
            self.action_generator = action_generator

        if episodic_memory is not None:
            self.episodic_memory = episodic_memory
        
        if semantic_memory is not None:
            self.semantic_memory = semantic_memory

        # Mental faculties
        if mental_faculties is not None:
            self._mental_faculties = mental_faculties
        
        if enable_basic_action_repetition_prevention:
            self.enable_basic_action_repetition_prevention = enable_basic_action_repetition_prevention
        
        assert name is not None, "A TinyPerson must have a name."
        self.name = name

        # @post_init makes sure that _post_init is called after __init__

    
    def _post_init(self, **kwargs):
        """
        This will run after __init__, since the class has the @post_init decorator.
        It is convenient to separate some of the initialization processes to make deserialize easier.
        """

        from tinytroupe.agent.action_generator import ActionGenerator # import here to avoid circular import issues


        ############################################################
        # Default values
        ############################################################

        self.current_messages = []
        
        # the current environment in which the agent is acting
        self.environment = None

        # The list of actions that this agent has performed so far, but which have not been
        # consumed by the environment yet.
        self._actions_buffer = []

        # The list of agents that this agent can currently interact with.
        # This can change over time, as agents move around the world.
        self._accessible_agents = []

        # the buffer of communications that have been displayed so far, used for
        # saving these communications to another output form later (e.g., caching)
        self._displayed_communications_buffer = []

        if not hasattr(self, '_current_episode_event_count'):
            self._current_episode_event_count = 0  # the number of events in the current episode, used to limit the episode length

        if not hasattr(self, 'action_generator'):
            # This default value MUST NOT be in the method signature, otherwise it will be shared across all instances.
            self.action_generator = ActionGenerator(max_attempts=config_manager.get("action_generator_max_attempts"),
                                                    enable_quality_checks=config_manager.get("action_generator_enable_quality_checks"),
                                                    enable_regeneration=config_manager.get("action_generator_enable_regeneration"),
                                                    enable_direct_correction=config_manager.get("action_generator_enable_direct_correction"),
                                                    enable_quality_check_for_persona_adherence=config_manager.get("action_generator_enable_quality_check_for_persona_adherence"),
                                                    enable_quality_check_for_selfconsistency=config_manager.get("action_generator_enable_quality_check_for_selfconsistency"),
                                                    enable_quality_check_for_fluency=config_manager.get("action_generator_enable_quality_check_for_fluency"),
                                                    enable_quality_check_for_suitability=config_manager.get("action_generator_enable_quality_check_for_suitability"),
                                                    enable_quality_check_for_similarity=config_manager.get("action_generator_enable_quality_check_for_similarity"),
                                                    continue_on_failure=config_manager.get("action_generator_continue_on_failure"),
                                                    quality_threshold=config_manager.get("action_generator_quality_threshold"))

        if not hasattr(self, 'episodic_memory'):
            # This default value MUST NOT be in the method signature, otherwise it will be shared across all instances.
            self.episodic_memory = EpisodicMemory(fixed_prefix_length= config_manager.get("episodic_memory_fixed_prefix_length"),
                                                   lookback_length=config_manager.get("episodic_memory_lookback_length"))
        
        if not hasattr(self, 'semantic_memory'):
            # This default value MUST NOT be in the method signature, otherwise it will be shared across all instances.
            self.semantic_memory = SemanticMemory()
        
        # _mental_faculties
        if not hasattr(self, '_mental_faculties'):
            # This default value MUST NOT be in the method signature, otherwise it will be shared across all instances.
            self._mental_faculties = []
        
        # basic action repetition prevention
        if not hasattr(self, 'enable_basic_action_repetition_prevention'):
            self.enable_basic_action_repetition_prevention = True

        # create the persona configuration dictionary
        if not hasattr(self, '_persona'):          
            self._persona = {
                "name": self.name,
                "age": None,
                "nationality": None,
                "country_of_residence": None,
                "occupation": None
            }
        
        if not hasattr(self, 'name'): 
            self.name = self._persona["name"]

        # create the mental state dictionary
        if not hasattr(self, '_mental_state'):
            self._mental_state = {
                "datetime": None,
                "location": None,
                "context": [],
                "goals": [],
                "attention": None,
                "emotions": "Feeling nothing in particular, just calm.",
                "memory_context": None,
                "accessible_agents": []  # [{"agent": agent_1, "relation": "My friend"}, {"agent": agent_2, "relation": "My colleague"}, ...]
            }
        
        if not hasattr(self, '_extended_agent_summary'):
            self._extended_agent_summary = None
        
        if not hasattr(self, 'actions_count'):
            self.actions_count = 0
        
        if not hasattr(self, 'stimuli_count'):
            self.stimuli_count = 0

        self._prompt_template_path = os.path.join(
            os.path.dirname(__file__), "prompts/tiny_person.mustache"
        )
        self._init_system_message = None  # initialized later


        ############################################################
        # Special mechanisms used during deserialization
        ############################################################

        # rename agent to some specific name?
        if kwargs.get("new_agent_name") is not None:
            self._rename(kwargs.get("new_agent_name"))
        
        # If auto-rename, use the given name plus some new number ...
        if kwargs.get("auto_rename") is True:
            new_name = self.name # start with the current name
            rename_succeeded = False
            while not rename_succeeded:
                try:
                    self._rename(new_name)
                    TinyPerson.add_agent(self)
                    rename_succeeded = True                
                except ValueError:
                    new_id = utils.fresh_id(self.__class__.__name__)
                    new_name = f"{self.name}_{new_id}"
        
        # ... otherwise, just register the agent
        else:
            # register the agent in the global list of agents
            TinyPerson.add_agent(self)

        # start with a clean slate
        self.reset_prompt()

        # it could be the case that the agent is being created within a simulation scope, in which case
        # the simulation_id must be set accordingly
        if current_simulation() is not None:
            current_simulation().add_agent(self)
        else:
            self.simulation_id = None
    
    def _rename(self, new_name:str):    
        self.name = new_name
        self._persona["name"] = self.name


    def generate_agent_system_prompt(self):
        with open(self._prompt_template_path, "r") as f:
            agent_prompt_template = f.read()

        # let's operate on top of a copy of the configuration, because we'll need to add more variables, etc.
        template_variables = self._persona.copy()    
        template_variables["persona"] = json.dumps(self._persona.copy(), indent=4)    

        # add mental state to the template variables
        template_variables["mental_state"] = json.dumps(self._mental_state, indent=4)

        # Prepare additional action definitions and constraints
        actions_definitions_prompt = ""
        actions_constraints_prompt = ""
        for faculty in self._mental_faculties:
            actions_definitions_prompt += f"{faculty.actions_definitions_prompt()}\n"
            actions_constraints_prompt += f"{faculty.actions_constraints_prompt()}\n"
        
        # Make the additional prompt pieces available to the template. 
        # Identation here is to align with the text structure in the template.
        template_variables['actions_definitions_prompt'] = textwrap.indent(actions_definitions_prompt.strip(), "  ")
        template_variables['actions_constraints_prompt'] = textwrap.indent(actions_constraints_prompt.strip(), "  ")

        # RAI prompt components, if requested
        template_variables = utils.add_rai_template_variables_if_enabled(template_variables)

        return chevron.render(agent_prompt_template, template_variables)

    def reset_prompt(self):

        # render the template with the current configuration
        self._init_system_message = self.generate_agent_system_prompt()

        # - reset system message
        # - make it clear that the provided events are past events and have already had their effects
        self.current_messages = [
            {"role": "system", "content": self._init_system_message},
            {"role": "system", "content": "The next messages refer to past interactions you had recently and are meant to help you contextualize your next actions. "\
                                        + "They are the most recent episodic memories you have, including stimuli and actions. "\
                                        + "Their effects already took place and led to your present cognitive state (described above), so you can use them in conjunction "\
                                        + "with your cognitive state to inform your next actions and perceptions. Please consider them and then proceed with your next actions right after. "}
        ]

        # sets up the actual interaction messages to use for prompting
        self.current_messages += self.retrieve_recent_memories()


    #########################################################################
    # Persona definitions
    #########################################################################
    
    # 
    # Conveniences to access the persona configuration via dictionary-like syntax using
    # the [] operator. e.g., agent["nationality"] = "American"
    #
    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.define(key, value)

    #
    # Conveniences to import persona definitions via the '+' operator, 
    #  e.g., agent + {"nationality": "American", ...}
    #
    #  e.g., agent + "path/to/fragment.json"
    #
    def __add__(self, other):
        """
        Allows using the '+' operator to add persona definitions or import a fragment.
        If 'other' is a dict, calls include_persona_definitions().
        If 'other' is a string, calls import_fragment().
        """
        if isinstance(other, dict):
            self.include_persona_definitions(other)
        elif isinstance(other, str):
            self.import_fragment(other)
        else:
            raise TypeError("Unsupported operand type for +. Must be a dict or a string path to fragment.")
        return self

    #
    # Various other conveniences to manipulate the persona configuration
    #

    def get(self, key):
        """
        Returns the value of a key in the TinyPerson's persona configuration.
        Supports dot notation for nested keys (e.g., "address.city").
        """
        keys = key.split(".")
        value = self._persona
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, None)
            else:
                return None  # If the path is invalid, return None
        return value
    
    @transactional()
    def import_fragment(self, path):
        """
        Imports a fragment of a persona configuration from a JSON file.
        """
        with open(path, "r") as f:
            fragment = json.load(f)

        # check the type is "Fragment" and that there's also a "persona" key
        if fragment.get("type", None) == "Fragment" and fragment.get("persona", None) is not None:
            self.include_persona_definitions(fragment["persona"])
        else:
            raise ValueError("The imported JSON file must be a valid fragment of a persona configuration.")
        
        # must reset prompt after adding to configuration
        self.reset_prompt()

    @transactional()
    def include_persona_definitions(self, additional_definitions: dict):
        """
        Imports a set of definitions into the TinyPerson. They will be merged with the current configuration.
        It is also a convenient way to include multiple bundled definitions into the agent.

        Args:
            additional_definitions (dict): The additional definitions to import.
        """

        self._persona = utils.merge_dicts(self._persona, additional_definitions)

        # must reset prompt after adding to configuration
        self.reset_prompt()
        
    
    @transactional()
    def define(self, key, value, merge=False, overwrite_scalars=True):
        """
        Define a value to the TinyPerson's persona configuration. Value can either be a scalar or a dictionary.
        If the value is a dictionary or list, you can choose to merge it with the existing value or replace it. 
        If the value is a scalar, you can choose to overwrite the existing value or not.

        Args:
            key (str): The key to define.
            value (Any): The value to define.
            merge (bool, optional): Whether to merge the dict/list values with the existing values or replace them. Defaults to False.
            overwrite_scalars (bool, optional): Whether to overwrite scalar values or not. Defaults to True.
        """

        # dedent value if it is a string
        if isinstance(value, str):
            value = textwrap.dedent(value)

        # if the value is a dictionary, we can choose to merge it with the existing value or replace it
        if isinstance(value, dict) or isinstance(value, list):
            if merge:
                self._persona = utils.merge_dicts(self._persona, {key: value})
            else:
                self._persona[key] = value

        # if the value is a scalar, we can choose to overwrite it or not
        elif overwrite_scalars or (key not in self._persona):
            self._persona[key] = value
        
        else:
            raise ValueError(f"The key '{key}' already exists in the persona configuration and overwrite_scalars is set to False.")

            
        # must reset prompt after adding to configuration
        self.reset_prompt()

    
    @transactional()
    def define_relationships(self, relationships, replace=True):
        """
        Defines or updates the TinyPerson's relationships.

        Args:
            relationships (list or dict): The relationships to add or replace. Either a list of dicts mapping agent names to relationship descriptions,
              or a single dict mapping one agent name to its relationship description.
            replace (bool, optional): Whether to replace the current relationships or just add to them. Defaults to True.
        """
        
        if (replace == True) and (isinstance(relationships, list)):
            self._persona['relationships'] = relationships

        elif replace == False:
            current_relationships = self._persona['relationships']
            if isinstance(relationships, list):
                for r in relationships:
                    current_relationships.append(r)
                
            elif isinstance(relationships, dict) and len(relationships) == 2: #{"Name": ..., "Description": ...}
                current_relationships.append(relationships)

            else:
                raise Exception("Only one key-value pair is allowed in the relationships dict.")

        else:
            raise Exception("Invalid arguments for define_relationships.")

    ##############################################################################
    # Relationships
    ##############################################################################

    @transactional()
    def clear_relationships(self):
        """
        Clears the TinyPerson's relationships.
        """
        self._persona['relationships'] = []  

        return self      
    
    @transactional()
    def related_to(self, other_agent, description, symmetric_description=None):
        """
        Defines a relationship between this agent and another agent.

        Args:
            other_agent (TinyPerson): The other agent.
            description (str): The description of the relationship.
            symmetric (bool): Whether the relationship is symmetric or not. That is, 
              if the relationship is defined for both agents.
        
        Returns:
            TinyPerson: The agent itself, to facilitate chaining.
        """
        self.define_relationships([{"Name": other_agent.name, "Description": description}], replace=False)
        if symmetric_description is not None:
            other_agent.define_relationships([{"Name": self.name, "Description": symmetric_description}], replace=False)
        
        return self
    
    ############################################################################
    
    def add_mental_faculties(self, mental_faculties):
        """
        Adds a list of mental faculties to the agent.
        """
        for faculty in mental_faculties:
            self.add_mental_faculty(faculty)
        
        return self

    def add_mental_faculty(self, faculty):
        """
        Adds a mental faculty to the agent.
        """
        # check if the faculty is already there or not
        if faculty not in self._mental_faculties:
            self._mental_faculties.append(faculty)
        else:
            raise Exception(f"The mental faculty {faculty} is already present in the agent.")
        
        return self

    @transactional()
    @config_manager.config_defaults(max_content_length="max_content_display_length")
    def act(
        self,
        until_done=True,
        n=None,
        return_actions=False,
        max_content_length=None,
    ):
        """
        Acts in the environment and updates its internal cognitive state.
        Either acts until the agent is done and needs additional stimuli, or acts a fixed number of times,
        but not both.

        Args:
            until_done (bool): Whether to keep acting until the agent is done and needs additional stimuli.
            n (int): The number of actions to perform. Defaults to None.
            return_actions (bool): Whether to return the actions or not. Defaults to False.
        """

        # either act until done or act a fixed number of times, but not both
        assert not (until_done and n is not None)
        if n is not None:
            assert n < TinyPerson.MAX_ACTIONS_BEFORE_DONE

        contents = []

        # A separate function to run before each action, which is not meant to be repeated in case of errors.
        def aux_pre_act():
            # TODO maybe we don't need this at all anymore?
            #
            # A quick thought before the action. This seems to help with better model responses, perhaps because
            # it interleaves user with assistant messages.
            pass # self.think("I will now think, reflect and act a bit, and then issue DONE.")        

        # Aux function to perform exactly one action.
        # Occasionally, the model will return JSON missing important keys, so we just ask it to try again
        # Sometimes `content` contains EpisodicMemory's MEMORY_BLOCK_OMISSION_INFO message, which raises a TypeError on line 443
        @repeat_on_error(retries=5, exceptions=[KeyError, TypeError])
        def aux_act_once():
            # ensure we have the latest prompt (initial system message + selected messages from memory)
            self.reset_prompt()
            
            action, role, content, all_negative_feedbacks = self.action_generator.generate_next_action(self, self.current_messages)
            logger.debug(f"{self.name}'s action: {action}")

            # check the next action similarity, and if it is too similar, put a system warning instruction in memory too
            next_action_similarity = utils.next_action_jaccard_similarity(self, action)

            # we have a redundant repetition check here, because this an be computed quickly and is often very useful.
            if self.enable_basic_action_repetition_prevention and \
               (TinyPerson.MAX_ACTION_SIMILARITY is not None) and (next_action_similarity > TinyPerson.MAX_ACTION_SIMILARITY):
                
                logger.warning(f"[{self.name}] Action similarity is too high ({next_action_similarity}), replacing it with DONE.")

                # replace the action with a DONE
                action = {"type": "DONE", "content": "", "target": ""}
                content["action"] = action	
                content["cognitive_state"] = {}

                self.store_in_memory({'role': 'system', 
                                    'content': \
                                        f"""
                                        # EXCESSIVE ACTION SIMILARITY WARNING

                                        You were about to generate a repetitive action (jaccard similarity = {next_action_similarity}).
                                        Thus, the action was discarded and replaced by an artificial DONE.

                                        DO NOT BE REPETITIVE. This is not a human-like behavior, therefore you **must** avoid this in the future.
                                        Your alternatives are:
                                        - produce more diverse actions.
                                        - aggregate similar actions into a single, larger, action and produce it all at once.
                                        - as a **last resort only**, you may simply not acting at all by issuing a DONE.

                                        
                                        """,
                                    'type': 'feedback',
                                    'simulation_timestamp': self.iso_datetime()})

            # All checks done, we can commit the action to memory.
            self.store_in_memory({'role': role, 'content': content, 
                                    'type': 'action', 
                                    'simulation_timestamp': self.iso_datetime()})
                
            self._actions_buffer.append(action)
            
            if "cognitive_state" in content:
                cognitive_state = content["cognitive_state"]
                logger.debug(f"[{self.name}] Cognitive state: {cognitive_state}")
                
                self._update_cognitive_state(goals=cognitive_state.get("goals", None),
                                             context=cognitive_state.get("context", None),
                                             attention=cognitive_state.get("emotions", None),
                                             emotions=cognitive_state.get("emotions", None))
            
            contents.append(content)          
            if TinyPerson.communication_display:
                self._display_communication(role=role, content=content, kind='action', simplified=True, max_content_length=max_content_length)
            
            #
            # Some actions induce an immediate stimulus or other side-effects. We need to process them here, by means of the mental faculties.
            #
            for faculty in self._mental_faculties:
                faculty.process_action(self, action)
            
            #
            # turns all_negative_feedbacks list into a system message
            #
            # TODO improve this?
            #
            ##if len(all_negative_feedbacks) > 0:
            ##    feedback = """
            ##    # QUALITY FEEDBACK
            ##
            ##    Up to the present moment, we monitored actions and tentative aborted actions (i.e., that were not actually executed), 
            ##    and some of them were not of good quality.
            ##    Some of those were replaced by regenerated actions of better quality. In the process of doing so, some
            ##    important quality feedback was produced, which is now given below.
            ##    
            ##    To improve your performance, and prevent future similar quality issues, you **MUST** take into account the following feedback
            ##    whenever computing your future actions. Note that the feedback might also include the actual action or tentative action
            ##    that was of low quality, so that you can understand what was wrong with it and avoid similar mistakes in the future.
            ##
            ##    """
            ##    for i, feedback_item in enumerate(all_negative_feedbacks):
            ##        feedback += f"{feedback_item}\n\n"
            ##        feedback += f"\n\n *** \n\n"
            ##
            ##    self.store_in_memory({'role': 'system', 'content': feedback, 
            ##                          'type': 'feedback',
            ##                          'simulation_timestamp': self.iso_datetime()})
            ##

            

            # count the actions as this can be useful for taking decisions later
            self.actions_count += 1             
            

        #
        # How to proceed with a sequence of actions.
        #

        ##### Option 1: run N actions ######
        if n is not None:
            for i in range(n):
                aux_pre_act()
                aux_act_once()

        ##### Option 2: run until DONE ######
        elif until_done:
            while (len(contents) == 0) or (
                not contents[-1]["action"]["type"] == "DONE"
            ):


                # check if the agent is acting without ever stopping
                if len(contents) > TinyPerson.MAX_ACTIONS_BEFORE_DONE:
                    logger.warning(f"[{self.name}] Agent {self.name} is acting without ever stopping. This may be a bug. Let's stop it here anyway.")
                    break
                # Check for loops by looking at the last N actions
                if len(contents) >= TinyPerson.LOOP_DETECTION_THRESHOLD:
                    last_n_actions = [c['action'] for c in contents[-TinyPerson.LOOP_DETECTION_THRESHOLD:]]
                    # Check if all actions in the last_n_actions list are identical
                    if len(set(map(json.dumps, last_n_actions))) == 1:
                        logger.warning(
                            f"[{self.name}] Agent {self.name} is acting in a loop (last "
                            f"{TinyPerson.LOOP_DETECTION_THRESHOLD} actions are identical). "
                            f"This may be a bug. Let's stop it here anyway."
                        )
                        break

                aux_pre_act()
                aux_act_once()


            # Determine if reflection should be triggered
            should_reflect = False
            if contents: # Check if any actions were performed
                if contents[-1]["action"]["type"] == "DONE":
                    should_reflect = True
                    logger.info(f"[{self.name}] Triggering reflection because agent is DONE with current actions.")
                elif len(contents) >= TinyPerson.REFLECTION_TRIGGER_ACTION_COUNT:
                    should_reflect = True
                    logger.info(f"[{self.name}] Triggering reflection after {len(contents)} actions in this turn.")

            if should_reflect:
                self.reflect_and_synthesize_knowledge()

        # The end of a sequence of actions is always considered to mark the end of an episode.
        self.consolidate_episode_memories()


        if return_actions:
            return contents

    @transactional()
    @config_manager.config_defaults(max_content_length="max_content_display_length")
    def listen(
        self,
        speech,
        source: AgentOrWorld = None,
        max_content_length=None,
    ):
        """
        Listens to another agent (artificial or human) and updates its internal cognitive state.

        Args:
            speech (str): The speech to listen to.
            source (AgentOrWorld, optional): The source of the speech. Defaults to None.
        """

        return self._observe(
            stimulus={
                "type": "CONVERSATION",
                "content": speech,
                "source": name_or_empty(source),
            },
            max_content_length=max_content_length,
        )

    @config_manager.config_defaults(max_content_length="max_content_display_length")
    def socialize(
        self,
        social_description: str,
        source: AgentOrWorld = None,
        max_content_length=None,
    ):
        """
        Perceives a social stimulus through a description and updates its internal cognitive state.

        Args:
            social_description (str): The description of the social stimulus.
            source (AgentOrWorld, optional): The source of the social stimulus. Defaults to None.
        """
        return self._observe(
            stimulus={
                "type": "SOCIAL",
                "content": social_description,
                "source": name_or_empty(source),
            },
            max_content_length=max_content_length,
        )

    @config_manager.config_defaults(max_content_length="max_content_display_length")
    def see(
        self,
        visual_description,
        source: AgentOrWorld = None,
        max_content_length=None,
    ):
        """
        Perceives a visual stimulus through a description and updates its internal cognitive state.

        Args:
            visual_description (str): The description of the visual stimulus.
            source (AgentOrWorld, optional): The source of the visual stimulus. Defaults to None.
        """
        return self._observe(
            stimulus={
                "type": "VISUAL",
                "content": visual_description,
                "source": name_or_empty(source),
            },
            max_content_length=max_content_length,
        )

    @config_manager.config_defaults(max_content_length="max_content_display_length")
    def think(self, thought, max_content_length=None):
        """
        Forces the agent to think about something and updates its internal cognitive state.

        """
        return self._observe(
            stimulus={
                "type": "THOUGHT",
                "content": thought,
                "source": name_or_empty(self),
            },
            max_content_length=max_content_length,
        )

    @config_manager.config_defaults(max_content_length="max_content_display_length")
    def internalize_goal(
        self, goal, max_content_length=None
    ):
        """
        Internalizes a goal and updates its internal cognitive state.
        """
        return self._observe(
            stimulus={
                "type": "INTERNAL_GOAL_FORMULATION",
                "content": goal,
                "source": name_or_empty(self),
            },
            max_content_length=max_content_length,
        )

    @transactional()
    @config_manager.config_defaults(max_content_length="max_content_display_length")
    def _observe(self, stimulus, max_content_length=None):
        stimuli = [stimulus]

        content = {"stimuli": stimuli}

        logger.debug(f"[{self.name}] Observing stimuli: {content}")

        # whatever comes from the outside will be interpreted as coming from 'user', simply because
        # this is the counterpart of 'assistant'

        self.store_in_memory({'role': 'user', 'content': content, 
                              'type': 'stimulus',
                              'simulation_timestamp': self.iso_datetime()})

        if TinyPerson.communication_display:
            self._display_communication(
                role="user",
                content=content,
                kind="stimuli",
                simplified=True,
max_content_length=max_content_length,
            )
        
        # count the stimuli as this can be useful for taking decisions later
        self.stimuli_count += 1

        return self  # allows easier chaining of methods

    @transactional()
    def listen_and_act(
        self,
        speech,
        return_actions=False,
        max_content_length=None,
    ):
        """
        Convenience method that combines the `listen` and `act` methods.
        """

        self.listen(speech, max_content_length=max_content_length)
        return self.act(
            return_actions=return_actions, max_content_length=max_content_length
        )

    @transactional()
    @config_manager.config_defaults(max_content_length="max_content_display_length")
    def see_and_act(
        self,
        visual_description,
        return_actions=False,
        max_content_length=None,
    ):
        """
        Convenience method that combines the `see` and `act` methods.
        """

        self.see(visual_description, max_content_length=max_content_length)
        return self.act(
            return_actions=return_actions, max_content_length=max_content_length
        )

    @transactional()
    @config_manager.config_defaults(max_content_length="max_content_display_length")
    def think_and_act(
        self,
        thought,
        return_actions=False,
        max_content_length=None,
    ):
        """
        Convenience method that combines the `think` and `act` methods.
        """

        self.think(thought, max_content_length=max_content_length)
        return self.act(return_actions=return_actions, max_content_length=max_content_length)

    def read_documents_from_folder(self, documents_path:str):
        """
        Reads documents from a directory and loads them into the semantic memory.
        """
        logger.info(f"Setting documents path to {documents_path} and loading documents.")

        self.semantic_memory.add_documents_path(documents_path)
    
    def read_document_from_file(self, file_path:str):
        """
        Reads a document from a file and loads it into the semantic memory.
        """
        logger.info(f"Reading document from file: {file_path}")

        self.semantic_memory.add_document_path(file_path)
    
    def read_documents_from_web(self, web_urls:list):
        """
        Reads documents from web URLs and loads them into the semantic memory.
        """
        logger.info(f"Reading documents from the following web URLs: {web_urls}")

        self.semantic_memory.add_web_urls(web_urls)
    
    def read_document_from_web(self, web_url:str):
        """
        Reads a document from a web URL and loads it into the semantic memory.
        """
        logger.info(f"Reading document from web URL: {web_url}")

        self.semantic_memory.add_web_url(web_url)
    
    @transactional()
    def move_to(self, location, context=[]):
        """
        Moves to a new location and updates its internal cognitive state.
        """
        self._mental_state["location"] = location

        # context must also be updated when moved, since we assume that context is dictated partly by location.
        self.change_context(context)

    @transactional()
    def change_context(self, context: list):
        """
        Changes the context and updates its internal cognitive state.
        """
        self._mental_state["context"] = {
            "description": item for item in context
        }

        self._update_cognitive_state(context=context)

    @transactional()
    def make_agent_accessible(
        self,
        agent: Self,
        relation_description: str = "An agent I can currently interact with.",
    ):
        """
        Makes an agent accessible to this agent.
        """
        if agent not in self._accessible_agents:
            self._accessible_agents.append(agent)
            self._mental_state["accessible_agents"].append(
                {"name": agent.name, "relation_description": relation_description}
            )
        else:
            logger.warning(
                f"[{self.name}] Agent {agent.name} is already accessible to {self.name}."
            )
    @transactional()
    def make_agents_accessible(self, agents: list, relation_description: str = "An agent I can currently interact with."):
        """
        Makes a list of agents accessible to this agent.
        """
        for agent in agents:
            self.make_agent_accessible(agent, relation_description)

    @transactional()
    def make_agent_inaccessible(self, agent: Self):
        """
        Makes an agent inaccessible to this agent.
        """
        if agent in self._accessible_agents:
            self._accessible_agents.remove(agent)
        else:
            logger.warning(
                f"[{self.name}] Agent {agent.name} is already inaccessible to {self.name}."
            )

    @transactional()
    def make_all_agents_inaccessible(self):
        """
        Makes all agents inaccessible to this agent.
        """
        self._accessible_agents = []
        self._mental_state["accessible_agents"] = []

    @property
    def accessible_agents(self):
        """
        Property to access the list of accessible agents.
        """
        return self._accessible_agents

    ###########################################################
    # Internal cognitive state changes
    ###########################################################
    @transactional()
    def _update_cognitive_state(
        self, goals=None, context=None, attention=None, emotions=None
    ):
        """
        Update the TinyPerson's cognitive state.
        """

        # Update current datetime. The passage of time is controlled by the environment, if any.
        if self.environment is not None and self.environment.current_datetime is not None:
            self._mental_state["datetime"] = utils.pretty_datetime(self.environment.current_datetime)

        # update current goals
        if goals is not None:
            self._mental_state["goals"] = goals

        # update current context
        if context is not None:
            self._mental_state["context"] = context

        # update current attention
        if attention is not None:
            self._mental_state["attention"] = attention

        # update current emotions
        if emotions is not None:
            self._mental_state["emotions"] = emotions
        

        # update relevant memories for the current situation
        # Create the target string for relevance (reusing logic from retrieve_relevant_memories_for_current_context)
        context_str = str(self._mental_state["context"])
        goals_str = str(self._mental_state["goals"])
        attention_str = str(self._mental_state["attention"])
        emotions_str = str(self._mental_state["emotions"])
        # Use a smaller number of recent memories for this specific context string to avoid excessive length
        recent_episodic_for_target = "\n".join([f"  - {json.dumps(m.get('content'))}" for m in self.retrieve_memories(first_n=0, last_n=5, max_content_length=100, include_omission_info=False)])

        relevance_target_for_episodic = f"""
        Current Context: {context_str}
        Current Goals: {goals_str}
        Current Attention: {attention_str}
        Current Emotions: {emotions_str}
        Recent Activities:
        {recent_episodic_for_target}
        """

        relevant_episodic_memories = self.retrieve_relevant_episodic_memories(relevance_target_for_episodic, top_k=3) # Retrieve, for example, top 3

        # Combine relevant semantic and episodic memories for the mental state
        combined_memory_context = []
        if relevant_episodic_memories:
            combined_memory_context.append("Relevant past experiences (episodic):")
            for mem in relevant_episodic_memories:
                timestamp = mem.get('simulation_timestamp', 'unknown time')
                content = mem.get('content', {})
                if isinstance(content, dict): # content can be a dict for actions/stimuli
                    # Safely access nested content based on 'type'
                    if mem.get('type') == 'action':
                        actual_content = content.get('content', str(content))
                    elif mem.get('type') == 'stimulus':
                        # Stimuli content can be a list; take the first one's content
                        stimuli_list = content.get('stimuli', [])
                        if stimuli_list and isinstance(stimuli_list, list) and len(stimuli_list) > 0:
                            actual_content = stimuli_list[0].get('content', str(stimuli_list[0]))
                        else: # Fallback for unexpected structure
                            actual_content = str(content)
                    else: # Fallback for unknown type
                        actual_content = str(content)
                else: # if content is already a string
                    actual_content = str(content)

                # Ensure actual_content is a string
                if not isinstance(actual_content, str):
                    actual_content = json.dumps(actual_content) # Serialize if it's still not a string (e.g. a list or dict)

                combined_memory_context.append(f"- At {timestamp}: {actual_content[:150]}") # Truncate for brevity
            combined_memory_context.append("\n") # Separator

        # Append existing relevant semantic memories
        retrieved_semantic_memories = self.retrieve_relevant_memories_for_current_context() # Call it again
        if retrieved_semantic_memories:
            combined_memory_context.append("Relevant general knowledge (semantic):")
            # retrieved_semantic_memories is a list of strings, so extend is appropriate
            combined_memory_context.extend(retrieved_semantic_memories)

        self._mental_state["memory_context"] = "\n".join(combined_memory_context)


        self.reset_prompt()

    ###########################################################
    # Memory management
    ###########################################################
    ###########################################################
    # Memory management
    ###########################################################
    def store_in_memory(self, value: Any) -> list:
        self.episodic_memory.store(value)
        self._extract_and_store_semantic_insight(value)

        self._current_episode_event_count += 1
        logger.debug(f"[{self.name}] Current episode event count: {self._current_episode_event_count}.")

        if self._current_episode_event_count >= self.MAX_EPISODE_LENGTH:
            logger.warning(f"[{self.name}] Episode length exceeded {self.MAX_EPISODE_LENGTH} events. Committing episode to memory.")
            self.consolidate_episode_memories()

    def _extract_and_store_semantic_insight(self, episodic_entry: dict):
        try:
            entry_summary = f"Event Type: {episodic_entry.get('type')}, Content: {episodic_entry.get('content')}"
            if episodic_entry.get('simulation_timestamp'):
                entry_summary += f", Timestamp: {episodic_entry.get('simulation_timestamp')}"

            prompt_messages = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that extracts key insights or facts from event descriptions. "
                               "The insight should be a concise statement. If no clear, distinct fact or insight can be extracted, output the exact string 'None'."
                },
                {
                    "role": "user",
                    "content": f"From the following event experienced by an agent: {entry_summary}. Extracted Insight:"
                }
            ]

            llm_response = openai_utils.client().send_message(prompt_messages, temperature=0.0, max_tokens=60)
            insight_text = llm_response.get('content', "").strip()

            if insight_text and insight_text.lower() != 'none' and insight_text != "":
                logger.debug(f"[{self.name}] Extracted semantic insight: '{insight_text}' from event: {entry_summary}")

                semantic_payload = {
                    'type': 'semantic_insight',
                    'simulation_timestamp': self.iso_datetime(),
                    'content': insight_text,
                    'source_event_type': episodic_entry.get('type'),
                    'source_event_timestamp': episodic_entry.get('simulation_timestamp')
                }
                self.semantic_memory.store(semantic_payload)
            else:
                logger.debug(f"[{self.name}] No distinct semantic insight extracted from event: {entry_summary}")

        except Exception as e:
            logger.error(f"[{self.name}] Error in _extract_and_store_semantic_insight: {e}", exc_info=True)

    def consolidate_episode_memories(self):
        """
        Applies all memory consolidation or transformation processes appropriate to the conclusion of one simulation episode.
        """
        if self._current_episode_event_count > self.MIN_EPISODE_LENGTH:
            logger.debug(f"[{self.name}] ***** Consolidating current episode memories into semantic memory *****")

            if config_manager.get("enable_memory_consolidation"):
                episodic_consolidator = EpisodicConsolidator()
                episode = self.episodic_memory.get_current_episode(item_types=["action", "stimulus"])
                logger.debug(f"[{self.name}] Current episode: {episode}")

                consolidated = episodic_consolidator.process(
                    episode,
                    timestamp=self._mental_state["datetime"],
                    context=self._mental_state,
                    persona=self.minibio()
                )["consolidation"]

                if consolidated:
                    logger.info(f"[{self.name}] Consolidating current {len(episode)} episodic events as consolidated semantic memories.")
                    logger.debug(f"[{self.name}] Consolidated memories: {consolidated}")
                    self.semantic_memory.store_all(consolidated)
                else:
                    logger.debug(f"[{self.name}] No memories to consolidate from the current episode.")
            else:
                logger.warning(f"[{self.name}] Memory consolidation is disabled by config.")

            self.episodic_memory.commit_episode()
            self._current_episode_event_count = 0
            logger.debug(f"[{self.name}] Current episode event count reset to 0 after consolidation.")


    def optimize_memory(self):
        # TODO: This method is intended to house more sophisticated memory optimization strategies,
        # including summarization/condensation of episodic memory and optimization of semantic memory.
        # Currently, direct modification of EpisodicMemory class for condensation is facing tooling issues.
        logger.info(f"[{self.name}] optimize_memory() called. Future optimization logic to be implemented here.")
        pass

    def clear_episodic_memory(self, max_prefix_to_clear=None, max_suffix_to_clear=None):
        """
        Clears the episodic memory, causing a permanent "episodic amnesia". Note that this does not
        change other memories, such as semantic memory.  
        """
        self.episodic_memory.clear(max_prefix_to_clear=max_prefix_to_clear, max_suffix_to_clear=max_suffix_to_clear)

    def retrieve_memories(self, first_n: int, last_n: int, include_omission_info:bool=True, max_content_length:int=None) -> list:
        episodes = self.episodic_memory.retrieve(first_n=first_n, last_n=last_n, include_omission_info=include_omission_info)

        if max_content_length is not None:
            episodes = utils.truncate_actions_or_stimuli(episodes, max_content_length)

        return episodes


    def retrieve_recent_memories(self, max_content_length:int=None) -> list:
        episodes = self.episodic_memory.retrieve_recent()

        if max_content_length is not None:
            episodes = utils.truncate_actions_or_stimuli(episodes, max_content_length)

        return episodes

    def retrieve_relevant_memories(self, relevance_target:str, top_k=20) -> list:
        relevant = self.semantic_memory.retrieve_relevant(relevance_target, top_k=top_k)

        return relevant

    def retrieve_relevant_episodic_memories(self, relevance_target: str, top_k=5) -> list:
        """Retrieves relevant episodic memories based on a target string."""
        if hasattr(self, 'episodic_memory') and hasattr(self.episodic_memory, 'retrieve_relevant'):
            return self.episodic_memory.retrieve_relevant(relevance_target, top_k=top_k)
        return []

    def retrieve_relevant_memories_for_current_context(self, top_k=7) -> list:
        # current context is composed of th recent memories, plus context, goals, attention, and emotions
        context = self._mental_state["context"]
        goals = self._mental_state["goals"]
        attention = self._mental_state["attention"]
        emotions = self._mental_state["emotions"]

        recent_memories = "\n".join([f"  - {json.dumps(m.get('content'))}" for m in self.retrieve_memories(first_n=0, last_n=20, max_content_length=500, include_omission_info=False)])


        # put everything together in a nice markdown string to fetch relevant memories
        target = f"""
        Current Context: {context}
        Current Goals: {goals}
        Current Attention: {attention}
        Current Emotions: {emotions}
        Selected Episodic Memories (from oldest to newest):
        {recent_memories}
        """

        logger.debug(f"Retrieving relevant memories for contextual target: {target}")

        return self.retrieve_relevant_memories(target, top_k=top_k)

    def reflect_and_synthesize_knowledge(self, num_episodic_memories_to_reflect_on: int = 20):
        """
        Reflects on recent episodic memories to extract key insights and store them in semantic memory.
        """
        if not hasattr(self, 'episodic_memory') or not hasattr(self, 'semantic_memory'):
            logger.warning(f"[{self.name}] Missing episodic or semantic memory, skipping reflection.")
            return

        recent_episodes = self.episodic_memory.retrieve_last(n=num_episodic_memories_to_reflect_on, include_omission_info=False)

        if not recent_episodes:
            logger.info(f"[{self.name}] No recent episodic memories to reflect upon.")
            return

        # Format memories for the LLM prompt
        formatted_episodes_for_prompt = []
        for i, mem in enumerate(recent_episodes):
            timestamp = mem.get('simulation_timestamp', 'unknown time')
            role = mem.get('role', 'unknown_role')
            mem_content = mem.get('content', {})

            # Extract primary content from action or stimulus
            actual_text_content = ""
            if mem.get('type') == 'action' and isinstance(mem_content, dict):
                action_details = mem_content.get('action', {})
                actual_text_content = f"Action: {action_details.get('type')}, Content: {action_details.get('content', '')}"
            elif mem.get('type') == 'stimulus' and isinstance(mem_content, dict):
                stimulus_details = mem_content.get('stimuli', [{}])[0] # Taking the first stimulus
                actual_text_content = f"Stimulus: {stimulus_details.get('type')}, Content: {stimulus_details.get('content', '')}, Source: {stimulus_details.get('source', '')}"
            elif isinstance(mem_content, str): # Fallback for simpler content
                actual_text_content = mem_content
            else: # Fallback for other complex content
                actual_text_content = json.dumps(mem_content)

            formatted_episodes_for_prompt.append(f"Memory {i+1} (Timestamp: {timestamp}, Role: {role}):\n{actual_text_content}\n---")

        episodes_string = "\n".join(formatted_episodes_for_prompt)

        # Define the system prompt for the LLM
        reflection_system_prompt = textwrap.dedent("""
        You are a reflective agent. Your task is to analyze a sequence of recent experiences (episodic memories) provided by an agent.
        Identify key insights, patterns, important facts learned, or general conclusions that can be drawn from these experiences.
        These synthesized insights will be stored as semantic knowledge.
        Focus on extracting information that is novel, significant, or contributes to a better understanding of the agent's world, tasks, or interactions.
        Avoid trivial observations.
        Respond with a JSON list of strings, where each string is a concise piece of synthesized knowledge. For example:
        ["Finding X often leads to Y.", "User Z prefers options that are clearly explained.", "The system requires authentication before accessing sensitive data."]
        If no significant insights can be derived, return an empty list [].
        """)

        reflection_user_prompt = f"Here are my recent experiences:\n\n{episodes_string}\n\nPlease provide your synthesized knowledge based on these experiences, in the specified JSON format."

        messages_for_llm = [
            {"role": "system", "content": reflection_system_prompt},
            {"role": "user", "content": reflection_user_prompt}
        ]

        logger.info(f"[{self.name}] Requesting LLM to synthesize knowledge from {len(recent_episodes)} episodes.")

        synthesized_knowledge_list = []
        try:
            # Assuming openai_utils.client().send_message can handle this structure
            # For response_model, if list[str] is not directly supported as a Pydantic model,
            # we would expect a raw string and parse it using utils.extract_json.
            # Let's assume 'send_message' returns a dict like {'role': 'assistant', 'content': '["insight1", ... ]'}
            # or if response_model=list[str] works, it might directly return the list.
            llm_response_raw = openai_utils.client().send_message(messages_for_llm) # Potentially add response_model=list[str] if supported

            if isinstance(llm_response_raw, dict) and 'content' in llm_response_raw:
                # This case assumes the LLM returns a JSON string within the 'content' field.
                extracted_json = utils.extract_json(llm_response_raw['content'])
                if isinstance(extracted_json, list):
                    synthesized_knowledge_list = [item for item in extracted_json if isinstance(item, str)]
                else:
                    logger.warning(f"[{self.name}] LLM reflection response JSON content was not a list: {extracted_json}")
            # This case would be if send_message with response_model=list[str] (or similar) directly returns a list.
            # elif isinstance(llm_response_raw, list):
            #    synthesized_knowledge_list = [item for item in llm_response_raw if isinstance(item, str)]
            else:
                # Fallback or if the response_model=list[str] was intended to be used with a Pydantic model:
                # If the direct output of send_message (without a specific response_model for list[str])
                # is expected to be the list itself due to how `openai_utils` might be set up.
                # This part is speculative based on `response_model=list[str]` in the prompt.
                # For now, we'll rely on the dict structure as it's more common with current `send_message` usage.
                logger.warning(f"[{self.name}] Unexpected LLM reflection response format: {llm_response_raw}")

        except Exception as e:
            logger.error(f"[{self.name}] Error during LLM call for knowledge synthesis: {e}")
            # synthesized_knowledge_list remains []

        if not synthesized_knowledge_list:
            logger.info(f"[{self.name}] No new knowledge synthesized from reflection.")
            return

        logger.info(f"[{self.name}] Synthesized {len(synthesized_knowledge_list)} pieces of knowledge.")
        for knowledge_statement in synthesized_knowledge_list:
            if not isinstance(knowledge_statement, str) or not knowledge_statement.strip():
                logger.warning(f"[{self.name}] Skipping empty or invalid knowledge statement: {knowledge_statement}")
                continue

            new_knowledge_item = {
                'type': 'synthesized_knowledge',
                'content': knowledge_statement,
                'source_reflection_timestamp': self.iso_datetime(), # Timestamp of when reflection occurred
                'reflected_episodes_count': len(recent_episodes)
            }
            # Assuming semantic_memory.store() can handle a dict and will process it into a Document or suitable format.
            self.semantic_memory.store(new_knowledge_item)

        logger.info(f"[{self.name}] Finished storing synthesized knowledge.")


    ###########################################################
    # Inspection conveniences
    ###########################################################

    def last_remembered_action(self, ignore_done:bool=True):
        """
        Returns the last remembered action.

        Args:
            ignore_done (bool): Whether to ignore the "DONE" action or not. Defaults to True.
        """
        action = None 
        
        memory_items_list = self.episodic_memory.retrieve_last(include_omission_info=False, item_type="action")

        if len(memory_items_list) > 0:
            # iterate from last to first while the action type is not "DONE"
            for candidate_item in memory_items_list[::-1]:
                if candidate_item["content"]["action"]["type"] != "DONE":
                    action = candidate_item["content"]["action"]
                    break
                else:
                    if ignore_done:
                        continue
                    else:
                        action = candidate_item["content"]["action"]
                        break

        return action 


    ###########################################################
    # Communication display and action execution 
    ###########################################################

    def _display_communication(
        self,
        role,
        content,
        kind,
        simplified=True,
        max_content_length=default["max_content_display_length"],
    ):
        """
        Displays the current communication and stores it in a buffer for later use.
        """
        # CONCURRENT PROTECTION, as we'll access shared display buffers
        with concurrent_agent_action_lock:
            if kind == "stimuli":
                rendering = self._pretty_stimuli(
                    role=role,
                    content=content,
                    simplified=simplified,
                    max_content_length=max_content_length,
                )
                source = content["stimuli"][0].get("source", None)
                target = self.name
                
            elif kind == "action":
                rendering = self._pretty_action(
                    role=role,
                    content=content,
                    simplified=simplified,
                    max_content_length=max_content_length,
                )
                source = self.name
                target = content["action"].get("target", None)

            else:
                raise ValueError(f"Unknown communication kind: {kind}")

            # if the agent has no parent environment, then it is a free agent and we can display the communication.
            # otherwise, the environment will display the communication instead. This is important to make sure that
            # the communication is displayed in the correct order, since environments control the flow of their underlying
            # agents.
            if self.environment is None:
                self._push_and_display_latest_communication({"kind": kind, "rendering":rendering, "content": content, "source":source, "target": target})
            else:
                self.environment._push_and_display_latest_communication({"kind": kind, "rendering":rendering, "content": content, "source":source, "target": target})

    def _push_and_display_latest_communication(self, communication):
        """
        Pushes the latest communications to the agent's buffer.
        """
        self._displayed_communications_buffer.append(communication)
        print(communication["rendering"])

    def pop_and_display_latest_communications(self):
        """
        Pops the latest communications and displays them.
        """
        communications = self._displayed_communications_buffer
        self._displayed_communications_buffer = []

        for communication in communications:
            print(communication["rendering"])

        return communications

    def clear_communications_buffer(self):
        """
        Cleans the communications buffer.
        """
        self._displayed_communications_buffer = []

    @transactional()
    def pop_latest_actions(self) -> list:
        """
        Returns the latest actions performed by this agent. Typically used
        by an environment to consume the actions and provide the appropriate
        environmental semantics to them (i.e., effects on other agents).
        """
        actions = self._actions_buffer
        self._actions_buffer = []
        return actions

    @transactional()
    def pop_actions_and_get_contents_for(
        self, action_type: str, only_last_action: bool = True
    ) -> list:
        """
        Returns the contents of actions of a given type performed by this agent.
        Typically used to perform inspections and tests.

        Args:
            action_type (str): The type of action to look for.
            only_last_action (bool, optional): Whether to only return the contents of the last action. Defaults to False.
        """
        actions = self.pop_latest_actions()
        # Filter the actions by type
        actions = [action for action in actions if action["type"] == action_type]

        # If interested only in the last action, return the latest one
        if only_last_action:
            return actions[-1].get("content", "")

        # Otherwise, return all contents from the filtered actions
        return "\n".join([action.get("content", "") for action in actions])

    #############################################################################################
    # Formatting conveniences
    #
    # For rich colors,
    #    see: https://rich.readthedocs.io/en/latest/appendix/colors.html#appendix-colors
    #############################################################################################

    def __repr__(self):
        return f"TinyPerson(name='{self.name}')"

    @transactional()
    def minibio(self, extended=True, requirements=None):
        """
        Returns a mini-biography of the TinyPerson.

        Args:
            extended (bool): Whether to include extended information or not.
            requirements (str): Additional requirements for the biography (e.g., focus on a specific aspect relevant for the scenario).

        Returns:
            str: The mini-biography.
        """

        # if occupation is a dict and has a "title" key, use that as the occupation 
        if isinstance(self._persona['occupation'], dict) and 'title' in self._persona['occupation']:
            occupation = self._persona['occupation']['title']
        else:
            occupation = self._persona['occupation']

        base_biography = f"{self.name} is a {self._persona['age']} year old {occupation}, {self._persona['nationality']}, currently living in {self._persona['residence']}."

        if self._extended_agent_summary is None and extended:
            logger.debug(f"Generating extended agent summary for {self.name}.")
            self._extended_agent_summary = LLMChat(
                                                system_prompt=f"""
                                                You are given a short biography of an agent, as well as a detailed specification of his or her other characteristics
                                                You must then produce a short paragraph (3 or 4 sentences) that **complements** the short biography, adding details about
                                                personality, interests, opinions, skills, etc. Do not repeat the information already given in the short biography.
                                                repeating the information already given. The paragraph should be coherent, consistent and comprehensive. All information
                                                must be grounded on the specification, **do not** create anything new.

                                                {"Additional constraints: "+ requirements if requirements is not None else ""}
                                                """, 

                                                user_prompt=f"""
                                                **Short biography:** {base_biography}

                                                **Detailed specification:** {self._persona}
                                                """).call()

        if extended:
            biography = f"{base_biography} {self._extended_agent_summary}"
        else:
            biography = base_biography

        return biography

    def pp_current_interactions(
        self,
        simplified=True,
        skip_system=True,
        max_content_length=default["max_content_display_length"],
        first_n=None, 
        last_n=None, 
        include_omission_info:bool=True
    ):
        """
        Pretty prints the current messages.
        """
        print(
            self.pretty_current_interactions(
                simplified=simplified,
                skip_system=skip_system,
                max_content_length=max_content_length,
                first_n=first_n,
                last_n=last_n,
                include_omission_info=include_omission_info
            )
        )

    def pp_last_interactions(
        self,
        n=3,
        simplified=True,
        skip_system=True,
        max_content_length=default["max_content_display_length"],
        include_omission_info:bool=True
    ):
        """
        Pretty prints the last n messages. Useful to examine the conclusion of an experiment.
        """
        print(
            self.pretty_current_interactions(
                simplified=simplified,
                skip_system=skip_system,
                max_content_length=max_content_length,
                first_n=None,
                last_n=n,
                include_omission_info=include_omission_info
            )
        )

    def pretty_current_interactions(self, simplified=True, skip_system=True, max_content_length=default["max_content_display_length"], first_n=None, last_n=None, include_omission_info:bool=True):
      """
      Returns a pretty, readable, string with the current messages.
      """
      lines = [f"**** BEGIN SIMULATION TRAJECTORY FOR {self.name} ****"]
      last_step = 0
      for i, message in enumerate(self.episodic_memory.retrieve(first_n=first_n, last_n=last_n, include_omission_info=include_omission_info)):
        try:
            if not (skip_system and message['role'] == 'system'):
                msg_simplified_type = ""
                msg_simplified_content = ""
                msg_simplified_actor = ""

                last_step = i
                lines.append(f"Agent simulation trajectory event #{i}:")
                lines.append(self._pretty_timestamp(message['role'], message['simulation_timestamp']))

                if message["role"] == "system":
                    msg_simplified_actor = "SYSTEM"
                    msg_simplified_type = message["role"]
                    msg_simplified_content = message["content"]

                    lines.append(
                        f"[dim] {msg_simplified_type}: {msg_simplified_content}[/]"
                    )

                elif message["role"] == "user":
                    lines.append(
                        self._pretty_stimuli(
                            role=message["role"],
                            content=message["content"],
                            simplified=simplified,
                            max_content_length=max_content_length,
                        )
                    )

                elif message["role"] == "assistant":
                    lines.append(
                        self._pretty_action(
                            role=message["role"],
                            content=message["content"],
                            simplified=simplified,
                            max_content_length=max_content_length,
                        )
                    )
                else:
                    lines.append(f"{message['role']}: {message['content']}")
        except:
            # print(f"ERROR: {message}")
            continue

      lines.append(f"The last agent simulation trajectory event number was {last_step}, thus the current number of the NEXT POTENTIAL TRAJECTORY EVENT is {last_step + 1}.")
      lines.append(f"**** END SIMULATION TRAJECTORY FOR {self.name} ****\n\n")
      return "\n".join(lines)

    def _pretty_stimuli(
        self,
        role,
        content,
        simplified=True,
        max_content_length=default["max_content_display_length"],
    ) -> list:
        """
        Pretty prints stimuli.
        """

        lines = []
        msg_simplified_actor = "USER"
        for stimus in content["stimuli"]:
            if simplified:
                if stimus["source"] != "":
                    msg_simplified_actor = stimus["source"]

                else:
                    msg_simplified_actor = "USER"

                msg_simplified_type = stimus["type"]
                msg_simplified_content = utils.break_text_at_length(
                    stimus["content"], max_length=max_content_length
                )

                indent = " " * len(msg_simplified_actor) + "      > "
                msg_simplified_content = textwrap.fill(
                    msg_simplified_content,
                    width=TinyPerson.PP_TEXT_WIDTH,
                    initial_indent=indent,
                    subsequent_indent=indent,
                )

                #
                # Using rich for formatting. Let's make things as readable as possible!
                #

                rich_style = utils.RichTextStyle.get_style_for("stimulus", msg_simplified_type)
                lines.append(
                    f"[{rich_style}][underline]{msg_simplified_actor}[/] --> [{rich_style}][underline]{self.name}[/]: [{msg_simplified_type}] \n{msg_simplified_content}[/]"
                )
            else:
                lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _pretty_action(
        self,
        role,
        content,
        simplified=True,
        max_content_length=default["max_content_display_length"],
    ) -> str:
        """
        Pretty prints an action.
        """
        if simplified:
            msg_simplified_actor = self.name
            msg_simplified_type = content["action"]["type"]
            msg_simplified_content = utils.break_text_at_length(
                content["action"].get("content", ""), max_length=max_content_length
            )

            indent = " " * len(msg_simplified_actor) + "      > "
            msg_simplified_content = textwrap.fill(
                msg_simplified_content,
                width=TinyPerson.PP_TEXT_WIDTH,
                initial_indent=indent,
                subsequent_indent=indent,
            )

            #
            # Using rich for formatting. Let's make things as readable as possible!
            #
            rich_style = utils.RichTextStyle.get_style_for("action", msg_simplified_type)
            return f"[{rich_style}][underline]{msg_simplified_actor}[/] acts: [{msg_simplified_type}] \n{msg_simplified_content}[/]"
        
        else:
            return f"{role}: {content}"
    
    def _pretty_timestamp(
        self,
        role,
        timestamp,
    ) -> str:
        """
        Pretty prints a timestamp.
        """
        return f">>>>>>>>> Date and time of events: {timestamp}"

    def iso_datetime(self) -> str:
        """
        Returns the current datetime of the environment, if any.

        Returns:
            datetime: The current datetime of the environment in ISO forat.
        """
        if self.environment is not None and self.environment.current_datetime is not None:
            return self.environment.current_datetime.isoformat()
        else:
            return None

    ###########################################################
    # IO
    ###########################################################

    def save_specification(self, path, include_mental_faculties=True, include_memory=False, include_mental_state=False):
        """
        Saves the current configuration to a JSON file.
        """
        
        suppress_attributes = []

        # should we include the mental faculties?
        if not include_mental_faculties:
            suppress_attributes.append("_mental_faculties")

        # should we include the memory?
        if not include_memory:
            suppress_attributes.append("episodic_memory")
            suppress_attributes.append("semantic_memory")

        # should we include the mental state?
        if not include_mental_state:
            suppress_attributes.append("_mental_state")
        

        self.to_json(suppress=suppress_attributes, file_path=path,
                     serialization_type_field_name="type")

    
    @staticmethod
    def load_specification(path_or_dict, suppress_mental_faculties=False, suppress_memory=False, suppress_mental_state=False, 
                           auto_rename_agent=False, new_agent_name=None):
        """
        Loads a JSON agent specification.

        Args:
            path_or_dict (str or dict): The path to the JSON file or the dictionary itself.
            suppress_mental_faculties (bool, optional): Whether to suppress loading the mental faculties. Defaults to False.
            suppress_memory (bool, optional): Whether to suppress loading the memory. Defaults to False.
            suppress_memory (bool, optional): Whether to suppress loading the memory. Defaults to False.
            suppress_mental_state (bool, optional): Whether to suppress loading the mental state. Defaults to False.
            auto_rename_agent (bool, optional): Whether to auto rename the agent. Defaults to False.
            new_agent_name (str, optional): The new name for the agent. Defaults to None.
        """

        suppress_attributes = []

        # should we suppress the mental faculties?
        if suppress_mental_faculties:
            suppress_attributes.append("_mental_faculties")

        # should we suppress the memory?
        if suppress_memory:
            suppress_attributes.append("episodic_memory")
            suppress_attributes.append("semantic_memory")
        
        # should we suppress the mental state?
        if suppress_mental_state:
            suppress_attributes.append("_mental_state")

        return TinyPerson.from_json(json_dict_or_path=path_or_dict, suppress=suppress_attributes, 
                                    serialization_type_field_name="type",
                                    post_init_params={"auto_rename_agent": auto_rename_agent, "new_agent_name": new_agent_name})
    @staticmethod
    def load_specifications_from_folder(folder_path:str, file_suffix=".agent.json", suppress_mental_faculties=False, 
                                        suppress_memory=False, suppress_mental_state=False, auto_rename_agent=False, 
                                        new_agent_name=None) -> list:
        """	
        Loads all JSON agent specifications from a folder.

        Args:
            folder_path (str): The path to the folder containing the JSON files.
            file_suffix (str, optional): The suffix of the JSON files. Defaults to ".agent.json".
            suppress_mental_faculties (bool, optional): Whether to suppress loading the mental faculties. Defaults to False.
            suppress_memory (bool, optional): Whether to suppress loading the memory. Defaults to False.
            suppress_mental_state (bool, optional): Whether to suppress loading the mental state. Defaults to False.
            auto_rename_agent (bool, optional): Whether to auto rename the agent. Defaults to False.
            new_agent_name (str, optional): The new name for the agent. Defaults to None.
        """

        agents = []
        for file in os.listdir(folder_path):
            if file.endswith(file_suffix):
                file_path = os.path.join(folder_path, file)
                agent = TinyPerson.load_specification(file_path, suppress_mental_faculties=suppress_mental_faculties,
                                                      suppress_memory=suppress_memory, suppress_mental_state=suppress_mental_state,
                                                      auto_rename_agent=auto_rename_agent, new_agent_name=new_agent_name)
                agents.append(agent)

        return agents
        


    def encode_complete_state(self) -> dict:
        """
        Encodes the complete state of the TinyPerson, including the current messages, accessible agents, etc.
        This is meant for serialization and caching purposes, not for exporting the state to the user.
        """
        to_copy = copy.copy(self.__dict__)

        # delete the logger and other attributes that cannot be serialized
        del to_copy["environment"]
        del to_copy["_mental_faculties"]
        del to_copy["action_generator"]

        to_copy["_accessible_agents"] = [agent.name for agent in self._accessible_agents]
        to_copy['episodic_memory'] = self.episodic_memory.to_json()
        to_copy['semantic_memory'] = self.semantic_memory.to_json()
        to_copy["_mental_faculties"] = [faculty.to_json() for faculty in self._mental_faculties]

        state = copy.deepcopy(to_copy)

        return state

    def decode_complete_state(self, state: dict) -> Self:
        """
        Loads the complete state of the TinyPerson, including the current messages,
        and produces a new TinyPerson instance.
        """
        state = copy.deepcopy(state)
        
        self._accessible_agents = [TinyPerson.get_agent_by_name(name) for name in state["_accessible_agents"]]
        self.episodic_memory = EpisodicMemory.from_json(state['episodic_memory'])
        self.semantic_memory = SemanticMemory.from_json(state['semantic_memory'])
        
        for i, faculty in enumerate(self._mental_faculties):
            faculty = faculty.from_json(state['_mental_faculties'][i])

        # delete fields already present in the state
        del state["_accessible_agents"]
        del state['episodic_memory']
        del state['semantic_memory']
        del state['_mental_faculties']

        # restore other fields
        self.__dict__.update(state)


        return self
    
    def create_new_agent_from_current_spec(self, new_name:str) -> Self:
        """
        Creates a new agent from the current agent's specification. 

        Args:
            new_name (str): The name of the new agent. Agent names must be unique in the simulation, 
              this is why we need to provide a new name.
        """
        new_agent = TinyPerson(name=new_name, spec_path=None)
        
        new_persona = copy.deepcopy(self._persona)
        new_persona['name'] = new_name

        new_agent._persona = new_persona

        return new_agent
        

    @staticmethod
    def add_agent(agent):
        """
        Adds an agent to the global list of agents. Agent names must be unique,
        so this method will raise an exception if the name is already in use.
        """
        if agent.name in TinyPerson.all_agents:
            raise ValueError(f"Agent name {agent.name} is already in use.")
        else:
            TinyPerson.all_agents[agent.name] = agent

    @staticmethod
    def has_agent(agent_name: str):
        """
        Checks if an agent is already registered.
        """
        return agent_name in TinyPerson.all_agents

    @staticmethod
    def set_simulation_for_free_agents(simulation):
        """
        Sets the simulation if it is None. This allows free agents to be captured by specific simulation scopes
        if desired.
        """
        for agent in TinyPerson.all_agents.values():
            if agent.simulation_id is None:
                simulation.add_agent(agent)

    @staticmethod
    def get_agent_by_name(name):
        """
        Gets an agent by name.
        """
        if name in TinyPerson.all_agents:
            return TinyPerson.all_agents[name]
        else:
            return None
    
    @staticmethod
    def all_agents_names():
        """
        Returns the names of all agents.
        """
        return list(TinyPerson.all_agents.keys())

    @staticmethod
    def clear_agents():
        """
        Clears the global list of agents.
        """
        TinyPerson.all_agents = {}
