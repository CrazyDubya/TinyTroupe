from tinytroupe.agent.mental_faculty import TinyMentalFaculty
from tinytroupe.agent.grounding import BaseSemanticGroundingConnector
import tinytroupe.utils as utils

from llama_index.core import Document
from typing import Any
import copy
import json
import uuid

#######################################################################################################################
# Memory mechanisms 
#######################################################################################################################

class TinyMemory(TinyMentalFaculty):
    """
    Base class for different types of memory.
    """

    def _preprocess_value_for_storage(self, value: Any) -> Any:
        """
        Preprocesses a value before storing it in memory.
        """
        # by default, we don't preprocess the value
        return value

    def _store(self, value: Any) -> None:
        """
        Stores a value in memory.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def store(self, value: dict) -> None:
        """
        Stores a value in memory.
        """
        self._store(self._preprocess_value_for_storage(value))
    
    def store_all(self, values: list) -> None:
        """
        Stores a list of values in memory.
        """
        for value in values:
            self.store(value)

    def retrieve(self, first_n: int, last_n: int, include_omission_info:bool=True) -> list:
        """
        Retrieves the first n and/or last n values from memory. If n is None, all values are retrieved.

        Args:
            first_n (int): The number of first values to retrieve.
            last_n (int): The number of last values to retrieve.
            include_omission_info (bool): Whether to include an information message when some values are omitted.

        Returns:
            list: The retrieved values.
        
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_recent(self) -> list:
        """
        Retrieves the n most recent values from memory.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_all(self) -> list:
        """
        Retrieves all values from memory.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_relevant(self, relevance_target:str, top_k=20) -> list:
        """
        Retrieves all values from memory that are relevant to a given target.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class EpisodicMemory(TinyMemory):
    """
    Provides episodic memory capabilities to an agent. Cognitively, episodic memory is the ability to remember specific events,
    or episodes, in the past. This class provides a simple implementation of episodic memory, where the agent can store and retrieve
    messages from memory.
    
    Subclasses of this class can be used to provide different memory implementations.
    """

    MEMORY_BLOCK_OMISSION_INFO = {'role': 'assistant', 'content': "Info: there were other messages here, but they were omitted for brevity.", 'simulation_timestamp': None}

    def __init__(
        self, fixed_prefix_length: int = 100, lookback_length: int = 100
    ) -> None:
        """
        Initializes the memory.

        Args:
            fixed_prefix_length (int): The fixed prefix length. Defaults to 20.
            lookback_length (int): The lookback length. Defaults to 20.
        """
        self.fixed_prefix_length = fixed_prefix_length
        self.lookback_length = lookback_length

        self.memory = []
        self.semantic_connector = BaseSemanticGroundingConnector(name="Episodic Memory Index")
        self.memory_id_map = {}

    def _store(self, value: Any) -> None:
        """
        Stores a value in memory.
        """
        memory_id = str(uuid.uuid4())
        self.memory.append(value)
        memory_idx = len(self.memory) - 1
        self.memory_id_map[memory_id] = memory_idx

        doc_text = json.dumps(value)
        document = Document(text=doc_text, metadata={'memory_id': memory_id, 'original_timestamp': value.get('simulation_timestamp')})
        self.semantic_connector.add_document(document)

    def count(self) -> int:
        """
        Returns the number of values in memory.
        """
        return len(self.memory)

    def retrieve(self, first_n: int, last_n: int, include_omission_info:bool=True) -> list:
        """
        Retrieves the first n and/or last n values from memory. If n is None, all values are retrieved.

        Args:
            first_n (int): The number of first values to retrieve.
            last_n (int): The number of last values to retrieve.
            include_omission_info (bool): Whether to include an information message when some values are omitted.

        Returns:
            list: The retrieved values.
        
        """

        omisssion_info = [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []

        # use the other methods in the class to implement
        if first_n is not None and last_n is not None:
            return self.retrieve_first(first_n) + omisssion_info + self.retrieve_last(last_n)
        elif first_n is not None:
            return self.retrieve_first(first_n)
        elif last_n is not None:
            return self.retrieve_last(last_n)
        else:
            return self.retrieve_all()

    def retrieve_recent(self, include_omission_info:bool=True) -> list:
        """
        Retrieves the n most recent values from memory.
        """
        omisssion_info = [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []

        # compute fixed prefix
        fixed_prefix = self.memory[: self.fixed_prefix_length] + omisssion_info

        # how many lookback values remain?
        remaining_lookback = min(
            len(self.memory) - len(fixed_prefix), self.lookback_length
        )

        # compute the remaining lookback values and return the concatenation
        if remaining_lookback <= 0:
            return fixed_prefix
        else:
            return fixed_prefix + self.memory[-remaining_lookback:]

    def retrieve_all(self) -> list:
        """
        Retrieves all values from memory.
        """
        return copy.copy(self.memory)

    def retrieve_relevant(self, relevance_target: str, top_k:int) -> list:
        """
        Retrieves top-k values from memory that are most relevant to a given target.
        """
        retrieved_nodes = self.semantic_connector.retrieve_relevant(relevance_target, top_k=top_k)
        relevant_memories = []
        for node_info in retrieved_nodes:
            metadata = node_info.get('metadata', {})
            memory_id = metadata.get('memory_id')
            if memory_id:
                memory_idx = self.memory_id_map.get(memory_id)
                # Check if memory_idx is not None and is a valid index
                if memory_idx is not None and 0 <= memory_idx < len(self.memory):
                    relevant_memories.append(self.memory[memory_idx])
        return relevant_memories

    def retrieve_first(self, n: int, include_omission_info:bool=True) -> list:
        """
        Retrieves the first n values from memory.
        """
        omisssion_info = [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []
        
        return self.memory[:n] + omisssion_info
    
    def retrieve_last(self, n: int, include_omission_info:bool=True) -> list:
        """
        Retrieves the last n values from memory.
        """
        omisssion_info = [EpisodicMemory.MEMORY_BLOCK_OMISSION_INFO] if include_omission_info else []

        return omisssion_info + self.memory[-n:]


@utils.post_init
class SemanticMemory(TinyMemory):
    """
    In Cognitive Psychology, semantic memory is the memory of meanings, understandings, and other concept-based knowledge unrelated to specific 
    experiences. It is not ordered temporally, and it is not about remembering specific events or episodes. This class provides a simple implementation
    of semantic memory, where the agent can store and retrieve semantic information.
    """

    serializable_attrs = ["memories"]

    def __init__(self, memories: list=None) -> None:
        self.memories = memories

        # @post_init ensures that _post_init is called after the __init__ method

    def _post_init(self): 
        """
        This will run after __init__, since the class has the @post_init decorator.
        It is convenient to separate some of the initialization processes to make deserialize easier.
        """

        if not hasattr(self, 'memories') or self.memories is None:
            self.memories = []

        self.semantic_grounding_connector = BaseSemanticGroundingConnector("Semantic Memory Storage")
        self.semantic_grounding_connector.add_documents(self._build_documents_from(self.memories))
    
        
    def _preprocess_value_for_storage(self, value: dict) -> Any:
        engram = None 

        if value['type'] == 'action':
            engram = f"# Fact\n" +\
                     f"I have performed the following action at date and time {value['simulation_timestamp']}:\n\n"+\
                     f" {value['content']}"
        
        elif value['type'] == 'stimulus':
            engram = f"# Stimulus\n" +\
                     f"I have received the following stimulus at date and time {value['simulation_timestamp']}:\n\n"+\
                     f" {value['content']}"

        elif value['type'] == 'synthesized_knowledge':
            engram = f"# Synthesized Knowledge (Reflected on: {value.get('source_reflection_timestamp', 'N/A')}, From: {value.get('reflected_episodes_count', 'N/A')} episodes)\n" +\
                     f"Insight: {value['content']}"

        # else: # Anything else here?
        # Ensure value is a dict before accessing type, otherwise return None or handle error
        elif not isinstance(value, dict):
            # If value is not a dict, we cannot process its type.
            # Depending on desired behavior, either return str(value) or None, or log an error.
            # For now, let's assume it should have been a dict if it reached here with an unhandled type.
            # Or, if direct string storage is allowed, this preprocessing step might not be called for them.
            return None # Or str(value) if strings should be stored as is.

        return engram

    # Override the base class store method to ensure _store receives the original dictionary
    def store(self, value: dict) -> None:
        """
        Stores a value in semantic memory.
        The value is expected to be a dictionary, which will be preprocessed.
        """
        if not isinstance(value, dict):
            # Optional: Log a warning or raise an error if value is not a dict,
            # as SemanticMemory expects dicts for preprocessing and metadata.
            # For now, let it pass to _store, which has some handling for strings.
            print(f"Warning: SemanticMemory.store called with non-dict value: {value}") # Or use logger
        self._store(value) # Pass the original dict to _store

    def _store(self, value: Any) -> None: # value here is the original dict
        engram_text = self._preprocess_value_for_storage(value) # value is the dict here

        if engram_text is None:
            # If value is a string, it might have been passed here directly if store() was called with a string.
            # The main store() method expects a dict, so this path implies an issue or direct _store call.
            if isinstance(value, str): # Allow storing raw strings if they bypass preprocessing
                engram_text = value
                metadata = {'type': 'raw_string'}
            else:
                # from tinytroupe.agent import logger # Add this import at the top of the file if not present
                # logger.warning(f"Preprocessing returned None for value: {value}. Skipping storage in semantic memory.")
                # For now, let's print, assuming logger might not be configured here.
                print(f"Warning: Preprocessing returned None for value: {value}. Skipping storage in semantic memory.")
                return

        # If engram_text was set by preprocessing a dict, or if value was a string.
        metadata = {}
        if isinstance(value, dict): # Only try to get metadata if value is a dict
            metadata = {
                'original_timestamp': value.get('simulation_timestamp') or value.get('source_reflection_timestamp'),
                'type': value.get('type')
            }

        # Ensure engram_text is not None before building document
        if engram_text:
             engram_doc = self._build_document_from(engram_text, metadata=metadata)
             self.semantic_grounding_connector.add_document(engram_doc)
        else:
             # print(f"Warning: Engram text is None for value: {value}. Document not built or stored.") # Redundant due to above check
             pass # Already handled
    
    def retrieve_relevant(self, relevance_target:str, top_k=20) -> list:
        """
        Retrieves all values from memory that are relevant to a given target.
        """
        return self.semantic_grounding_connector.retrieve_relevant(relevance_target, top_k)

    #####################################
    # Auxiliary compatibility methods
    #####################################

    def _build_document_from(self, memory_text: str, metadata: dict = None) -> Document:
        if metadata is None:
            metadata = {}
        return Document(text=str(memory_text), metadata=metadata)
    
    def _build_documents_from(self, memories: list) -> list:
        docs = []
        for mem_item in memories:
            if isinstance(mem_item, dict):
                # Attempt to extract text and metadata if it's a structured dict
                text_content = mem_item.get('content', str(mem_item))
                metadata = {
                    'original_timestamp': mem_item.get('simulation_timestamp') or mem_item.get('source_reflection_timestamp'),
                    'type': mem_item.get('type', 'unknown_init_type')
                }
                # Preprocess if it looks like a raw memory item that needs engram formatting
                # This ensures that during initial load, dicts are also formatted correctly.
                if mem_item.get('type') in ['action', 'stimulus', 'synthesized_knowledge']: # Add other types if they have specific preprocessing
                     processed_text = self._preprocess_value_for_storage(mem_item)
                     if processed_text: # Ensure preprocessing didn't return None
                         text_content = processed_text

                docs.append(self._build_document_from(text_content, metadata=metadata))
            elif isinstance(mem_item, str):
                # If it's just a string, treat it as text with no specific metadata initially
                # However, if SemanticMemory's .store() is consistently called with dicts,
                # string items in self.memories at init might represent already-processed engrams.
                docs.append(self._build_document_from(mem_item, metadata={'type': 'unknown_init_str'}))
            else:
                # Fallback for other types
                docs.append(self._build_document_from(str(mem_item), metadata={'type': 'unknown_init_fallback'}))
        return docs
    
   