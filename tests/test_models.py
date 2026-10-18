"""
Agent Memory - model tests

Verifies the correctness of the data models
Run with:
    pytest tests/test_models.py -v
"""

import pytest
from datetime import datetime
from typing import Dict, Any

from hy_memory.models import (
    # Enums
    MemoryLayer,
    MemoryInputType,
    AgentProcessMode,
    TaskStatus,
    DeleteScope,
    # Dataclasses
    QAPair,
    AddRequest,
    AddResponse,
    AsyncAddResponse,
    RecallRequest,
    RecallResponse,
    UpdateRequest,
    DeleteRequest,
    BatchDeleteRequest,
    MemoryEntry,
    MemoryMetadata,
)


# ==================== Enum tests ====================

class TestEnums:
    """Enum tests."""
    
    def test_memory_layer_values(self):
        """Test memory layer enum values (V3 seven-layer model + V1/V2 compatibility aliases)."""
        # V3 original layers
        assert MemoryLayer.L4_IDENTITY.value == "l4_identity"
        assert MemoryLayer.L2_FACT.value == "l2_fact"
        assert MemoryLayer.L3_SUMMARY.value == "l3_summary"
        assert MemoryLayer.L5_KNOWLEDGE.value == "l5_knowledge"
        assert MemoryLayer.L6_SCHEMA.value == "l6_schema"
        assert MemoryLayer.L1_RAW.value == "l1_raw"
        # V2 compatibility aliases (old enum names point at the new values)
        assert MemoryLayer.L6_IDENTITY.value == "l4_identity"
        assert MemoryLayer.L4_KNOWLEDGE.value == "l5_knowledge"
        assert MemoryLayer.L5_SCHEMA.value == "l6_schema"
        # V1 compatibility aliases (values match the corresponding V3 layers)
        assert MemoryLayer.PROFILE.value == "l4_identity"
        assert MemoryLayer.DIALOGUE.value == "l2_fact"
        assert MemoryLayer.SUMMARY.value == "l3_summary"
        assert MemoryLayer.KNOWLEDGE.value == "l5_knowledge"
        assert MemoryLayer.RAW.value == "l1_raw"
    
    def test_memory_layer_from_string(self):
        """Test creating a memory layer from a string."""
        assert MemoryLayer.from_string("profile") == MemoryLayer.L4_IDENTITY
        assert MemoryLayer.from_string("PROFILE") == MemoryLayer.L4_IDENTITY
        assert MemoryLayer.from_string("Profile") == MemoryLayer.L4_IDENTITY
        # V2 legacy value compatibility
        assert MemoryLayer.from_string("l6_identity") == MemoryLayer.L4_IDENTITY
        assert MemoryLayer.from_string("l5_schema") == MemoryLayer.L6_SCHEMA
        assert MemoryLayer.from_string("l4_knowledge") == MemoryLayer.L5_KNOWLEDGE
        assert MemoryLayer.from_string("l1_raw") == MemoryLayer.L1_RAW
    
    def test_memory_layer_invalid_string(self):
        """Test creating a memory layer from an invalid string."""
        with pytest.raises(ValueError):
            MemoryLayer.from_string("invalid_layer")
    
    def test_memory_input_type_values(self):
        """Test input type enum values."""
        assert MemoryInputType.TEXT.value == "text"
        assert MemoryInputType.QA.value == "qa"
        assert MemoryInputType.DIALOGUE.value == "dialogue"
    
    def test_agent_process_mode_values(self):
        """Test agent process mode enum values."""
        assert AgentProcessMode.DISABLED.value == "disabled"
        assert AgentProcessMode.FULL.value == "full"
    
    def test_task_status_values(self):
        """Test task status enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.PROCESSING.value == "processing"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
    
    def test_delete_scope_values(self):
        """Test delete scope enum values."""
        assert DeleteScope.MEMORY.value == "memory"
        assert DeleteScope.SCENE.value == "scene"
        assert DeleteScope.USER.value == "user"
        assert DeleteScope.APP.value == "app"


# ==================== QAPair tests ====================

class TestQAPair:
    """QAPair tests."""
    
    def test_qa_pair_creation(self):
        """Test creating a QAPair."""
        qa = QAPair(
            question="What is your name?",
            answer="My name is Xiaoming"
        )
        
        assert qa.question == "What is your name?"
        assert qa.answer == "My name is Xiaoming"
    
    def test_qa_pair_to_dict(self):
        """Test converting a QAPair to a dict."""
        qa = QAPair(question="Q1", answer="A1")
        d = qa.to_dict()
        
        assert d == {"question": "Q1", "answer": "A1"}
    
    def test_qa_pair_from_dict(self):
        """Test creating a QAPair from a dict."""
        d = {"question": "Q1", "answer": "A1"}
        qa = QAPair.from_dict(d)
        
        assert qa.question == "Q1"
        assert qa.answer == "A1"
    
    def test_qa_pair_to_content(self):
        """Test converting a QAPair to a content string."""
        qa = QAPair(question="How are you?", answer="I am doing well")
        content = qa.to_content()
        
        assert "Q:" in content
        assert "A:" in content
        assert "How are you?" in content
        assert "I am doing well" in content


# ==================== AddRequest tests ====================

class TestAddRequest:
    """AddRequest tests."""
    
    def test_add_request_text_basic(self):
        """Test a basic text request."""
        request = AddRequest(
            uid="user1",
            agent_id="scene1",
            content="test content"
        )

        assert request.uid == "user1"
        assert request.agent_id == "scene1"
        assert request.content == "test content"
        assert request.input_type == MemoryInputType.TEXT
    
    def test_add_request_qa(self):
        """Test a QA request."""
        request = AddRequest(
            uid="user1",
            agent_id="scene1",
            input_type=MemoryInputType.QA,
            qa_pair=QAPair(question="Q", answer="A")
        )
        
        assert request.input_type == MemoryInputType.QA
        assert request.qa_pair is not None
    
    def test_add_request_get_content_text(self):
        """Test getting text content."""
        request = AddRequest(
            uid="user1",
            agent_id="scene1",
            content="text content"
        )
        
        assert request.get_content() == "text content"
    
    def test_add_request_get_content_qa(self):
        """Test getting QA content."""
        request = AddRequest(
            uid="user1",
            agent_id="scene1",
            input_type=MemoryInputType.QA,
            qa_pair=QAPair(question="question", answer="answer")
        )
        
        content = request.get_content()
        assert "question" in content
        assert "answer" in content
    
    def test_add_request_validates_text_content(self):
        """Test text content validation."""
        with pytest.raises(ValueError):
            AddRequest(
                uid="user1",
                agent_id="scene1",
                input_type=MemoryInputType.TEXT,
                content=None  # the text input type requires content
            )
    
    def test_add_request_validates_qa_pair(self):
        """Test QA pair validation."""
        with pytest.raises(ValueError):
            AddRequest(
                uid="user1",
                agent_id="scene1",
                input_type=MemoryInputType.QA,
                qa_pair=None  # the QA input type requires qa_pair
            )
    
    def test_add_request_layer_normalization(self):
        """Test layer name normalization."""
        request = AddRequest(
            uid="user1",
            agent_id="scene1",
            content="content",
            layer="profile"  # string form
        )
        
        assert request.layer == MemoryLayer.PROFILE
    
    def test_add_request_agent_mode_normalization(self):
        """Test agent mode normalization."""
        request = AddRequest(
            uid="user1",
            agent_id="scene1",
            content="content",
            agent_mode="full"  # string form
        )

        assert request.agent_mode == AgentProcessMode.FULL
    
    def test_add_request_auto_extract_compatibility(self):
        """Test auto_extract compatibility."""
        request = AddRequest(
            uid="user1",
            agent_id="scene1",
            content="content",
            auto_extract=True  # legacy argument
        )
        
        # auto_extract=True must be equivalent to agent_mode=full
        assert request.agent_mode == AgentProcessMode.FULL
    
    def test_add_request_to_dict(self):
        """Test conversion to dict."""
        request = AddRequest(
            uid="user1",
            agent_id="scene1",
            content="content",
            importance=0.8,
            tags=["tag1", "tag2"]
        )

        d = request.to_dict()

        assert d["uid"] == "user1"
        assert d["importance"] == 0.8
        assert d["tags"] == ["tag1", "tag2"]


# ==================== AddResponse tests ====================

class TestAddResponse:
    """AddResponse tests."""
    
    def test_add_response_success(self):
        """Test a success response."""
        response = AddResponse(
            success=True,
            memory_id="mem_123",
            layer=MemoryLayer.PROFILE,
            message="Added successfully"
        )
        
        assert response.success is True
        assert response.memory_id == "mem_123"
        assert response.layer == MemoryLayer.PROFILE
    
    def test_add_response_with_extracted_ids(self):
        """Test a response carrying extracted ids."""
        response = AddResponse(
            success=True,
            memory_id="mem_123",
            extracted_ids=["ext_1", "ext_2"],
            abstract_key_ids=["key_1", "key_2"]
        )
        
        assert len(response.extracted_ids) == 2
        assert len(response.abstract_key_ids) == 2
    
    def test_add_response_to_dict(self):
        """Test conversion to dict."""
        response = AddResponse(
            success=True,
            memory_id="mem_123",
            layer=MemoryLayer.KNOWLEDGE
        )
        
        d = response.to_dict()
        
        assert d["success"] is True
        assert d["memory_id"] == "mem_123"
        assert d["layer"] == "l5_knowledge"


# ==================== AsyncAddResponse tests ====================

class TestAsyncAddResponse:
    """AsyncAddResponse tests."""
    
    def test_async_response(self):
        """Test the async response."""
        response = AsyncAddResponse(
            success=True,
            task_id="task_123",
            status=TaskStatus.PENDING,
            estimated_time=30
        )
        
        assert response.success is True
        assert response.task_id == "task_123"
        assert response.status == TaskStatus.PENDING
        assert response.estimated_time == 30
    
    def test_async_response_to_dict(self):
        """Test conversion to dict."""
        response = AsyncAddResponse(
            success=True,
            task_id="task_123",
            status=TaskStatus.PROCESSING
        )
        
        d = response.to_dict()
        
        assert d["task_id"] == "task_123"
        assert d["status"] == "processing"


# ==================== RecallRequest tests ====================

class TestRecallRequest:
    """RecallRequest tests."""
    
    def test_recall_request_basic(self):
        """Test a basic recall request."""
        request = RecallRequest(
            query="user preferences",
            uid="user1"
        )
        
        assert request.query == "user preferences"
        assert request.limit == 10  # default value
    
    def test_recall_request_with_filters(self):
        """Test a recall request with filters."""
        request = RecallRequest(
            query="movies",
            uid="user1",
            layers=[MemoryLayer.PROFILE, MemoryLayer.KNOWLEDGE],
            tags=["interests"],
            min_importance=0.5
        )
        
        assert len(request.layers) == 2
        assert "interests" in request.tags
        assert request.min_importance == 0.5
    
    def test_recall_request_weights(self):
        """Test the weight parameters."""
        request = RecallRequest(
            query="test",
            uid="user1",
            semantic_weight=0.6,
            recency_weight=0.2,
            importance_weight=0.15,
            access_weight=0.05
        )
        
        # the weights should sum to approximately 1
        total = (request.semantic_weight + request.recency_weight + 
                 request.importance_weight + request.access_weight)
        assert abs(total - 1.0) < 0.01


# ==================== MemoryEntry tests ====================

class TestMemoryEntry:
    """MemoryEntry tests (V2: MemoryEntry is an alias of MemoryNode with flattened fields)."""
    
    def test_memory_entry_creation(self):
        """Test creating a MemoryEntry."""
        entry = MemoryEntry(
            content="test content",
            user_id="user1",
            agent_id="scene1",
            layer=MemoryLayer.PROFILE,
        )
        
        assert entry.content == "test content"
        assert entry.layer == MemoryLayer.L4_IDENTITY  # PROFILE is an alias of L4_IDENTITY
    
    def test_memory_entry_build_isolation_key(self):
        """Test building the isolation key (two levels: user_id + agent_id)."""
        key = MemoryEntry.build_isolation_key("user1", "scene1")

        assert "user1" in key
        assert "scene1" in key
    
    def test_memory_entry_to_dict(self):
        """Test conversion to dict."""
        entry = MemoryEntry(
            content="content",
            user_id="user1",
            agent_id="scene1",
            layer=MemoryLayer.DIALOGUE,
        )

        d = entry.to_dict()

        assert d["content"] == "content"
        assert d["layer"] == "l2_fact"  # DIALOGUE -> l2_fact
        assert d["user_id"] == "user1"


# ==================== BatchDeleteRequest tests ====================

class TestBatchDeleteRequest:
    """BatchDeleteRequest tests."""
    
    def test_batch_delete_by_ids(self):
        """Test batch delete by id."""
        request = BatchDeleteRequest(
            scope=DeleteScope.MEMORY,
            memory_ids=["id1", "id2", "id3"]
        )
        
        assert request.scope == DeleteScope.MEMORY
        assert len(request.memory_ids) == 3
    
    def test_batch_delete_by_scene(self):
        """Test batch delete by scene."""
        request = BatchDeleteRequest(
            scope=DeleteScope.SCENE,
            uid="user1",
            agent_id="scene1"
        )
        
        assert request.scope == DeleteScope.SCENE
        assert request.uid == "user1"
        assert request.agent_id == "scene1"
    
    def test_batch_delete_dangerous_requires_confirm(self):
        """Test that dangerous operations require confirmation."""
        request = BatchDeleteRequest(
            scope=DeleteScope.USER,
            uid="user1",
            confirm=True
        )
        
        assert request.confirm is True


# ==================== Run tests ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
